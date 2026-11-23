// 러스트 코드 코퍼스 (analyze-code): code_crates에 적은 크레이트만 crates.io에서 최신 안정 버전으로 받아(의존성은 받지 않음)
// 주석 내용을 빼고 키 연속을 셈.
// - 받은 크레이트는 <result_path>/crates/<이름>-<버전>/에 풀어 두고, 같은 버전이 있으면 다시 받지 않음.
//   인터넷이 안 되면 받아 둔 가장 새 버전을 씀
// - 패키지 본체 코드를 모두 읽음: tests·benches·examples·fuzz 폴더와 build.rs, 200KB가 넘는 파일(생성된 표 등),
//   숨김·target 폴더는 뺌 (src 안의 단위 테스트는 들어감)
// - 크레이트마다 같은 무게: 크레이트별로 센 뒤 글자 수가 같아지게 늘리거나 줄여서 더함. 큰 크레이트 몇 개가
//   통계를 독차지하지 않고, 크레이트의 일부만 잘라 읽지도 않음
// - 주석은 내용을 빼고 표시(//, ///, //!)만 주석 블록당 한 번 남김 (이어지는 줄은 에디터가 이어 쓰는 경우가 많고,
//   내용은 영어·한국어 문장이라 코드의 기호 통계를 흐림). 문자열은 그대로 (형식 문자열 "{}" 등도 실제로 침)
#![allow(clippy::cast_precision_loss)]
use std::io::Read;
use std::path::{Path, PathBuf};
use anyhow::Context;
use rayon::prelude::*;
use crate::keycorpus::{self, KeyCorpus};

const MAX_FILE_BYTES: u64 = 200_000;
// 크레이트 하나를 맞출 글자 수 (반올림 오차가 무시될 만큼 크게)
const CRATE_LETTERS: f64 = 1e9;
const INDEX_URL: &str = "https://index.crates.io";
const DOWNLOAD_URL: &str = "https://static.crates.io/crates";

fn is_ident(b: u8) -> bool { b.is_ascii_alphanumeric() || b == b'_' }

// UTF-8 첫 바이트로 문자 길이
fn utf8_len(b: u8) -> usize {
  match b { 0xF0..=0xFF => 4, 0xE0..=0xEF => 3, 0xC0..=0xDF => 2, _ => 1 }
}

// 주석 내용을 지운 소스. 줄 주석은 표시만, 블록 주석은 통째로(줄바꿈은 남겨 줄 구조 유지)
pub fn strip_comments(src: &str) -> String {
  let b = src.as_bytes();
  let mut out: Vec<u8> = Vec::with_capacity(b.len());
  let mut i = 0;
  while i < b.len() {
    let rest = &b[i..];
    if rest.starts_with(b"//") {
      let mark = if rest.get(2) == Some(&b'/') && rest.get(3) != Some(&b'/') { &b"///"[..] }
        else if rest.get(2) == Some(&b'!') { &b"//!"[..] } else { &b"//"[..] };
      out.extend_from_slice(mark);
      i += rest.iter().position(|&c| c == b'\n').unwrap_or(rest.len());
      continue;
    }
    if rest.starts_with(b"/*") {
      let mut depth = 0usize;
      let mut j = 0;
      while j < rest.len() {
        if rest[j..].starts_with(b"/*") { depth += 1; j += 2; continue; }
        if rest[j..].starts_with(b"*/") { depth -= 1; j += 2; if depth == 0 { break; } continue; }
        if rest[j] == b'\n' { out.push(b'\n'); }
        j += 1;
      }
      i += j;
      continue;
    }
    let prev_ident = i > 0 && is_ident(b[i - 1]);
    // 날 문자열: r"..", r#".."#, br#".."# (앞이 식별자의 일부가 아닐 때)
    let raw_at = if rest[0] == b'r' && !prev_ident { Some(1) }
      else if rest.starts_with(b"br") && !prev_ident { Some(2) } else { None };
    if let Some(k) = raw_at {
      let hashes = rest[k..].iter().take_while(|&&c| c == b'#').count();
      if rest.get(k + hashes) == Some(&b'"') {
        let close: Vec<u8> = std::iter::once(b'"').chain(std::iter::repeat_n(b'#', hashes)).collect();
        let body = k + hashes + 1;
        let end = rest[body..].windows(close.len()).position(|w| w == close.as_slice()).map_or(rest.len(), |p| body + p + close.len());
        out.extend_from_slice(&rest[..end]);
        i += end;
        continue;
      }
    }
    if rest[0] == b'"' {
      let mut j = 1;
      while j < rest.len() && rest[j] != b'"' { j += if rest[j] == b'\\' { 2 } else { 1 }; }
      let end = (j + 1).min(rest.len());
      out.extend_from_slice(&rest[..end]);
      i += end;
      continue;
    }
    // 문자 리터럴('a', '"', '\n', '\u{..}')은 통째로 넘김 (안의 " 를 문자열 시작으로 오인하지 않게). 아니면 수명·레이블
    if rest[0] == b'\'' {
      let lit_end = if rest.get(1) == Some(&b'\\') {
        rest.iter().take(12).skip(3).position(|&c| c == b'\'').map(|p| p + 4)
      } else {
        rest.get(1).map(|&c| 1 + utf8_len(c)).filter(|&q| rest.get(q) == Some(&b'\'')).map(|q| q + 1)
      };
      if let Some(e) = lit_end {
        out.extend_from_slice(&rest[..e]);
        i += e;
        continue;
      }
    }
    out.push(rest[0]);
    i += 1;
  }
  // 주석만 있는 줄이 이어지면 첫 줄의 표시만 남김
  let text = String::from_utf8_lossy(&out).into_owned();
  let mut lines: Vec<&str> = Vec::new();
  let mut prev_comment = false;
  for line in text.split('\n') {
    let comment = line.trim_start().starts_with("//");
    lines.push(if comment && prev_comment { "" } else { line });
    prev_comment = comment;
  }
  lines.join("\n")
}

// crates.io 색인(sparse index)에서 크레이트 파일의 경로: 1/a, 2/ab, 3/a/abc, ab/cd/abcd...
pub fn index_path(name: &str) -> String {
  let n = name.to_ascii_lowercase();
  match n.len() {
    1 => format!("1/{n}"),
    2 => format!("2/{n}"),
    3 => format!("3/{}/{n}", &n[..1]),
    _ => format!("{}/{}/{n}", &n[..2], &n[2..4]),
  }
}

// 버전 비교용 숫자 (1.0.210 → [1, 0, 210]). 빌드 정보(+...)는 무시
fn version_key(v: &str) -> Vec<u64> {
  v.split('+').next().unwrap_or(v).split('.').map(|x| x.parse().unwrap_or(0)).collect()
}

// 색인 파일(버전마다 JSON 한 줄)에서 회수(yank)되지 않은 가장 새 안정 버전의 (이름, 버전)
pub fn latest_stable(index: &str) -> Option<(String, String)> {
  index.lines()
    .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
    .filter(|v| v["yanked"].as_bool() != Some(true))
    .filter_map(|v| Some((v["name"].as_str()?.to_string(), v["vers"].as_str()?.to_string())))
    .filter(|(_, vers)| !vers.contains('-'))
    .max_by_key(|(_, vers)| version_key(vers))
}

// dir에 받아 둔 이 크레이트의 가장 새 버전 폴더
fn newest_local(name: &str, dir: &Path) -> Option<(String, PathBuf)> {
  let prefix = format!("{name}-");
  std::fs::read_dir(dir).ok()?.filter_map(Result::ok)
    .filter_map(|e| {
      let f = e.file_name().to_string_lossy().into_owned();
      let vers = f.strip_prefix(&prefix)?.to_string();
      (vers.starts_with(|c: char| c.is_ascii_digit()) && e.path().is_dir()).then(|| (vers, e.path()))
    })
    .max_by_key(|(vers, _)| version_key(vers))
}

// .crate(gzip으로 묶은 tar)를 dir에 풂. 안의 <이름>-<버전>/ 폴더가 그대로 생김.
// 중간에 끊겨도 반쯤 풀린 폴더가 남지 않게 임시 폴더에 풀고 옮김
pub fn unpack(bytes: &[u8], dir: &Path, folder: &str) -> anyhow::Result<PathBuf> {
  let tmp = dir.join(format!(".tmp-{folder}"));
  let _ = std::fs::remove_dir_all(&tmp);
  std::fs::create_dir_all(&tmp)?;
  tar::Archive::new(flate2::read::GzDecoder::new(bytes)).unpack(&tmp)?;
  let target = dir.join(folder);
  std::fs::rename(tmp.join(folder), &target).with_context(|| format!("the archive has no {folder}/ folder"))?;
  std::fs::remove_dir_all(&tmp)?;
  Ok(target)
}

// 이름마다 최신 안정 버전을 받아 dir/<이름>-<버전>/에 풀고 (이름-버전, 폴더)를 돌려줌. 의존성은 받지 않음
pub fn fetch_crates(names: &[String], dir: &Path) -> anyhow::Result<Vec<(String, PathBuf)>> {
  std::fs::create_dir_all(dir)?;
  let client = crate::download::http_client()?;
  let get = |url: &str| -> anyhow::Result<Vec<u8>> {
    let mut body = Vec::new();
    client.get(url).send()?.error_for_status()?.read_to_end(&mut body)?;
    Ok(body)
  };
  let mut out = Vec::new();
  for name in names {
    let latest = get(&format!("{INDEX_URL}/{}", index_path(name)))
      .with_context(|| format!("looking up {name} on crates.io"))
      .and_then(|idx| latest_stable(&String::from_utf8_lossy(&idx)).ok_or_else(|| anyhow::anyhow!("{name} has no stable version on crates.io")));
    let (label, path) = match latest {
      Ok((real, vers)) => {
        let folder = format!("{real}-{vers}");
        let path = dir.join(&folder);
        if !path.is_dir() {
          println!("[Code Corpus] downloading {folder}");
          let bytes = get(&format!("{DOWNLOAD_URL}/{real}/{folder}.crate")).with_context(|| format!("downloading {folder}"))?;
          unpack(&bytes, dir, &folder)?;
        }
        (folder, path)
      }
      Err(e) => {
        let (vers, path) = newest_local(name, dir).ok_or(e)?;
        println!("[Code Corpus] warning: could not reach crates.io for {name}; using the downloaded {name}-{vers}");
        (format!("{name}-{vers}"), path)
      }
    };
    out.push((label, path));
  }
  Ok(out)
}

// 크레이트 폴더 아래 본체 코드 .rs 파일을 경로 순서로 (숨김·target 폴더, 패키지 맨 위의 tests·benches·examples·fuzz
// 폴더와 build.rs, 200KB가 넘는 파일은 뺌). top: 크레이트 폴더 바로 아래인지
fn rs_files(dir: &Path, top: bool, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
  let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(Result::ok).collect();
  entries.sort_by_key(std::fs::DirEntry::file_name);
  for e in entries {
    let name = e.file_name().to_string_lossy().into_owned();
    let Ok(ft) = e.file_type() else { continue };
    if ft.is_dir() {
      if name.starts_with('.') || name == "target" || (top && matches!(name.as_str(), "tests" | "benches" | "examples" | "fuzz")) { continue; }
      rs_files(&e.path(), false, out)?;
    } else if ft.is_file() && name.ends_with(".rs") && !(top && name == "build.rs") && e.metadata().is_ok_and(|m| m.len() <= MAX_FILE_BYTES) {
      out.push(e.path());
    }
  }
  Ok(())
}

// 글자(a-z) 수가 CRATE_LETTERS가 되게 모든 수를 같은 비율로 늘리거나 줄임
fn equal_weight(k: &KeyCorpus) -> KeyCorpus {
  let letters: u64 = k.unigrams[..26].iter().sum();
  let mut out = KeyCorpus::new();
  if letters == 0 { return out; }
  #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
  let scale = |v: &mut [u64], from: &[u64]| for (a, &b) in v.iter_mut().zip(from) { *a = (b as f64 * CRATE_LETTERS / letters as f64).round() as u64; };
  scale(&mut out.unigrams, &k.unigrams);
  scale(&mut out.bigrams, &k.bigrams);
  scale(&mut out.trigrams, &k.trigrams);
  out
}

pub struct CodeScan {
  // 크레이트마다 (이름-버전, 읽은 바이트)
  pub crates: Vec<(String, u64)>,
  pub files: usize,
}

// 크레이트 폴더들을 셈: 크레이트마다 본체 코드를 모두 세고, 크레이트끼리 같은 무게로 더함
pub fn analyze_crates(crates: &[(String, PathBuf)]) -> anyhow::Result<(KeyCorpus, CodeScan)> {
  let per: Vec<(KeyCorpus, usize, u64)> = crates.par_iter().map(|(_, dir)| -> anyhow::Result<_> {
    let mut files = Vec::new();
    rs_files(dir, true, &mut files).with_context(|| format!("reading {}", dir.display()))?;
    let mut k = KeyCorpus::new();
    let (mut n, mut bytes) = (0usize, 0u64);
    for f in &files {
      let Ok(code) = std::fs::read(f) else { continue };
      let text = strip_comments(&String::from_utf8_lossy(&code));
      keycorpus::count_text(text.as_bytes(), &mut k);
      n += 1;
      bytes += text.len() as u64;
    }
    Ok((equal_weight(&k), n, bytes))
  }).collect::<anyhow::Result<_>>()?;
  let mut total = KeyCorpus::new();
  let mut scan = CodeScan { crates: Vec::new(), files: 0 };
  for ((label, _), (k, n, bytes)) in crates.iter().zip(per) {
    total.merge(&k);
    scan.files += n;
    scan.crates.push((label.clone(), bytes));
  }
  Ok((total, scan))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn comments_keep_only_the_marker_once_per_block() {
    let src = "fn a() {} // trailing note\n/// Docs line one\n/// line two\n//! inner\nlet x = 1; /* block\nstill */ let y = 2;\n//// not a doc\n";
    let s = strip_comments(src);
    let lines: Vec<&str> = s.split('\n').collect();
    assert_eq!(lines[0], "fn a() {} //");
    // 주석만 있는 줄이 이어지면(///, ///, //!) 첫 줄의 표시만
    assert_eq!((lines[1], lines[2], lines[3]), ("///", "", ""));
    // 블록 주석은 지우고 줄바꿈은 남김
    assert_eq!((lines[4], lines[5]), ("let x = 1; ", " let y = 2;"));
    assert_eq!(lines[6], "//");
  }

  #[test]
  fn strings_and_char_literals_are_not_comments() {
    let src = r##"let u = "http://x"; let q = '"'; let r = r#"a // b"#; f::<'a>(); let e = '\''; let c = "\" // in";"##;
    assert_eq!(strip_comments(src), src);
    // 수명 뒤의 주석은 지워짐
    assert_eq!(strip_comments("fn f<'a>(x: &'a str) // c"), "fn f<'a>(x: &'a str) //");
    // 중첩 블록 주석
    assert_eq!(strip_comments("a /* x /* y */ z */ b"), "a  b");
    // 유니코드 문자 리터럴
    assert_eq!(strip_comments("let c = 'é'; // c"), "let c = 'é'; //");
  }

  #[test]
  fn index_paths_and_latest_stable_version() {
    assert_eq!((index_path("a"), index_path("ab"), index_path("syn"), index_path("Serde")), ("1/a".into(), "2/ab".into(), "3/s/syn".into(), "se/rd/serde".into()));
    let idx = r#"{"name":"foo","vers":"1.2.0","yanked":false}
{"name":"foo","vers":"1.10.0","yanked":false}
{"name":"foo","vers":"2.0.0-rc.1","yanked":false}
{"name":"foo","vers":"1.11.0","yanked":true}
{"name":"foo","vers":"1.9.9+build","yanked":false}"#;
    // 1.10.0 > 1.9.9 (숫자로 비교), 회수된 1.11.0과 시험판 2.0.0-rc.1은 뺌
    assert_eq!(latest_stable(idx), Some(("foo".into(), "1.10.0".into())));
    assert_eq!(latest_stable("{\"name\":\"foo\",\"vers\":\"0.1.0-alpha\",\"yanked\":false}"), None);
  }

  #[test]
  fn unpack_and_analyze_weight_each_crate_equally() {
    let dir = std::env::temp_dir().join(format!("p2anut-code-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // .crate 흉내: foo-1.0.0/ 아래 본체 코드, 벤치·시험 코드, build.rs, 생성된 큰 파일
    let mut ar = tar::Builder::new(flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast()));
    for (path, body) in [("foo-1.0.0/benches/b.rs", "bench(x);".repeat(10)), ("foo-1.0.0/src/lib.rs", "fn foo() -> u8 { a(b) } // note".to_string()),
                         ("foo-1.0.0/tests/t.rs", "only_test();".to_string()), ("foo-1.0.0/build.rs", "only_build();".to_string()),
                         ("foo-1.0.0/src/big.rs", "x".repeat(MAX_FILE_BYTES as usize + 1))] {
      let mut h = tar::Header::new_gnu();
      h.set_size(body.len() as u64);
      h.set_mode(0o644);
      h.set_cksum();
      ar.append_data(&mut h, path, body.as_bytes()).unwrap();
    }
    let bytes = ar.into_inner().unwrap().finish().unwrap();
    let foo = unpack(&bytes, &dir, "foo-1.0.0").unwrap();
    assert!(foo.join("src/lib.rs").is_file() && !dir.join(".tmp-foo-1.0.0").exists());
    std::fs::create_dir_all(dir.join("bar-0.2.0/src")).unwrap();
    std::fs::write(dir.join("bar-0.2.0/src/main.rs"), "let s = \"{}\";").unwrap();
    // 받아 둔 버전 찾기 (인터넷이 안 될 때)
    std::fs::create_dir_all(dir.join("bar-0.10.0")).unwrap();
    assert_eq!(newest_local("bar", &dir).unwrap().0, "0.10.0");
    let crates = vec![("foo-1.0.0".to_string(), foo), ("bar-0.2.0".to_string(), dir.join("bar-0.2.0"))];
    let (k, scan) = analyze_crates(&crates).unwrap();
    // foo는 src/lib.rs만 (벤치·시험·build.rs·큰 파일은 안 읽음), bar는 src/main.rs
    assert_eq!(scan.files, 2);
    let id = |c: char| keycorpus::char_key(c).unwrap() as usize;
    assert_eq!((k.unigrams[id('h')], k.unigrams[id('x')], k.unigrams[id('y')]), (0, 0, 0));
    // 크레이트마다 글자 수를 같게 맞춰서 더함: foo는 글자 8개에 { 하나, bar는 글자 4개에 { 하나
    assert_eq!(k.unigrams[..26].iter().sum::<u64>(), 2 * CRATE_LETTERS as u64);
    assert_eq!(k.unigrams[id('{')], (CRATE_LETTERS / 8.0 + CRATE_LETTERS / 4.0) as u64);
    assert_eq!(k.bigrams[id('"') * keycorpus::KEYS + id('{')], (CRATE_LETTERS / 4.0) as u64);
    std::fs::remove_dir_all(&dir).unwrap();
  }
}
