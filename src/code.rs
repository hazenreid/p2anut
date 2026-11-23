// 러스트 코드 코퍼스 (analyze-code): code_dirs 아래의 .rs 파일에서 주석 내용을 빼고 키 연속을 셈.
// - 크레이트는 Cargo.toml이 있는 폴더 단위. 이름 끝의 버전(-1.2.3)을 떼고 같은 이름은 한 크레이트로 봄 (여러 버전이 받아져 있어도 한 번)
// - 크레이트마다 code_crate_cap 바이트까지만 읽음 (경로 순서). 큰 크레이트 몇 개가 통계를 독차지하지 않게
// - code_exclude에 맞는 크레이트(자동 생성된 바인딩 등)와 200KB가 넘는 파일(생성된 표 등)은 뺌
// - 숨김 폴더와 target 폴더는 들어가지 않음
// - 주석은 내용을 빼고 표시(//, ///, //!)만 주석 블록당 한 번 남김 (이어지는 줄은 에디터가 이어 쓰는 경우가 많고,
//   내용은 영어·한국어 문장이라 코드의 기호 통계를 흐림). 문자열은 그대로 (형식 문자열 "{}" 등도 실제로 침)
#![allow(clippy::cast_precision_loss)]
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use crate::keycorpus::{self, KeyCorpus};

const MAX_FILE_BYTES: u64 = 200_000;

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

// 폴더 이름에서 끝의 버전을 뗀 크레이트 이름 (serde-1.0.210 → serde, tokio-macros-2.1.0 → tokio-macros)
pub fn crate_name(dir: &str) -> &str {
  match dir.rmatch_indices('-').find(|(i, _)| dir[i + 1..].starts_with(|c: char| c.is_ascii_digit())) {
    Some((i, _)) if i > 0 => &dir[..i],
    _ => dir,
  }
}

// "abc" 정확히, "abc*" 앞부분, "*abc" 끝부분
pub fn excluded(name: &str, patterns: &[String]) -> bool {
  patterns.iter().any(|p| match (p.strip_prefix('*'), p.strip_suffix('*')) {
    (Some(suffix), _) => name.ends_with(suffix),
    (_, Some(prefix)) => name.starts_with(prefix),
    _ => name == p,
  })
}

pub fn expand_home(dir: &str) -> PathBuf {
  match (dir.strip_prefix("~/"), std::env::var_os("HOME")) {
    (Some(rest), Some(home)) => Path::new(&home).join(rest),
    _ if dir == "~" => std::env::var_os("HOME").map_or_else(|| PathBuf::from(dir), PathBuf::from),
    _ => PathBuf::from(dir),
  }
}

struct Crate {
  name: String,
  files: Vec<PathBuf>,
}

// 폴더를 돌며 (크레이트 이름, .rs 파일)을 경로 순서대로 모음. crate_of: 지금 폴더가 속한 크레이트 이름
fn walk(dir: &Path, crate_of: &str, files: &mut Vec<(String, PathBuf)>) -> std::io::Result<()> {
  let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(Result::ok).collect();
  entries.sort_by_key(std::fs::DirEntry::file_name);
  let here = if entries.iter().any(|e| e.file_name() == "Cargo.toml") {
    crate_name(&dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()).to_string()
  } else { crate_of.to_string() };
  for e in entries {
    let name = e.file_name().to_string_lossy().into_owned();
    let Ok(ft) = e.file_type() else { continue };
    if ft.is_dir() {
      if name.starts_with('.') || name == "target" { continue; }
      walk(&e.path(), &here, files)?;
    } else if ft.is_file() && name.ends_with(".rs") && e.metadata().is_ok_and(|m| m.len() <= MAX_FILE_BYTES) {
      files.push((here.clone(), e.path()));
    }
  }
  Ok(())
}

pub struct CodeScan {
  pub crates: usize,
  pub excluded: usize,
  pub files: usize,
  pub bytes: u64,
}

pub fn analyze_code(dirs: &[String], exclude: &[String], cap: usize) -> anyhow::Result<(KeyCorpus, CodeScan)> {
  let mut all: Vec<(String, PathBuf)> = Vec::new();
  for d in dirs {
    let path = expand_home(d);
    anyhow::ensure!(path.is_dir(), "code_dirs entry {} is not a folder", path.display());
    let top = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    walk(&path, crate_name(&top), &mut all)?;
  }
  // 같은 이름(다른 버전·다른 폴더)은 하나로: 경로 순서대로 이어 붙이고 상한은 함께 적용
  let mut merged: Vec<Crate> = Vec::new();
  let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
  for (name, file) in all {
    let k = *index.entry(name.clone()).or_insert_with(|| { merged.push(Crate { name, files: Vec::new() }); merged.len() - 1 });
    merged[k].files.push(file);
  }
  let total = merged.len();
  merged.retain(|c| !excluded(&c.name, exclude));
  let skipped = total - merged.len();
  let (corpus, files, bytes) = merged.par_iter()
    .map(|c| {
      let mut k = KeyCorpus::new();
      let (mut files, mut bytes) = (0usize, 0u64);
      for f in &c.files {
        if bytes >= cap as u64 { break; }
        let Ok(src) = std::fs::read(f) else { continue };
        let text = strip_comments(&String::from_utf8_lossy(&src));
        keycorpus::count_text(text.as_bytes(), &mut k);
        files += 1;
        bytes += text.len() as u64;
      }
      (k, files, bytes)
    })
    .reduce(|| (KeyCorpus::new(), 0, 0), |mut a, b| { a.0.merge(&b.0); (a.0, a.1 + b.1, a.2 + b.2) });
  Ok((corpus, CodeScan { crates: merged.len(), excluded: skipped, files, bytes }))
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
  fn crate_names_drop_versions_and_match_exclusions() {
    assert_eq!(crate_name("serde-1.0.210"), "serde");
    assert_eq!(crate_name("tokio-macros-2.1.0"), "tokio-macros");
    assert_eq!(crate_name("utf-8-0.7.6"), "utf-8");
    assert_eq!(crate_name("x25519-dalek-2.0.0"), "x25519-dalek");
    assert_eq!(crate_name("p2anut"), "p2anut");
    let pats: Vec<String> = ["windows*", "*-sys", "libc"].iter().map(|s| (*s).to_string()).collect();
    assert!(excluded("windows_x86_64_gnu", &pats) && excluded("openssl-sys", &pats) && excluded("libc", &pats));
    assert!(!excluded("libcrux", &pats) && !excluded("serde", &pats));
  }

  #[test]
  fn analyze_code_groups_by_crate_and_caps_each() {
    let dir = std::env::temp_dir().join(format!("p2anut-code-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let make = |rel: &str, body: &str| {
      let p = dir.join(rel);
      std::fs::create_dir_all(p.parent().unwrap()).unwrap();
      std::fs::write(p, body).unwrap();
    };
    make("reg/foo-1.0.0/Cargo.toml", "");
    make("reg/foo-1.0.0/src/lib.rs", "fn foo() -> u8 { a(b); } // note");
    make("reg/foo-2.0.0/Cargo.toml", "");
    make("reg/foo-2.0.0/src/lib.rs", &"x(y);\n".repeat(100));
    make("reg/bar-sys-0.1.0/Cargo.toml", "");
    make("reg/bar-sys-0.1.0/src/lib.rs", "extern { fn bar(); }");
    make("reg/baz-0.3.0/Cargo.toml", "");
    make("reg/baz-0.3.0/target/debug/gen.rs", "ignored();");
    make("reg/baz-0.3.0/src/main.rs", "let s = \"{}\";");
    let pats = vec!["*-sys".to_string()];
    let (k, scan) = analyze_code(&[dir.join("reg").to_string_lossy().into_owned()], &pats, 10).unwrap();
    // foo의 두 버전은 한 크레이트 (상한 10바이트라 첫 파일만), bar-sys는 제외, target 폴더는 안 읽음
    assert_eq!((scan.crates, scan.excluded, scan.files), (2, 1, 2));
    let id = |c: char| keycorpus::char_key(c).unwrap() as usize;
    assert_eq!(k.unigrams[id('x')], 0);
    assert_eq!(k.unigrams[id('{')], 2);
    assert_eq!(k.bigrams[id('"') * keycorpus::KEYS + id('{')], 1);
    std::fs::remove_dir_all(&dir).unwrap();
  }
}
