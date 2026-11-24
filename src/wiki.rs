// 위키 글자 코퍼스 (analyze): 덤프를 받고(없으면) 순차(스트리밍·앞쪽 N개) 또는 멀티스트림 덤프 파일의 병렬 분석
// (중간 저장·이어서 분석)으로 셈. 문서 고르기와 마크업 지우기는 dump.rs
//
// 멀티스트림 덤프(*-pages-articles-multistream.xml.bz2)는 문서 약 100개씩 독립된 bz2 스트림을 이어 붙인 파일이라
// 스트림 경계에서 잘라 여러 스레드가 따로 풀 수 있음. 경계는 바이트 정렬된 스트림 머리("BZh9" + 블록 표지)를 찾고,
// 그 자리부터 풀었을 때 <page>로 시작하는지로 확인함 (색인 파일 불필요).
// 일반 덤프(단일 스트림)는 경계가 없어서 덩어리 하나로 순차 분석됨
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use bzip2::read::{BzDecoder, MultiBzDecoder};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use anyhow::Context;
use crate::config::Config;
use crate::corpus::{CorpusStats, FastCorpus};
use crate::download;
use crate::dump::{self, ArticleCleaner, ScanCounts};
use crate::{SHUTDOWN, SPACE_ID};

// 병렬 분석 덩어리 크기 (압축 상태 기준). 27GB 덤프면 약 400개. 중간 저장 단위이기도 함
const CHUNK_BYTES: u64 = 64 << 20;
// 스트림 경계를 찾을 때 한 번에 읽는 크기
const SCAN_BYTES: usize = 4 << 20;

// 글자 세기: a-z와 단어 사이 공백을 셈.
// - 공백은 양옆이 글자인 공백 하나(단어 사이)만 셈: `the end` → he_, e_e, _en.
//   문장부호·숫자 옆(`. The`, `end.` 뒤), 연속 공백, 줄 처음·끝, 마크업 옆 공백은 세지 않고 연속을 끊음
// - 문장부호·숫자 자체는 n-gram에 넣지 않고 끊음
// - 세지 않는 공백도 경우별 개수는 따로 셈 (분석 뒤 확인용 출력)
// - 같은 글자 연속(ll)은 빅그램으로 세지 않고, 그 쌍이 낀 트라이그램도 세지 않음

// 실제 글의 문장부호와 숫자 (공백 경우별 개수 출력에서 마크업 찌꺼기와 나눠 보이려고 씀)
fn is_text_punct(b: u8) -> bool {
  b.is_ascii_digit() || matches!(b, b'.' | b',' | b';' | b':' | b'!' | b'?' | b')' | b']' | b'(' | b'[' | b'"' | b'\'')
}

// 공백 옆에 오는 것의 종류. Other = 줄 처음·끝, 마크업이나 그 밖의 기호
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side { Letter, Punct, Other }

// 키 연속(사슬)을 이어 가며 n-gram을 셈
struct Counter<'a> {
  stats: &'a mut FastCorpus,
  hist: [u8; 3],
  len: usize,
}

impl Counter<'_> {
  fn commit(&mut self, id: u8) {
    self.stats.add_letter(id);
    self.hist = [self.hist[1], self.hist[2], id];
    self.len += 1;
    let h = self.hist;
    if self.len >= 2 && h[1] != 255 && h[1] != h[2] {
      self.stats.add_bigram(h[1], h[2]);
      if self.len >= 3 && h[0] != 255 && h[0] != h[1] { self.stats.add_trigram(h[0], h[1], h[2]); }
    }
  }

  fn reset(&mut self) { self.len = 0; self.hist = [255; 3]; }

  // 공백 run개가 left와 right 사이에서 끝남
  fn spaces(&mut self, left: Side, right: Side, run: usize) {
    let counted = left != Side::Other && right != Side::Other;
    let case = if !counted { 4 } else if run > 1 { 3 } else {
      match (left, right) { (Side::Letter, Side::Letter) => 0, (Side::Letter, _) | (_, Side::Letter) => 1, _ => 2 }
    };
    self.stats.space_cases[case] += 1;
    // 단어 사이(글자 _ 글자)만 셈. 왼쪽 글자의 사슬에 이어 붙이고, 오른쪽 글자가 이어서 붙음
    if case == 0 { self.commit(SPACE_ID); } else { self.reset(); }
  }
}

// 문서 하나(마크업을 지운 글)의 글자를 위 규칙대로 셈
pub fn analyze_bytes(bytes: &[u8], stats: &mut FastCorpus) {
  let mut c = Counter { stats, hist: [255; 3], len: 0 };
  let mut prev = Side::Other; // 마지막으로 본 공백이 아닌 문자 (글 처음은 Other)
  let mut left = Side::Other;
  let mut run = 0usize;
  for &b in bytes {
    if b == b' ' {
      if run == 0 { left = prev; }
      run += 1;
      continue;
    }
    let lch = b.to_ascii_lowercase();
    let cur = if lch.is_ascii_lowercase() { Side::Letter } else if is_text_punct(b) { Side::Punct } else { Side::Other };
    if run > 0 { c.spaces(left, cur, run); run = 0; }
    if cur == Side::Letter { c.commit(lch - b'a'); } else { c.reset(); }
    prev = cur;
  }
  // 글 끝의 공백은 세지 않음
  if run > 0 { c.spaces(left, Side::Other, run); }
}

fn interrupted() -> anyhow::Error { anyhow::anyhow!("interrupted") }

// 압축 해제된 XML에서 문서를 골라 통계에 더함
fn analyze_xml(xml: impl std::io::BufRead, sample: f64, limit: usize, stats: &mut FastCorpus, progress: bool) -> anyhow::Result<ScanCounts> {
  let mut cleaner = ArticleCleaner::new();
  let mut n = 0usize;
  dump::scan_articles(xml, sample, |_, raw| {
    if SHUTDOWN.load(Ordering::SeqCst) { return Err(interrupted()); }
    analyze_bytes(cleaner.clean(raw), stats);
    n += 1;
    if progress && n.is_multiple_of(10_000) {
      print!("\r[Wikipedia Dump] {n} articles          ");
      let _ = std::io::stdout().flush();
    }
    Ok(limit == 0 || n < limit)
  })
}

// bz2 덤프를 처음부터 순차 분석 (HTTP 스트림 또는 파일). limit > 0이면 앞쪽 limit개 문서만
pub fn analyze_sequential(source: impl Read, sample: f64, limit: usize) -> anyhow::Result<(FastCorpus, ScanCounts)> {
  let decoder = MultiBzDecoder::new(BufReader::with_capacity(1 << 20, source));
  let mut stats = FastCorpus::new();
  let counts = analyze_xml(BufReader::with_capacity(1 << 20, decoder), sample, limit, &mut stats, true)
    .map_err(|e| if SHUTDOWN.load(Ordering::SeqCst) { anyhow::anyhow!("corpus analysis interrupted") } else { e })?;
  println!("\r[Wikipedia Dump] done: {} articles          ", counts.articles);
  Ok((stats, counts))
}

// 바이트 정렬된 bz2 스트림 머리: "BZh" + 블록 크기(1~9) + 첫 블록 표지(0x314159265359 = "1AY&SY")
fn is_stream_header(w: &[u8]) -> bool {
  w.len() >= 10 && &w[..3] == b"BZh" && (b'1'..=b'9').contains(&w[3]) && &w[4..10] == b"1AY&SY"
}

// off에서 시작하는 bz2 스트림을 풀면 <page>로 시작하는지 (압축 데이터 안의 우연한 일치를 걸러냄)
fn starts_page(path: &Path, off: u64) -> bool {
  let Ok(mut f) = File::open(path) else { return false; };
  if f.seek(SeekFrom::Start(off)).is_err() { return false; }
  let mut dec = BzDecoder::new(BufReader::new(f));
  let mut head = [0u8; 256];
  let mut got = 0;
  while got < head.len() {
    match dec.read(&mut head[got..]) {
      Ok(0) | Err(_) => break,
      Ok(n) => got += n,
    }
  }
  head[..got].trim_ascii_start().starts_with(b"<page>")
}

// from 이후 처음 나오는, <page>로 시작하는 스트림의 위치
fn next_page_stream(path: &Path, from: u64, len: u64) -> anyhow::Result<Option<u64>> {
  let mut f = File::open(path)?;
  let mut buf = vec![0u8; SCAN_BYTES + 9];
  let mut pos = from;
  while pos < len {
    f.seek(SeekFrom::Start(pos))?;
    let want = ((len - pos) as usize).min(buf.len());
    f.read_exact(&mut buf[..want])?;
    for i in 0..want.saturating_sub(9) {
      if buf[i] == b'B' && is_stream_header(&buf[i..want]) && starts_page(path, pos + i as u64) {
        return Ok(Some(pos + i as u64));
      }
    }
    // 마지막 9바이트는 다음 창과 겹치게 해서 경계에 걸친 머리도 찾음
    if want < buf.len() { break; }
    pos += SCAN_BYTES as u64;
  }
  Ok(None)
}

// 덩어리 경계 [b0=0, b1, ..., len]. 경계는 모두 <page>로 시작하는 스트림 위치
pub fn chunk_boundaries(path: &Path, len: u64, chunk: u64) -> anyhow::Result<Vec<u64>> {
  let n = len.div_ceil(chunk.max(1));
  let found: Vec<Option<u64>> = (1..n).into_par_iter()
    .map(|k| next_page_stream(path, k * chunk, len))
    .collect::<anyhow::Result<_>>()?;
  let mut b: Vec<u64> = std::iter::once(0).chain(found.into_iter().flatten()).chain(std::iter::once(len)).collect();
  b.sort_unstable();
  b.dedup();
  Ok(b)
}

// 파일의 [start, end) 바이트(스트림 경계로 자른 덩어리)를 분석
fn analyze_range(path: &Path, start: u64, end: u64, sample: f64) -> anyhow::Result<(FastCorpus, ScanCounts)> {
  let mut f = File::open(path)?;
  f.seek(SeekFrom::Start(start))?;
  let decoder = MultiBzDecoder::new(BufReader::with_capacity(1 << 20, f.take(end - start)));
  let mut stats = FastCorpus::new();
  let counts = analyze_xml(BufReader::with_capacity(1 << 20, decoder), sample, 0, &mut stats, false)?;
  Ok((stats, counts))
}

// 중간 저장. 덤프 크기·파서 버전·표본 비율이 같을 때만 이어서 씀
#[derive(Serialize, Deserialize)]
struct Checkpoint {
  parser_version: u32,
  dump_len: u64,
  sample_bits: u64,
  boundaries: Vec<u64>,
  done: Vec<bool>,
  counts: ScanCounts,
  letters: Vec<u64>,
  bigrams: Vec<u64>,
  trigrams: Vec<u64>,
  space_cases: Vec<u64>,
}

impl Checkpoint {
  fn matches(&self, dump_len: u64, sample: f64) -> bool {
    self.parser_version == dump::PARSER_VERSION && self.dump_len == dump_len && self.sample_bits == sample.to_bits()
      && self.done.len() + 1 == self.boundaries.len()
  }

  fn save(&self, path: &Path) -> anyhow::Result<()> {
    let tmp = path.with_extension("partial.tmp");
    std::fs::write(&tmp, bincode::serialize(self)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
  }

  fn load(path: &Path) -> Option<Self> {
    bincode::deserialize(&std::fs::read(path).ok()?).ok()
  }
}

struct Progress {
  stats: FastCorpus,
  counts: ScanCounts,
  done: Vec<bool>,
}

// 덤프 파일을 덩어리로 나눠 병렬 분석. 덩어리마다 checkpoint에 저장하고, 중단 후 다시 실행하면 이어서 분석.
// 다 끝나도 checkpoint는 남김 (호출한 쪽이 코퍼스를 저장한 뒤 지움)
pub fn analyze_file_parallel(path: &Path, sample: f64, checkpoint: &Path) -> anyhow::Result<(FastCorpus, ScanCounts)> {
  let len = std::fs::metadata(path)?.len();

  let resumed = Checkpoint::load(checkpoint).filter(|c| c.matches(len, sample));
  if resumed.is_none() && checkpoint.exists() {
    println!("[Wikipedia Dump] ignoring {} (different dump, parser or sample)", checkpoint.display());
  }
  let (boundaries, progress) = if let Some(c) = resumed {
    let stats = FastCorpus::from_vecs(&c.letters, &c.bigrams, &c.trigrams, &c.space_cases)?;
    let done_n = c.done.iter().filter(|&&d| d).count();
    println!("[Wikipedia Dump] resuming: {done_n}/{} chunks already done (delete {} to start over)", c.done.len(), checkpoint.display());
    (c.boundaries, Progress { stats, counts: c.counts, done: c.done })
  } else {
    print!("[Wikipedia Dump] finding bz2 stream boundaries..."); let _ = std::io::stdout().flush();
    let b = chunk_boundaries(path, len, CHUNK_BYTES)?;
    println!(" {} chunks", b.len() - 1);
    if b.len() == 2 && len > CHUNK_BYTES {
      println!("[Wikipedia Dump] no stream boundaries: not a multistream dump, analyzing in one thread without resume.");
      println!("                 Use *-pages-articles-multistream.xml.bz2 for parallel analysis.");
    }
    let n = b.len() - 1;
    (b, Progress { stats: FastCorpus::new(), counts: ScanCounts::default(), done: vec![false; n] })
  };

  let total = boundaries.len() - 1;
  let pending: Vec<usize> = (0..total).filter(|&k| !progress.done[k]).collect();
  let progress = Mutex::new(progress);
  println!("[Wikipedia Dump] analyzing {} chunks on {} threads", pending.len(), rayon::current_num_threads());

  let results: Vec<anyhow::Result<()>> = pending.par_iter().map(|&k| {
    if SHUTDOWN.load(Ordering::SeqCst) { return Ok(()); }
    let (stats, counts) = analyze_range(path, boundaries[k], boundaries[k + 1], sample)
      .map_err(|e| e.context(format!("chunk {k} (bytes {}..{})", boundaries[k], boundaries[k + 1])))?;
    let mut p = progress.lock().map_err(|_| anyhow::anyhow!("progress lock poisoned"))?;
    p.stats.merge(&stats);
    p.counts.add(&counts);
    p.done[k] = true;
    let (letters, bigrams, trigrams, space_cases) = p.stats.to_vecs();
    Checkpoint {
      parser_version: dump::PARSER_VERSION, dump_len: len, sample_bits: sample.to_bits(),
      boundaries: boundaries.clone(), done: p.done.clone(), counts: p.counts, letters, bigrams, trigrams, space_cases,
    }.save(checkpoint)?;
    let done_n = p.done.iter().filter(|&&d| d).count();
    print!("\r[Wikipedia Dump] {done_n}/{total} chunks ({:.1}%), {} articles          ",
      done_n as f64 * 100.0 / total as f64, p.counts.articles);
    let _ = std::io::stdout().flush();
    Ok(())
  }).collect();

  let p = progress.into_inner().map_err(|_| anyhow::anyhow!("progress lock poisoned"))?;
  let done_n = p.done.iter().filter(|&&d| d).count();
  if SHUTDOWN.load(Ordering::SeqCst) || done_n < total {
    let first_err = results.into_iter().find_map(Result::err).filter(|e| e.root_cause().to_string() != "interrupted");
    let saved = format!("{done_n}/{total} chunks saved in {}; re-run to resume", checkpoint.display());
    return Err(match first_err {
      Some(e) => e.context(saved),
      None => anyhow::anyhow!("corpus analysis interrupted: {saved}"),
    });
  }
  println!("\r[Wikipedia Dump] done: {total} chunks, {} articles          ", p.counts.articles);
  Ok((p.stats, p.counts))
}

// 덤프 분석 (analyze 명령): 받아 둔 파일이 있으면 병렬(멀티스트림)·이어서 분석, 아니면 스트리밍 순차 분석.
// wikipedia_limit > 0이면 앞쪽 N개만 순차로 (빠른 확인용). 결과는 corpus_path에 저장
pub fn analyze_dump(result_path: &Path, config: &Config, corpus_path: &Path) -> anyhow::Result<CorpusStats> {
  // 저장할 폴더를 먼저 만듦: 몇 시간짜리 분석이 끝난 뒤 저장 단계에서 실패하지 않도록
  if let Some(dir) = corpus_path.parent() { std::fs::create_dir_all(dir)?; }
  let urls = config.dump_urls();
  let sample = config.wikipedia_sample;
  let checkpoint = std::path::PathBuf::from(format!("{}.partial", corpus_path.display()));
  // 중간 저장은 병렬 분석을 끝까지 마쳤을 때만 지움 (앞쪽 N개만 보는 빠른 확인이 전체 분석의 진행을 지우지 않도록)
  let mut parallel = false;
  let (fast, counts) = if let Some(dump_path) = config.dump_path(result_path) {
    if !dump_path.exists() {
      if urls.is_empty() { anyhow::bail!("dump file {} not found and wikipedia_dump_url is empty", dump_path.display()); }
      if let Some(dir) = dump_path.parent() { std::fs::create_dir_all(dir)?; }
      println!("[Download] -> {}", dump_path.display());
      let opts = download::DownloadOpts { verify: config.wikipedia_verify, parallel: config.wikipedia_parallel_download, ..Default::default() };
      download::download_dump(&download::http_client()?, &urls, &dump_path, &opts)?;
    }
    println!("[Wikipedia Dump] Reading {}", dump_path.display());
    if config.wikipedia_limit > 0 {
      analyze_sequential(File::open(&dump_path)?, sample, config.wikipedia_limit)?
    } else {
      parallel = true;
      analyze_file_parallel(&dump_path, sample, &checkpoint)?
    }
  } else {
    println!("[Wikipedia Dump] streaming: single thread, no resume (set wikipedia_download = true for both)");
    let client = download::http_client()?;
    let url = if urls.len() > 1 {
      let opts = download::DownloadOpts::default();
      download::rank_mirrors(&client, &urls, &opts)?.remove(0).url
    } else { urls[0].clone() };
    println!("[Wikipedia Dump] Connecting to {url}");
    let response = client.get(&url).send()?.error_for_status()?;
    analyze_sequential(response, sample, config.wikipedia_limit)?
  };
  println!("[Wikipedia Dump] used {} articles; skipped {} redirects, {} list/disambiguation pages, {} outside sample",
    counts.articles, counts.redirects, counts.lists, counts.sampled_out);
  fast.print_space_cases();
  let stats = fast.into_stats();
  stats.save(corpus_path.to_str().context("Invalid corpus path")?, dump::PARSER_VERSION)?;
  if parallel && checkpoint.exists() { std::fs::remove_file(&checkpoint)?; }
  Ok(stats)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::io::Write;
  use crate::layout::CHARS;

  fn page(title: &str, text: &str) -> String {
    format!("  <page>\n    <title>{title}</title>\n    <ns>0</ns>\n    <revision><text bytes=\"1\">{text}</text></revision>\n  </page>\n")
  }

  fn bz2(data: &str) -> Vec<u8> {
    let mut e = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::best());
    e.write_all(data.as_bytes()).unwrap();
    e.finish().unwrap()
  }

  // 멀티스트림 덤프 흉내: 머리(siteinfo) 스트림 + 문서 스트림 여러 개 + 꼬리 스트림
  fn multistream(pages: &[String], per_stream: usize) -> Vec<u8> {
    let mut out = bz2("<mediawiki>\n  <siteinfo><sitename>W</sitename></siteinfo>\n");
    for group in pages.chunks(per_stream) { out.extend(bz2(&group.concat())); }
    out.extend(bz2("</mediawiki>\n"));
    out
  }

  fn pages() -> Vec<String> {
    let mut v: Vec<String> = (0..60).map(|i| page(&format!("Article {i}"), &format!("Some prose number {i} about rivers and hills."))).collect();
    v.push(page("Lutetia", "#REDIRECT [[Paris]]"));
    v.push(page("List of rivers", "* [[Seine]]"));
    v.push(page("Mercury", "'''Mercury''' may refer to: {{disambiguation}}"));
    v
  }

  fn tmp(name: &str, data: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("p2anut-wiki-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("dump.xml.bz2");
    std::fs::write(&p, data).unwrap();
    p
  }

  #[test]
  fn parallel_matches_sequential_and_skips_non_prose() {
    let data = multistream(&pages(), 5);
    let path = tmp("par", &data);
    let (seq, seq_counts) = analyze_sequential(&data[..], 1.0, 0).unwrap();
    assert_eq!((seq_counts.articles, seq_counts.redirects, seq_counts.lists), (60, 1, 2));

    // 작은 덩어리로 잘라 경계를 여러 개 만듦
    let b = chunk_boundaries(&path, data.len() as u64, 200).unwrap();
    assert!(b.len() > 4, "{b:?}");
    let mut merged = FastCorpus::new();
    let mut counts = ScanCounts::default();
    for w in b.windows(2) {
      let (s, c) = analyze_range(&path, w[0], w[1], 1.0).unwrap();
      merged.merge(&s);
      counts.add(&c);
    }
    assert_eq!(merged, seq);
    assert_eq!(counts.articles, 60);

    let cp = path.with_extension("partial");
    let (par, par_counts) = analyze_file_parallel(&path, 1.0, &cp).unwrap();
    assert_eq!(par, seq);
    assert_eq!(par_counts.articles, 60);
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
  }

  #[test]
  fn resume_uses_only_matching_checkpoint() {
    let data = multistream(&pages(), 5);
    let path = tmp("resume", &data);
    let len = data.len() as u64;
    let cp = path.with_extension("partial");
    let b = chunk_boundaries(&path, len, 200).unwrap();
    let n = b.len() - 1;

    // 첫 덩어리만 끝난 상태를 흉내냄. 덩어리 0의 통계를 일부러 2배로 넣어, 이어서 분석할 때 다시 세지 않는지 확인
    let (s0, c0) = analyze_range(&path, b[0], b[1], 1.0).unwrap();
    let mut doubled = s0.clone();
    doubled.merge(&s0);
    let (letters, bigrams, trigrams, space_cases) = doubled.to_vecs();
    let mut done = vec![false; n];
    done[0] = true;
    let checkpoint = Checkpoint { parser_version: dump::PARSER_VERSION, dump_len: len, sample_bits: 1.0f64.to_bits(),
      boundaries: b.clone(), done, counts: c0, letters, bigrams, trigrams, space_cases };
    checkpoint.save(&cp).unwrap();

    let (seq, _) = analyze_sequential(&data[..], 1.0, 0).unwrap();
    let (resumed, _) = analyze_file_parallel(&path, 1.0, &cp).unwrap();
    let mut expected = seq.clone();
    expected.merge(&s0);
    assert_eq!(resumed, expected, "done chunks must come from the checkpoint, not be re-analyzed");

    // 표본 비율이 다르면 중간 저장을 무시하고 처음부터
    checkpoint.save(&cp).unwrap();
    let (fresh, _) = analyze_file_parallel(&path, 0.5, &cp).unwrap();
    let (seq_half, _) = analyze_sequential(&data[..], 0.5, 0).unwrap();
    assert_eq!(fresh, seq_half);
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
  }

  #[test]
  fn single_stream_dump_is_one_chunk() {
    let mut xml = String::from("<mediawiki>\n");
    for p in pages() { xml.push_str(&p); }
    xml.push_str("</mediawiki>\n");
    let data = bz2(&xml);
    let path = tmp("single", &data);
    assert_eq!(chunk_boundaries(&path, data.len() as u64, 100).unwrap(), vec![0, data.len() as u64]);
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
  }

  #[test]
  fn sequential_limit_and_sample() {
    let data = multistream(&pages(), 5);
    let (_, c) = analyze_sequential(&data[..], 1.0, 7).unwrap();
    assert_eq!(c.articles, 7);
    let (_, c) = analyze_sequential(&data[..], 0.5, 0).unwrap();
    assert!(c.articles > 15 && c.articles < 45, "{}", c.articles);
    assert_eq!(c.articles + c.sampled_out, 60);
  }

  // 글자 세기 (analyze_bytes)
  fn id(c: char) -> usize { (c as u8 - b'a') as usize }
  fn run(text: &str) -> FastCorpus { let mut f = FastCorpus::new(); analyze_bytes(text.as_bytes(), &mut f); f }
  fn bi(f: &FastCorpus, s: &str) -> u64 { let c: Vec<char> = s.chars().collect(); f.bigrams[id(c[0]) * CHARS + id(c[1])] }
  fn tri(f: &FastCorpus, s: &str) -> u64 {
    let c: Vec<char> = s.chars().collect();
    f.trigrams[(id(c[0]) * CHARS + id(c[1])) * CHARS + id(c[2])]
  }

  const SP: char = ' ';
  fn idc(c: char) -> usize { if c == SP { 26 } else { id(c) } }
  fn bi_s(f: &FastCorpus, a: char, b: char) -> u64 { f.bigrams[idc(a) * CHARS + idc(b)] }
  fn tri_s(f: &FastCorpus, a: char, b: char, c: char) -> u64 { f.trigrams[(idc(a) * CHARS + idc(b)) * CHARS + idc(c)] }

  #[test]
  fn one_space_between_letters_is_counted() {
    let f = run("ab Cd");
    assert_eq!(f.letters[26], 1);
    assert_eq!((bi_s(&f, 'b', SP), bi_s(&f, SP, 'c')), (1, 1));
    assert_eq!((tri_s(&f, 'a', 'b', SP), tri_s(&f, 'b', SP, 'c'), tri_s(&f, SP, 'c', 'd')), (1, 1, 1));
    // 공백을 건너뛴 글자끼리는 이어지지 않음, 대문자는 소문자로
    assert_eq!((bi(&f, "bc"), tri(&f, "abc"), tri(&f, "bcd"), bi(&f, "cd")), (0, 0, 0, 1));
    assert_eq!(f.letters.iter().take(26).sum::<u64>(), 4);
    // 한 글자 단어: 공백-글자-공백 트라이그램
    let f = run("x a y");
    assert_eq!((tri_s(&f, 'x', SP, 'a'), tri_s(&f, SP, 'a', SP), tri_s(&f, 'a', SP, 'y')), (1, 1, 1));
    assert_eq!(f.letters[26], 2);
  }

  // 공백이 낀 n-gram이 하나도 없는지
  fn no_space_ngrams(f: &FastCorpus) -> bool {
    f.bigrams.iter().skip(26).step_by(CHARS).all(|&c| c == 0) && f.bigrams[26 * CHARS..27 * CHARS].iter().all(|&c| c == 0)
      && f.trigrams.iter().enumerate().all(|(k, &c)| c == 0 || (k % CHARS != 26 && k / CHARS % CHARS != 26 && k / (CHARS * CHARS) != 26))
  }

  #[test]
  fn spaces_next_to_punctuation_are_not_counted() {
    // 글자 _ 문장부호, 문장부호 _ 글자, 숫자 옆, 문장부호끼리: 세지 않고 끊음 (경우별 개수만)
    for (text, cases) in [("ab ,cd", [0, 1, 0, 0, 0]), ("ab, cd", [0, 1, 0, 0, 0]), ("ab 1 cd", [0, 2, 0, 0, 0]), ("ab. (cd", [0, 0, 1, 0, 0])] {
      let f = run(text);
      assert_eq!((f.letters[26], f.space_cases), (0, cases), "{text:?}");
      assert!(no_space_ngrams(&f), "{text:?}");
      assert_eq!((bi(&f, "ab"), bi(&f, "cd")), (1, 1), "{text:?}");
    }
    // ". The"의 공백은 세지 않지만 뒤 단어 안의 n-gram은 그대로
    let f = run("end. The");
    assert_eq!((f.letters[26], tri(&f, "the"), bi(&f, "dt")), (0, 1, 0));
  }

  #[test]
  fn double_spaces_are_not_counted() {
    for text in ["abc  De", "ab,   cd"] {
      let f = run(text);
      assert_eq!((f.letters[26], f.space_cases), (0, [0, 0, 0, 1, 0]), "{text:?}");
      assert!(no_space_ngrams(&f), "{text:?}");
    }
    let f = run("abc  De");
    assert_eq!((bi(&f, "ab"), bi(&f, "bc"), tri(&f, "abc"), bi(&f, "cd"), bi(&f, "de")), (1, 1, 1, 0, 1));
  }

  #[test]
  fn spaces_at_line_edges_or_next_to_markup_are_not_counted() {
    for (text, cases) in [(" ab ", 2), ("ab \ncd", 1), ("ab\n cd", 1), ("ab | cd", 2), ("ab =cd", 1), ("ab\tcd", 0)] {
      let f = run(text);
      assert_eq!((f.letters[26], f.space_cases), (0, [0, 0, 0, 0, cases]), "{text:?}");
      assert!(no_space_ngrams(&f), "{text:?}");
    }
  }

  #[test]
  fn letter_space_letter_counts_match_a_direct_scan() {
    // "a _ c"는 v4와 같은 규칙: 양옆이 글자인 공백 하나
    let text = crate::test_util::TEXT.repeat(2) + " A few, odd  spaces; and an I here. x y z (q) 3 dogs";
    let f = run(&text);
    let b = text.as_bytes();
    let mut want = vec![0u64; 26 * 26];
    for i in 1..b.len() - 1 {
      if b[i] == b' ' && b[i - 1].is_ascii_alphabetic() && b[i + 1].is_ascii_alphabetic() {
        want[id(b[i - 1].to_ascii_lowercase() as char) * 26 + id(b[i + 1].to_ascii_lowercase() as char)] += 1;
      }
    }
    for a in 0..26 { for c in 0..26 { assert_eq!(f.trigrams[(a * CHARS + 26) * CHARS + c], want[a * 26 + c]); } }
    // 누름은 단어 사이 공백만
    assert_eq!(f.letters[26], f.space_cases[0]);
  }

  #[test]
  fn repeated_letters_next_to_a_space_follow_the_old_rule() {
    // ll은 세지 않고 ll이 낀 트라이그램도 세지 않음. l_, l_i는 셈
    let f = run("all in");
    assert_eq!((bi(&f, "ll"), tri(&f, "all"), tri_s(&f, 'l', 'l', SP)), (0, 0, 0));
    assert_eq!((bi_s(&f, 'l', SP), tri_s(&f, 'l', SP, 'i'), tri_s(&f, SP, 'i', 'n')), (1, 1, 1));
  }

  #[test]
  fn letter_only_counts_match_the_old_space_breaking_rule() {
    let text = crate::test_util::TEXT.repeat(3) + " A few, odd  spaces; and an I here. x y z";
    let new = run(&text);
    // 예전 규칙: 공백도 연속을 끊음 (줄바꿈으로 바꿔 세면 같음)
    let old = run(&text.replace(' ', "\n"));
    for i in 0..26 { assert_eq!(new.letters[i], old.letters[i]); }
    for a in 0..26 { for b in 0..26 {
      assert_eq!(new.bigrams[a * CHARS + b], old.bigrams[a * CHARS + b]);
      for c in 0..26 { let k = (a * CHARS + b) * CHARS + c; assert_eq!(new.trigrams[k], old.trigrams[k]); }
    }}
    assert!(new.letters[26] > 0 && old.letters[26] == 0);
  }

  // 문서 몇 개짜리 bz2 덤프
  fn tiny_dump(path: &std::path::Path) {
    let mut xml = String::from("<mediawiki>\n");
    for i in 0..5 { xml.push_str(&format!("<page><title>Article {i}</title><ns>0</ns><revision><text>Some words here {i}.</text></revision></page>\n")); }
    xml.push_str("</mediawiki>\n");
    std::fs::write(path, bz2(&xml)).unwrap();
  }

  fn test_config(dump_file: &std::path::Path, limit: usize) -> crate::config::Config {
    let mut c = crate::config::load_config(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml"))).unwrap();
    c.wikipedia_dump_file = Some(dump_file.to_str().unwrap().to_string());
    c.wikipedia_limit = limit;
    c
  }

  #[test]
  fn analyze_creates_missing_result_dir_and_keeps_checkpoint_for_quick_runs() {
    let root = std::env::temp_dir().join(format!("p2anut-analyze-{}", std::process::id()));
    let dumps = root.join("dumps");
    std::fs::create_dir_all(&dumps).unwrap();
    let dump = dumps.join("d.xml.bz2");
    tiny_dump(&dump);

    // 결과 폴더가 없어도 (덤프는 다른 곳) 분석 후 저장까지 됨
    let result = root.join("new").join("result");
    let corpus = result.join("corpus.bin");
    let stats = analyze_dump(&result, &test_config(&dump, 2), &corpus).unwrap();
    assert!(corpus.exists() && stats.letters.values().sum::<u64>() > 0);

    // 앞쪽 N개만 보는 빠른 확인은 전체 분석의 중간 저장을 지우지 않음
    let checkpoint = std::path::PathBuf::from(format!("{}.partial", corpus.display()));
    std::fs::write(&checkpoint, b"in-progress").unwrap();
    analyze_dump(&result, &test_config(&dump, 2), &corpus).unwrap();
    assert_eq!(std::fs::read(&checkpoint).unwrap(), b"in-progress");

    // 병렬 분석을 끝까지 마치면 지움 (내용이 맞지 않는 중간 저장은 무시하고 처음부터 분석)
    analyze_dump(&result, &test_config(&dump, 0), &corpus).unwrap();
    assert!(!checkpoint.exists());
    std::fs::remove_dir_all(&root).unwrap();
  }

  #[test]
  fn punctuation_and_newlines_break_sequences() {
    let f = run("ab,cd\nef");
    assert_eq!((bi(&f, "bc"), bi(&f, "de"), bi(&f, "cd"), bi(&f, "ef")), (0, 0, 1, 1));
  }
}
