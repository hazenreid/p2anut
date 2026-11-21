// 위키 덤프 분석: 순차(스트리밍·앞쪽 N개) 또는 멀티스트림 덤프 파일의 병렬 분석(중간 저장·이어서 분석)
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
use crate::corpus::FastCorpus;
use crate::dump::{self, ArticleCleaner, ScanCounts};
use crate::SHUTDOWN;

// 병렬 분석 덩어리 크기 (압축 상태 기준). 27GB 덤프면 약 400개. 중간 저장 단위이기도 함
const CHUNK_BYTES: u64 = 64 << 20;
// 스트림 경계를 찾을 때 한 번에 읽는 크기
const SCAN_BYTES: usize = 4 << 20;

fn interrupted() -> anyhow::Error { anyhow::anyhow!("interrupted") }

// 압축 해제된 XML에서 문서를 골라 통계에 더함
fn analyze_xml(xml: impl std::io::BufRead, sample: f64, limit: usize, stats: &mut FastCorpus, progress: bool) -> anyhow::Result<ScanCounts> {
  let mut cleaner = ArticleCleaner::new();
  let mut n = 0usize;
  dump::scan_articles(xml, sample, |_, raw| {
    if SHUTDOWN.load(Ordering::SeqCst) { return Err(interrupted()); }
    crate::analyze_bytes(cleaner.clean(raw), stats);
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
    let stats = FastCorpus::from_vecs(&c.letters, &c.bigrams, &c.trigrams)?;
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
    let (letters, bigrams, trigrams) = p.stats.to_vecs();
    Checkpoint {
      parser_version: dump::PARSER_VERSION, dump_len: len, sample_bits: sample.to_bits(),
      boundaries: boundaries.clone(), done: p.done.clone(), counts: p.counts, letters, bigrams, trigrams,
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

#[cfg(test)]
mod tests {
  use super::*;
  use std::io::Write;

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
    let (letters, bigrams, trigrams) = doubled.to_vecs();
    let mut done = vec![false; n];
    done[0] = true;
    let checkpoint = Checkpoint { parser_version: dump::PARSER_VERSION, dump_len: len, sample_bits: 1.0f64.to_bits(),
      boundaries: b.clone(), done, counts: c0, letters, bigrams, trigrams };
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
}
