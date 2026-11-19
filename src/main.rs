mod config;
mod corpus;
mod dump;
mod layout;
mod optimizer;
mod precomp;
mod score;
#[cfg(test)]
mod test_util;

use std::path::Path;
use std::io::{BufReader, Write};
use reqwest::blocking::Client;
use std::time::Duration;
use std::fs;
use bzip2::read::MultiBzDecoder;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use anyhow::Context;
use std::sync::atomic::{AtomicBool, Ordering};

pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);

use config::{load_config, Config};
use corpus::CorpusStats;
use precomp::Precomp;
use layout::{init_score_norm, save_results, load_results, sort_unique_layouts};
use optimizer::Optimizer;

// a-z만 셈. 공백은 엄지 키라 배치 대상이 아니므로 세지 않고, 문장부호·숫자·줄바꿈처럼 연속을 끊음
// (단어 경계를 넘는 빅/트라이그램은 만들지 않음: "ab cd"에서 bc는 세지 않음)
fn analyze_bytes(bytes: &[u8], stats: &mut corpus::FastCorpus) {
  let mut hist = [255u8; 3];
  let mut valid_len = 0usize;

  macro_rules! commit {
    ($id:expr) => {
      stats.add_letter($id);
      hist[0] = hist[1]; hist[1] = hist[2]; hist[2] = $id;
      valid_len += 1;
      if valid_len >= 2 && hist[1] != 255 && hist[1] != hist[2] {
        stats.add_bigram(hist[1], hist[2]);
        if valid_len >= 3 && hist[0] != 255 && hist[0] != hist[1] {
          stats.add_trigram(hist[0], hist[1], hist[2]);
        }
      }
    };
  }

  for &b in bytes {
    let lch = b.to_ascii_lowercase();
    if lch.is_ascii_lowercase() {
      commit!(lch - b'a');
    } else {
      valid_len = 0; hist = [255, 255, 255];
    }
  }
}

fn http_client() -> anyhow::Result<Client> {
  Ok(Client::builder()
    .connect_timeout(Duration::from_secs(30))
    .timeout(None)
    // 상대가 사라진 연결(NAT 만료 등)을 OS 수준에서 감지
    .tcp_keepalive(Some(Duration::from_secs(60)))
    .user_agent("KeyboardLayoutOptimizer/1.0 (corpus builder)")
    .build()?)
}

// "bytes 0-99/1234" 또는 "bytes */1234" → 1234
fn content_range_total(resp: &reqwest::blocking::Response) -> Option<u64> {
  resp.headers().get(reqwest::header::CONTENT_RANGE)?
    .to_str().ok()?
    .rsplit('/').next()?
    .parse().ok()
}

// 한 번의 Range 요청으로 받는 크기와 그 요청 하나의 제한 시간.
// reqwest 0.11 blocking에는 읽기 타임아웃이 없어서, 연결이 데이터 없이 멈추면 read가 영원히 막힘
// (실제로 3GB 지점에서 ESTABLISHED 상태로 2시간 넘게 멈춤). 덩어리마다 요청 전체 제한 시간을 걸어
// 멈춘 연결은 끊고 받은 데까지 이어받음. 8MiB를 2분 안에 못 받으면(약 70KB/s 미만) 멈춘 것으로 봄.
// Wikimedia에서 실측한 속도(약 0.18MiB/s)로 한 덩어리에 약 45초 → 여유 약 2.7배.
// 이보다 느린 회선이면 매 덩어리가 시간 초과되지만, 받은 만큼은 저장되므로 계속 진행됨
const DOWNLOAD_CHUNK: u64 = 8 << 20;
const DOWNLOAD_CHUNK_TIMEOUT: Duration = Duration::from_secs(120);

// 덤프를 path로 내려받음. "<path>.part"에 덩어리 단위 Range 요청으로 이어 쓰고, 끝나면 rename
fn download_dump(url: &str, path: &Path) -> anyhow::Result<()> {
  const MAX_FAILURES: u32 = 10;
  let client = http_client()?;
  let part = std::path::PathBuf::from(format!("{}.part", path.display()));
  let mut failures = 0u32;
  let mut total: Option<u64> = None;
  let mut last_report = 0u64;

  loop {
    let offset = fs::metadata(&part).map_or(0, |m| m.len());
    if let Some(t) = total {
      if offset == t { break; }
      if offset > t { fs::remove_file(&part)?; anyhow::bail!("partial file larger than remote ({offset} > {t}); removed, re-run to restart"); }
    }

    // Ok(true) = 파일 끝까지 받음
    let attempt = |total: &mut Option<u64>, last_report: &mut u64| -> anyhow::Result<bool> {
      let range_end = offset + DOWNLOAD_CHUNK - 1;
      let mut resp = client.get(url)
        .header(reqwest::header::RANGE, format!("bytes={offset}-{range_end}"))
        .timeout(DOWNLOAD_CHUNK_TIMEOUT)
        .send()?;
      let status = resp.status();

      if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let t = content_range_total(&resp);
        if t == Some(offset) { return Ok(true); } // 이미 다 받음
        fs::remove_file(&part)?;
        anyhow::bail!("server rejected range at {offset} (remote size {t:?}); partial file removed");
      }
      let (mut file, expected) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        let t = content_range_total(&resp).context("206 response without Content-Range total")?;
        *total = Some(t);
        (fs::OpenOptions::new().create(true).append(true).open(&part)?, range_end.min(t - 1) + 1 - offset)
      } else if status.is_success() {
        // Range 미지원 서버: 덩어리 없이 처음부터 전부 받음 (요청 제한 시간도 걸 수 없음)
        anyhow::ensure!(offset == 0 || total.is_none(), "server stopped honoring Range");
        println!("\n[Download] server ignored Range, downloading whole file without stall detection");
        *total = resp.content_length();
        let mut file = fs::File::create(&part)?;
        let n = std::io::copy(&mut resp, &mut file)?;
        file.flush()?;
        if let Some(t) = *total { anyhow::ensure!(n == t, "connection closed at {n}/{t} bytes"); }
        return Ok(true);
      } else {
        anyhow::bail!("HTTP {status}");
      };

      let mut buf = vec![0u8; 1024 * 1024];
      let mut got = 0u64;
      loop {
        if SHUTDOWN.load(Ordering::SeqCst) {
          file.flush()?;
          anyhow::bail!("download interrupted at {} MiB (re-run to resume)", (offset + got) >> 20);
        }
        let n = std::io::Read::read(&mut resp, &mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
        got += n as u64;
        let done = offset + got;
        if done - *last_report >= 64 << 20 {
          *last_report = done;
          if let Some(t) = *total { print!("\r[Download] {} / {} MiB          ", done >> 20, t >> 20); }
          let _ = std::io::stdout().flush();
        }
      }
      file.flush()?;
      anyhow::ensure!(got == expected, "chunk ended early at {got}/{expected} bytes");
      Ok(total.is_some_and(|t| offset + got == t))
    };

    match attempt(&mut total, &mut last_report) {
      Ok(true) => break,
      Ok(false) => failures = 0,
      Err(e) if SHUTDOWN.load(Ordering::SeqCst) => return Err(e),
      Err(e) => {
        // 진행이 있었으면 연속 실패 횟수 초기화
        if fs::metadata(&part).map_or(0, |m| m.len()) > offset { failures = 0; }
        failures += 1;
        if failures > MAX_FAILURES { return Err(e.context(format!("download failed {MAX_FAILURES} times in a row"))); }
        let wait = Duration::from_secs(u64::from(5 * failures).min(60));
        eprintln!("\n[Download] {e} — retry {failures}/{MAX_FAILURES} in {}s", wait.as_secs());
        std::thread::sleep(wait);
      }
    }
  }
  fs::rename(&part, path)?;
  println!("\n[Download] done: {}", path.display());
  Ok(())
}

// 원본(bz2 덤프)이 HTTP 스트림이든 로컬 파일이든 동일하게 분석
// 중단·오류 시에는 부분 통계를 버리고 에러로 끝냄 (부분 코퍼스가 완성본처럼 쓰이지 않도록)
fn analyze_wikipedia_dump(source: impl std::io::Read, limit: usize) -> anyhow::Result<CorpusStats> {
  let decoder = MultiBzDecoder::new(BufReader::with_capacity(1024 * 1024, source));
  let mut reader = Reader::from_reader(BufReader::with_capacity(1024 * 1024, decoder));

  let mut stats = corpus::FastCorpus::new();
  let mut decode_buf: Vec<u8> = Vec::with_capacity(128 * 1024);
  let mut strip_buf:  Vec<u8> = Vec::with_capacity(128 * 1024);
  let mut buf = Vec::new();
  let mut in_text = false;
  let mut in_ns = false;
  let mut is_main_ns = false;
  let mut article_count = 0usize;

  loop {
    if SHUTDOWN.load(Ordering::SeqCst) {
      anyhow::bail!("corpus analysis interrupted after {article_count} articles");
    }
    match reader.read_event_into(&mut buf) {
      Ok(Event::Start(e)) => match e.name().as_ref() {
        b"page"  => { is_main_ns = false; }
        b"ns"    => { in_ns = true; }
        b"text" if is_main_ns => { in_text = true; }
        _ => {}
      },
      Ok(Event::Text(e)) => {
        if in_ns {
          is_main_ns = e.as_ref() as &[u8] == b"0";
          in_ns = false;
        } else if in_text {
          let raw = e.into_inner();
          if !raw.starts_with(b"#REDIRECT") && !raw.starts_with(b"#redirect") {
            dump::decode_html_entities_into(&raw, &mut decode_buf);
            dump::strip_wiki_markup_into(&decode_buf, &mut strip_buf);
            analyze_bytes(&strip_buf, &mut stats);
            article_count += 1;
            if article_count.is_multiple_of(10_000) {
              print!("\r[Wikipedia Dump] {article_count} articles          ");
              let _ = std::io::stdout().flush();
            }
            if limit > 0 && article_count >= limit { break; }
          }
          in_text = false;
        }
      },
      Ok(Event::End(e)) if e.name().as_ref() == b"text" => { in_text = false; }
      Ok(Event::Eof) => break,
      Err(e) => {
        anyhow::bail!("XML/stream error at {} after {article_count} articles: {e}", reader.buffer_position());
      }
      _ => {}
    }
    buf.clear();
  }

  println!("\r[Wikipedia Dump] done: {article_count} articles          ");
  Ok(stats.into_stats())
}

fn load_or_analyze(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<CorpusStats> {
  let corpus_path = result_path.join(&config.corpus_file);
  let corpus_path_str = corpus_path.to_str().context("Invalid corpus path")?;
  if !reset && corpus_path.exists() {
    println!("[Load Corpus] {}", corpus_path.display());
    Ok(CorpusStats::load(corpus_path_str)?)
  } else {
    let url = &config.wikipedia_dump_url;
    let mut stats = if let Some(file) = &config.wikipedia_dump_file {
      let dump_path = result_path.join(file);
      if !dump_path.exists() {
        if url.is_empty() { anyhow::bail!("dump file {} not found and wikipedia_dump_url is empty", dump_path.display()); }
        if let Some(dir) = dump_path.parent() { fs::create_dir_all(dir)?; }
        println!("[Download] {url} -> {}", dump_path.display());
        download_dump(url, &dump_path)?;
      }
      println!("[Wikipedia Dump] Reading {}", dump_path.display());
      analyze_wikipedia_dump(fs::File::open(&dump_path)?, config.wikipedia_limit)?
    } else {
      println!("[Wikipedia Dump] Connecting...");
      let response = http_client()?.get(url).send()?.error_for_status()?;
      analyze_wikipedia_dump(response, config.wikipedia_limit)?
    };
    stats.trim_by_coverage();
    stats.save(corpus_path_str)?;
    Ok(stats)
  }
}

fn print_results(results: &[layout::Layout]) {
  for (i, l) in results.iter().enumerate() {
    println!("[{}]", i + 1);
    l.print();
    println!();
  }
}

fn cmd_analyze(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<()> {
  let stats = load_or_analyze(result_path, config, reset)?;
  println!("letters: {}, bigrams: {}, trigrams: {}",
    stats.letters.len(), stats.bigrams.len(), stats.trigrams.len());
  stats.print();
  Ok(())
}

fn cmd_results(result_path: &Path, config: &Config) -> anyhow::Result<()> {
  let result_file = result_path.join(&config.result_file);
  let result_file_str = result_file.to_str().context("Invalid result file path")?;
  if !result_file.exists() {
    eprintln!("No results found at {}", result_file.display());
    std::process::exit(1);
  }
  let mut results = load_results(result_file_str)?;
  let corpus_path = result_path.join(&config.corpus_file);
  if corpus_path.exists() {
    let stats = CorpusStats::load(corpus_path.to_str().context("Invalid corpus path")?)?;
    let arrays = stats.to_arrays(false);
    results.retain(|l| l.key_count() == arrays.keys.len());
    let precomp = Precomp::new(&arrays.freq, &config.weights, config.min_hand_usage);
    let norm = init_score_norm(&arrays, &precomp, &config.rates);
    for l in &mut results {
      l.calc_scores(&arrays, &precomp);
      l.calc_total_score(&norm);
    }
    layout::sort_layouts(&mut results);
  }
  print_results(&results);
  Ok(())
}

fn run_optimize(result_path: &Path, config: &Config, stats: &corpus::CorpusStats, reset: bool) -> anyhow::Result<()> {
  let arrays = stats.to_arrays(false);
  print!("[Precomp] ..."); let _ = std::io::stdout().flush();
  let precomp = Precomp::new(&arrays.freq, &config.weights, config.min_hand_usage);
  println!(" done");
  print!("[Score Norm] ..."); let _ = std::io::stdout().flush();
  let norm = init_score_norm(&arrays, &precomp, &config.rates);
  println!(" done");
  let center_mask = arrays.rare_mask(config.center_rare);
  if center_mask != 0 {
    let letters: String = (0..arrays.keys.len()).filter(|&i| center_mask >> i & 1 == 1).map(|i| arrays.keys[i]).collect();
    println!("[Center] least frequent {} letters forced into the center columns: {letters}", config.center_rare);
  }
  let result_file = result_path.join(&config.result_file);
  let result_file_str = result_file.to_str().context("Invalid result file path")?;
  let base_layouts = if !reset && result_file.exists() {
    let mut layouts = load_results(result_file_str)?;
    // 공백을 그리드에 두던 예전 결과(27키)는 키 개수가 달라 교차(PMX)를 깨뜨리므로 시드로 쓰지 않음
    let before = layouts.len();
    layouts.retain(|l| l.key_count() == arrays.keys.len());
    if layouts.len() < before {
      println!("[Results] skipped {} saved layouts with a different key set (old 27-key layouts with space)", before - layouts.len());
    }
    // center_rare를 어기는 배열은 구역 안에서만 움직이는 SA로는 못 고치므로 시드로 쓰지 않음
    let before = layouts.len();
    layouts.retain(|l| l.center_ok(center_mask));
    if layouts.len() < before {
      println!("[Results] skipped {} saved layouts that break center_rare", before - layouts.len());
    }
    for l in &mut layouts { l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm); }
    layouts
  } else { vec![] };
  let optimizer = Optimizer::new(arrays, &precomp, &norm, config.elites, config.min_home_diff, config.sa.clone(), config.ga.clone(), center_mask);
  let mut best = base_layouts;
  let mut round = 1usize;
  loop {
    println!("[Optimize {round}]");
    let prev = best.clone();
    let (layouts, improved) = optimizer.run(best, result_file_str);
    let mut combined = prev;
    combined.extend(layouts);
    best = sort_unique_layouts(combined, config.elites, &[], config.min_home_diff);
    save_results(&best, result_file_str)?;
    if !improved || SHUTDOWN.load(Ordering::SeqCst) { break; }
    round += 1;
  }
  println!("\n[Results]");
  print_results(&best);
  Ok(())
}

fn cmd_run(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<()> {
  let stats = load_or_analyze(result_path, config, reset)?;
  run_optimize(result_path, config, &stats, reset)
}

fn main() -> anyhow::Result<()> {
  ctrlc::set_handler(move || {
    // 두 번째 Ctrl-C는 즉시 종료 (네트워크 read 등에서 막혀 정상 종료 지점에 못 가는 경우 대비)
    if SHUTDOWN.swap(true, Ordering::SeqCst) {
      eprintln!("\n[Shutdown] second Ctrl-C, exiting now");
      std::process::exit(130);
    }
    println!("\n[Graceful Shutdown] Ctrl-C received, saving and exiting after this round... (Ctrl-C again to force quit)");
  }).unwrap_or_else(|_| eprintln!("Error setting Ctrl-C handler"));

  let args: Vec<String> = std::env::args().collect();
  if args.len() < 2 {
    eprintln!("Usage: {} <result_path> [analyze|optimize|run|results] [--reset]", args[0]);
    std::process::exit(1);
  }
  let result_path = Path::new(&args[1]);
  let cmd = args.get(2).map(std::string::String::as_str);
  let reset = args.iter().any(|a| a == "--reset") && !matches!(cmd, Some("results"));
  let config = load_config(Path::new("config.toml"))?;
  // init_score_norm(무작위 표본)부터 최적화까지 모든 난수가 이 시드에서 나옴
  let seed = config.seed.unwrap_or_else(|| fastrand::u64(..));
  println!("[Seed] {seed}");
  fastrand::seed(seed);
  if reset { fs::create_dir_all(result_path)?; }
  match cmd {
    Some("analyze")  => cmd_analyze(result_path, &config, reset),
    Some("optimize") => {
      let corpus_path = result_path.join(&config.corpus_file);
      let corpus_path_str = corpus_path.to_str().context("Invalid corpus path")?;
      run_optimize(result_path, &config, &CorpusStats::load(corpus_path_str)?, reset)
    },
    Some("run")      => cmd_run(result_path, &config, reset),
    Some("results")  => cmd_results(result_path, &config),
    _ => {
      eprintln!("Usage: {} <result_path> [analyze|optimize|run|results] [--reset]", args[0]);
      std::process::exit(1);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::analyze_bytes;
  use corpus::FastCorpus;
  use crate::corpus;
  use crate::layout::CHARS;

  fn id(c: char) -> usize { (c as u8 - b'a') as usize }
  fn run(text: &str) -> FastCorpus { let mut f = FastCorpus::new(); analyze_bytes(text.as_bytes(), &mut f); f }
  fn bi(f: &FastCorpus, s: &str) -> u64 { let c: Vec<char> = s.chars().collect(); f.bigrams[id(c[0]) * CHARS + id(c[1])] }
  fn tri(f: &FastCorpus, s: &str) -> u64 {
    let c: Vec<char> = s.chars().collect();
    f.trigrams[(id(c[0]) * CHARS + id(c[1])) * CHARS + id(c[2])]
  }

  #[test]
  fn spaces_break_sequences_and_are_not_counted() {
    let f = run("abc  De");
    assert_eq!(f.letters[26], 0);
    assert!(f.bigrams.iter().skip(26).step_by(CHARS).all(|&c| c == 0));
    assert_eq!((bi(&f, "ab"), bi(&f, "bc"), tri(&f, "abc")), (1, 1, 1));
    // 단어 경계를 넘지 않음, 대문자는 소문자로
    assert_eq!((bi(&f, "cd"), tri(&f, "bcd"), bi(&f, "de")), (0, 0, 1));
    assert_eq!(f.letters.iter().sum::<u64>(), 5);
  }

  #[test]
  fn punctuation_and_newlines_break_sequences() {
    let f = run("ab,cd\nef");
    assert_eq!((bi(&f, "bc"), bi(&f, "de"), bi(&f, "cd"), bi(&f, "ef")), (0, 0, 1, 1));
  }
}
