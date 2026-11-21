mod config;
mod corpus;
mod download;
mod dump;
mod layout;
mod optimizer;
mod precomp;
mod score;
mod wiki;
#[cfg(test)]
mod test_util;

use std::path::Path;
use std::io::Write;
use std::fs;
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

// 덤프 분석: 받아 둔 파일이 있으면 병렬(멀티스트림)·이어서 분석, 아니면 스트리밍 순차 분석.
// wikipedia_limit > 0이면 앞쪽 N개만 순차로 (빠른 확인용)
fn analyze_dump(result_path: &Path, config: &Config, corpus_path: &Path) -> anyhow::Result<CorpusStats> {
  let urls = config.dump_urls();
  let sample = config.wikipedia_sample;
  let checkpoint = std::path::PathBuf::from(format!("{}.partial", corpus_path.display()));
  let (fast, counts) = if let Some(dump_path) = config.dump_path(result_path) {
    if !dump_path.exists() {
      if urls.is_empty() { anyhow::bail!("dump file {} not found and wikipedia_dump_url is empty", dump_path.display()); }
      if let Some(dir) = dump_path.parent() { fs::create_dir_all(dir)?; }
      println!("[Download] -> {}", dump_path.display());
      let opts = download::DownloadOpts { verify: config.wikipedia_verify, ..Default::default() };
      download::download_dump(&download::http_client()?, &urls, &dump_path, &opts)?;
    }
    println!("[Wikipedia Dump] Reading {}", dump_path.display());
    if config.wikipedia_limit > 0 {
      wiki::analyze_sequential(fs::File::open(&dump_path)?, sample, config.wikipedia_limit)?
    } else {
      wiki::analyze_file_parallel(&dump_path, sample, &checkpoint)?
    }
  } else {
    println!("[Wikipedia Dump] streaming: single thread, no resume (set wikipedia_download = true for both)");
    let client = download::http_client()?;
    let url = if urls.len() > 1 {
      let opts = download::DownloadOpts::default();
      download::rank_mirrors(&client, &urls, opts.probe_bytes, opts.probe_timeout)?.remove(0).url
    } else { urls[0].clone() };
    println!("[Wikipedia Dump] Connecting to {url}");
    let response = client.get(&url).send()?.error_for_status()?;
    wiki::analyze_sequential(response, sample, config.wikipedia_limit)?
  };
  println!("[Wikipedia Dump] used {} articles; skipped {} redirects, {} list/disambiguation pages, {} outside sample",
    counts.articles, counts.redirects, counts.lists, counts.sampled_out);
  let stats = fast.into_stats();
  stats.save(corpus_path.to_str().context("Invalid corpus path")?, dump::PARSER_VERSION)?;
  if checkpoint.exists() { fs::remove_file(&checkpoint)?; }
  Ok(stats)
}

// 저장된 코퍼스 불러오기 (전체 통계). 예전 파서로 만든 것이면 경고
fn load_corpus(corpus_path: &Path) -> anyhow::Result<CorpusStats> {
  println!("[Load Corpus] {}", corpus_path.display());
  let (stats, parser_version) = CorpusStats::load(corpus_path.to_str().context("Invalid corpus path")?)?;
  if parser_version != dump::PARSER_VERSION {
    println!("[Load Corpus] warning: built with parser v{parser_version}, current is v{}; rebuild with `analyze --reset` for the current rules", dump::PARSER_VERSION);
  }
  Ok(stats)
}

// 전체 통계. 코퍼스 파일이 없거나 --reset이면 덤프를 분석해서 저장
fn load_or_analyze(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<CorpusStats> {
  let corpus_path = result_path.join(&config.corpus_file);
  if !reset && corpus_path.exists() { load_corpus(&corpus_path) } else { analyze_dump(result_path, config, &corpus_path) }
}

// 최적화·채점용: 빈도가 낮은 빅/트라이그램을 config 기준으로 잘라 냄
fn for_scoring(mut stats: CorpusStats, config: &Config) -> CorpusStats {
  stats.trim_by_coverage(config.bigram_coverage, config.trigram_coverage);
  stats
}

fn print_results(results: &[layout::Layout], norm: Option<&score::ScoreNorm>) {
  for (i, l) in results.iter().enumerate() {
    println!("[{}]", i + 1);
    l.print();
    if let Some(n) = norm { l.print_goodness(n); }
    println!();
  }
}

fn cmd_analyze(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<()> {
  let stats = load_or_analyze(result_path, config, reset)?;
  let used = for_scoring(stats.clone(), config);
  println!("letters: {}, bigrams: {} (scoring uses {} at {}% coverage), trigrams: {} (scoring uses {} at {}% coverage)",
    stats.letters.len(), stats.bigrams.len(), used.bigrams.len(), config.bigram_coverage * 100.0,
    stats.trigrams.len(), used.trigrams.len(), config.trigram_coverage * 100.0);
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
  let mut norm = None;
  if corpus_path.exists() {
    let stats = for_scoring(load_corpus(&corpus_path)?, config);
    let arrays = stats.to_arrays(false);
    results.retain(|l| l.key_count() == arrays.keys.len());
    let precomp = Precomp::new(&arrays.freq, &config.weights, config.min_hand_usage);
    let n = init_score_norm(&arrays, &precomp, &config.rates);
    for l in &mut results {
      l.calc_scores(&arrays, &precomp);
      l.calc_total_score(&n);
    }
    layout::sort_layouts(&mut results);
    norm = Some(n);
  }
  print_results(&results, norm.as_ref());
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
  print_results(&best, Some(&norm));
  Ok(())
}

fn cmd_run(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<()> {
  let stats = for_scoring(load_or_analyze(result_path, config, reset)?, config);
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
      let stats = for_scoring(load_corpus(&result_path.join(&config.corpus_file))?, &config);
      run_optimize(result_path, &config, &stats, reset)
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
