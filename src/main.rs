mod config;
mod corpus;
mod download;
mod dump;
mod layout;
mod optimizer;
mod precomp;
mod score;
mod sensitivity;
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

// a-z와 단어 사이 공백을 셈.
// - 공백은 양옆이 글자인 공백 하나(단어 사이)만 셈: `the end` → he_, e_e, _en.
//   문장부호·숫자 옆(`. The`, `end.` 뒤), 연속 공백, 줄 처음·끝, 마크업 옆 공백은 세지 않고 연속을 끊음
// - 문장부호·숫자 자체는 n-gram에 넣지 않고 끊음
// - 세지 않는 공백도 경우별 개수는 따로 셈 (분석 뒤 확인용 출력)
// - 같은 글자 연속(ll)은 빅그램으로 세지 않고, 그 쌍이 낀 트라이그램도 세지 않음
pub const SPACE_ID: u8 = 26;

// 실제 글의 문장부호와 숫자 (공백 경우별 개수 출력에서 마크업 찌꺼기와 나눠 보이려고 씀)
fn is_text_punct(b: u8) -> bool {
  b.is_ascii_digit() || matches!(b, b'.' | b',' | b';' | b':' | b'!' | b'?' | b')' | b']' | b'(' | b'[' | b'"' | b'\'')
}

// 공백 옆에 오는 것의 종류. Other = 줄 처음·끝, 마크업이나 그 밖의 기호
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side { Letter, Punct, Other }

// 키 연속(사슬)을 이어 가며 n-gram을 셈
struct Counter<'a> {
  stats: &'a mut corpus::FastCorpus,
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

fn analyze_bytes(bytes: &[u8], stats: &mut corpus::FastCorpus) {
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

// 덤프 분석: 받아 둔 파일이 있으면 병렬(멀티스트림)·이어서 분석, 아니면 스트리밍 순차 분석.
// wikipedia_limit > 0이면 앞쪽 N개만 순차로 (빠른 확인용)
fn analyze_dump(result_path: &Path, config: &Config, corpus_path: &Path) -> anyhow::Result<CorpusStats> {
  // 저장할 폴더를 먼저 만듦: 몇 시간짜리 분석이 끝난 뒤 저장 단계에서 실패하지 않도록
  if let Some(dir) = corpus_path.parent() { fs::create_dir_all(dir)?; }
  let urls = config.dump_urls();
  let sample = config.wikipedia_sample;
  let checkpoint = std::path::PathBuf::from(format!("{}.partial", corpus_path.display()));
  // 중간 저장은 병렬 분석을 끝까지 마쳤을 때만 지움 (앞쪽 N개만 보는 빠른 확인이 전체 분석의 진행을 지우지 않도록)
  let mut parallel = false;
  let (fast, counts) = if let Some(dump_path) = config.dump_path(result_path) {
    if !dump_path.exists() {
      if urls.is_empty() { anyhow::bail!("dump file {} not found and wikipedia_dump_url is empty", dump_path.display()); }
      if let Some(dir) = dump_path.parent() { fs::create_dir_all(dir)?; }
      println!("[Download] -> {}", dump_path.display());
      let opts = download::DownloadOpts { verify: config.wikipedia_verify, parallel: config.wikipedia_parallel_download, ..Default::default() };
      download::download_dump(&download::http_client()?, &urls, &dump_path, &opts)?;
    }
    println!("[Wikipedia Dump] Reading {}", dump_path.display());
    if config.wikipedia_limit > 0 {
      wiki::analyze_sequential(fs::File::open(&dump_path)?, sample, config.wikipedia_limit)?
    } else {
      parallel = true;
      wiki::analyze_file_parallel(&dump_path, sample, &checkpoint)?
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
    wiki::analyze_sequential(response, sample, config.wikipedia_limit)?
  };
  println!("[Wikipedia Dump] used {} articles; skipped {} redirects, {} list/disambiguation pages, {} outside sample",
    counts.articles, counts.redirects, counts.lists, counts.sampled_out);
  fast.print_space_cases();
  let stats = fast.into_stats();
  stats.save(corpus_path.to_str().context("Invalid corpus path")?, dump::PARSER_VERSION)?;
  if parallel && checkpoint.exists() { fs::remove_file(&checkpoint)?; }
  Ok(stats)
}

// 채점·최적화에 쓸 수 있는 가장 오래된 파서 버전. 4부터 글자 사이 공백을 세서 엄지 방식의 공백 너머 비용을 계산할 수 있고,
// 스페이스를 배열에 둘 때는 단어 사이 공백만 세는 6부터 (5는 문장부호 옆 공백까지 세서 누름 횟수가 다름).
// 그 전 코퍼스로 채점하면 그 몫이 조용히 빠져 예전·지금 결과를 잘못 비교하게 됨
const MIN_THUMB_PARSER: u32 = 4;
const MIN_GRID_PARSER: u32 = 6;

fn scoring_parser(config: &Config) -> u32 { if config.space_on_grid() { MIN_GRID_PARSER } else { MIN_THUMB_PARSER } }

// 저장된 코퍼스 불러오기 (전체 통계). min_parser보다 오래된 파서로 만든 것은 거부 (0 = 통계만 볼 때). 그 밖에 파서가 다르면 경고
fn load_corpus(corpus_path: &Path, min_parser: u32) -> anyhow::Result<CorpusStats> {
  println!("[Load Corpus] {}", corpus_path.display());
  let (stats, parser_version) = CorpusStats::load(corpus_path.to_str().context("Invalid corpus path")?)?;
  if parser_version < min_parser {
    anyhow::bail!("{} was built with parser v{parser_version}, which does not count spaces the way this mode needs (v{min_parser}+); rebuild it with `analyze --reset`",
      corpus_path.display());
  }
  if parser_version != dump::PARSER_VERSION {
    println!("[Load Corpus] warning: built with parser v{parser_version}, current is v{}; rebuild with `analyze --reset` for the current rules", dump::PARSER_VERSION);
  }
  Ok(stats)
}

// 전체 통계. 코퍼스 파일이 없거나 --reset이면 덤프를 분석해서 저장
fn load_or_analyze(result_path: &Path, config: &Config, reset: bool, min_parser: u32) -> anyhow::Result<CorpusStats> {
  let corpus_path = result_path.join(&config.corpus_file);
  if !reset && corpus_path.exists() { load_corpus(&corpus_path, min_parser) } else { analyze_dump(result_path, config, &corpus_path) }
}

// 최적화·채점용: 빈도가 낮은 빅/트라이그램을 config 기준으로 잘라 냄
fn for_scoring(mut stats: CorpusStats, config: &Config) -> CorpusStats {
  stats.trim_by_coverage(config.bigram_coverage, config.trigram_coverage);
  stats
}

fn print_results(results: &[layout::Layout], scoring: Option<(&corpus::CorpusArrays, &Precomp, &score::ScoreNorm)>) {
  for (i, l) in results.iter().enumerate() {
    println!("[{}]", i + 1);
    l.print();
    if let Some((arrays, precomp, norm)) = scoring { l.print_goodness(norm, arrays, precomp); }
    println!();
  }
}

fn cmd_analyze(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<()> {
  let stats = load_or_analyze(result_path, config, reset, 0)?;
  let used = for_scoring(stats.clone(), config);
  // (글자끼리, 공백이 낀 것) 개수
  let split = |s: &CorpusStats| {
    let b = s.bigrams.keys().filter(|k| k.0 == ' ' || k.1 == ' ').count();
    let t = s.trigrams.keys().filter(|k| k.0 == ' ' || k.1 == ' ' || k.2 == ' ').count();
    ((s.bigrams.len() - b, b), (s.trigrams.len() - t, t))
  };
  let ((b, bs), (t, ts)) = split(&stats);
  let ((ub, ubs), (ut, uts)) = split(&used);
  println!("letters: {}, bigrams: {b} letters-only + {bs} with space (scoring uses {ub} + {ubs} at {}% coverage), trigrams: {t} + {ts} (scoring uses {ut} + {uts} at {}% coverage)",
    stats.letters.keys().filter(|&&c| c != ' ').count(), config.bigram_coverage * 100.0, config.trigram_coverage * 100.0);
  stats.print();
  Ok(())
}

fn cmd_results(result_path: &Path, config: &Config) -> anyhow::Result<()> {
  let result_file = result_path.join(config.result_file_for_mode());
  let result_file_str = result_file.to_str().context("Invalid result file path")?;
  if !result_file.exists() {
    eprintln!("No results found at {}", result_file.display());
    std::process::exit(1);
  }
  let mut results = load_results(result_file_str)?;
  let corpus_path = result_path.join(&config.corpus_file);
  let mut scoring = None;
  if corpus_path.exists() {
    let stats = for_scoring(load_corpus(&corpus_path, scoring_parser(config))?, config);
    let arrays = stats.to_arrays(config.space_on_grid());
    results.retain(|l| l.key_count() == arrays.keys.len());
    let precomp = Precomp::new(&arrays.freq, &config.weights, &config.costs, &config.effort, config.hand_hard_limit());
    let n = init_score_norm(&arrays, &precomp, &config.costs, score::HandRule::from_config(config));
    for l in &mut results {
      l.calc_scores(&arrays, &precomp);
      l.calc_total_score(&n);
    }
    layout::sort_layouts(&mut results);
    scoring = Some((arrays, precomp, n));
  }
  print_results(&results, scoring.as_ref().map(|(a, p, n)| (a, p, n)));
  Ok(())
}

// 배열 파일(행마다 10칸, 빈칸 '.')을 저장된 결과 1등과 같은 기준(같은 정규화 표본)으로 채점해 나란히 출력
fn cmd_score(result_path: &Path, config: &Config, files: &[String]) -> anyhow::Result<()> {
  anyhow::ensure!(!files.is_empty(), "usage: <result_path> score <layout_file>...");
  let corpus_path = result_path.join(&config.corpus_file);
  let full = load_corpus(&corpus_path, scoring_parser(config))?;
  let full_arrays = full.to_arrays(config.space_on_grid());
  let windows = full.windows();
  let arrays = for_scoring(full, config).to_arrays(config.space_on_grid());
  let precomp = Precomp::new(&arrays.freq, &config.weights, &config.costs, &config.effort, config.hand_hard_limit());
  let center_mask = arrays.rare_mask(config.center_rare);
  let norm = init_score_norm(&arrays, &precomp, &config.costs, score::HandRule::from_config(config));

  let mut entries: Vec<(String, layout::Layout)> = Vec::new();
  let result_file = result_path.join(config.result_file_for_mode());
  if result_file.exists() {
    let mut results = load_results(result_file.to_str().context("Invalid result file path")?)?;
    results.retain(|l| l.key_count() == arrays.keys.len());
    for l in &mut results { l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm); }
    layout::sort_layouts(&mut results);
    if let Some(best) = results.first() { entries.push(("best saved result".to_string(), *best)); }
  }
  for f in files {
    let mut l = layout::Layout::parse(&fs::read_to_string(f).with_context(|| format!("reading {f}"))?, config.space_on_grid())
      .with_context(|| format!("parsing {f}"))?;
    anyhow::ensure!(l.key_count() == arrays.keys.len(), "{f} has {} letters, the corpus has {}", l.key_count(), arrays.keys.len());
    if !l.center_ok(center_mask) { println!("[Score] note: {f} does not follow center_rare ({} letters)", config.center_rare); }
    l.calc_scores(&arrays, &precomp);
    l.calc_total_score(&norm);
    entries.push((f.clone(), l));
  }
  for (name, l) in &entries {
    println!("[{name}]");
    l.print();
    l.print_goodness(&norm, &arrays, &precomp);
    l.print_stats(&full_arrays, &precomp, &windows);
    println!();
  }
  Ok(())
}

// 점수 설정값을 하나씩 흔들어 저장된 결과와 배열 파일들의 1등이 바뀌는지 점검 (설정·결과 파일은 바꾸지 않음)
fn cmd_sensitivity(result_path: &Path, config: &Config, files: &[String]) -> anyhow::Result<()> {
  let arrays = for_scoring(load_corpus(&result_path.join(&config.corpus_file), scoring_parser(config))?, config).to_arrays(config.space_on_grid());
  let precomp = Precomp::new(&arrays.freq, &config.weights, &config.costs, &config.effort, config.hand_hard_limit());
  let center_mask = arrays.rare_mask(config.center_rare);
  let norm = init_score_norm(&arrays, &precomp, &config.costs, score::HandRule::from_config(config));
  let mut cands: Vec<(String, layout::Layout)> = Vec::new();
  let result_file = result_path.join(config.result_file_for_mode());
  if result_file.exists() {
    let mut results = load_results(result_file.to_str().context("Invalid result file path")?)?;
    results.retain(|l| l.key_count() == arrays.keys.len());
    for l in &mut results { l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm); }
    layout::sort_layouts(&mut results);
    cands.extend(results.into_iter().enumerate().map(|(i, l)| (format!("saved #{}", i + 1), l)));
  }
  for f in files {
    let l = layout::Layout::parse(&fs::read_to_string(f).with_context(|| format!("reading {f}"))?, config.space_on_grid())
      .with_context(|| format!("parsing {f}"))?;
    anyhow::ensure!(l.key_count() == arrays.keys.len(), "{f} has {} letters, the corpus has {}", l.key_count(), arrays.keys.len());
    if !l.center_ok(center_mask) { println!("[Sensitivity] note: {f} does not follow center_rare ({} letters)", config.center_rare); }
    cands.push((f.clone(), l));
  }
  anyhow::ensure!(!cands.is_empty(), "no layouts to check: run optimize first or pass layout files");
  sensitivity::run(&arrays, config, cands, center_mask);
  Ok(())
}

fn run_optimize(result_path: &Path, config: &Config, stats: &corpus::CorpusStats, reset: bool) -> anyhow::Result<()> {
  let arrays = stats.to_arrays(config.space_on_grid());
  print!("[Precomp] ..."); let _ = std::io::stdout().flush();
  let precomp = Precomp::new(&arrays.freq, &config.weights, &config.costs, &config.effort, config.hand_hard_limit());
  println!(" done");
  let center_mask = arrays.rare_mask(config.center_rare);
  print!("[Score Norm] ..."); let _ = std::io::stdout().flush();
  let norm = init_score_norm(&arrays, &precomp, &config.costs, score::HandRule::from_config(config));
  println!(" done");
  if center_mask != 0 {
    let letters: String = (0..arrays.keys.len()).filter(|&i| center_mask >> i & 1 == 1).map(|i| arrays.keys[i]).collect();
    println!("[Center] least frequent {} letters forced into the center columns: {letters}", config.center_rare);
  }
  let result_file = result_path.join(config.result_file_for_mode());
  let result_file_str = result_file.to_str().context("Invalid result file path")?;
  let base_layouts = if !reset && result_file.exists() {
    let mut layouts = load_results(result_file_str)?;
    // 키 구성이 다른 결과(스페이스 방식이 다를 때)는 시드로 쓰지 않음
    let before = layouts.len();
    layouts.retain(|l| l.key_count() == arrays.keys.len());
    if layouts.len() < before {
      println!("[Results] skipped {} saved layouts with a different key set (made with the other space_key setting)", before - layouts.len());
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
  let optimizer = Optimizer::new(arrays, &precomp, &norm, config.elites, config.min_home_diff, config.sa.clone(), center_mask);
  println!("[Optimize]");
  let layouts = optimizer.run(&base_layouts, result_file_str);
  let mut combined = base_layouts;
  combined.extend(layouts);
  let best = sort_unique_layouts(combined, config.elites, &[], config.min_home_diff);
  save_results(&best, result_file_str)?;
  println!("\n[Results]");
  print_results(&best, Some((optimizer.corpus(), &precomp, &norm)));
  Ok(())
}

fn cmd_run(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<()> {
  let stats = for_scoring(load_or_analyze(result_path, config, reset, scoring_parser(config))?, config);
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
    eprintln!("Usage: {} <result_path> [analyze|optimize|run|results|score <layout_file>...|sensitivity [layout_file]...] [--reset]", args[0]);
    std::process::exit(1);
  }
  let result_path = Path::new(&args[1]);
  let cmd = args.get(2).map(std::string::String::as_str);
  let reset = args.iter().any(|a| a == "--reset") && !matches!(cmd, Some("results" | "score" | "sensitivity"));
  let config = load_config(Path::new("config.toml"))?;
  // 최적화의 모든 난수(시작 배열, 담금질, 교차)가 이 시드에서 나옴
  let seed = config.seed.unwrap_or_else(|| fastrand::u64(..));
  println!("[Seed] {seed}");
  fastrand::seed(seed);
  if reset { fs::create_dir_all(result_path)?; }
  match cmd {
    Some("analyze")  => cmd_analyze(result_path, &config, reset),
    Some("optimize") => {
      let stats = for_scoring(load_corpus(&result_path.join(&config.corpus_file), scoring_parser(&config))?, &config);
      run_optimize(result_path, &config, &stats, reset)
    },
    Some("run")      => cmd_run(result_path, &config, reset),
    Some("results")  => cmd_results(result_path, &config),
    Some("score")    => cmd_score(result_path, &config, &args[3..].iter().filter(|a| *a != "--reset").cloned().collect::<Vec<_>>()),
    Some("sensitivity") => cmd_sensitivity(result_path, &config, &args[3..].iter().filter(|a| *a != "--reset").cloned().collect::<Vec<_>>()),
    _ => {
      eprintln!("Usage: {} <result_path> [analyze|optimize|run|results|score <layout_file>...|sensitivity [layout_file]...] [--reset]", args[0]);
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

  const SP: char = ' ';
  fn idc(c: char) -> usize { if c == SP { 26 } else { id(c) } }
  fn bi_s(f: &FastCorpus, a: char, b: char) -> u64 { f.bigrams[idc(a) * CHARS + idc(b)] }
  fn tri_s(f: &FastCorpus, a: char, b: char, c: char) -> u64 { f.trigrams[(idc(a) * CHARS + idc(b)) * CHARS + idc(c)] }

  #[test]
  fn one_space_between_letters_is_a_thumb_key() {
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
    use std::io::Write;
    let mut xml = String::from("<mediawiki>\n");
    for i in 0..5 { xml.push_str(&format!("<page><title>Article {i}</title><ns>0</ns><revision><text>Some words here {i}.</text></revision></page>\n")); }
    xml.push_str("</mediawiki>\n");
    let mut e = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
    e.write_all(xml.as_bytes()).unwrap();
    std::fs::write(path, e.finish().unwrap()).unwrap();
  }

  fn test_config(dump_file: &std::path::Path, limit: usize) -> crate::config::Config {
    let mut c = crate::config::load_config(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml"))).unwrap();
    c.wikipedia_dump_file = Some(dump_file.to_str().unwrap().to_string());
    c.wikipedia_limit = limit;
    c
  }

  #[test]
  fn analyze_creates_missing_result_dir_and_keeps_checkpoint_for_quick_runs() {
    let root = std::env::temp_dir().join(format!("p2anut-main-{}", std::process::id()));
    let dumps = root.join("dumps");
    std::fs::create_dir_all(&dumps).unwrap();
    let dump = dumps.join("d.xml.bz2");
    tiny_dump(&dump);

    // 결과 폴더가 없어도 (덤프는 다른 곳) 분석 후 저장까지 됨
    let result = root.join("new").join("result");
    let corpus = result.join("corpus.bin");
    let stats = super::analyze_dump(&result, &test_config(&dump, 2), &corpus).unwrap();
    assert!(corpus.exists() && stats.letters.values().sum::<u64>() > 0);

    // 앞쪽 N개만 보는 빠른 확인은 전체 분석의 중간 저장을 지우지 않음
    let checkpoint = std::path::PathBuf::from(format!("{}.partial", corpus.display()));
    std::fs::write(&checkpoint, b"in-progress").unwrap();
    super::analyze_dump(&result, &test_config(&dump, 2), &corpus).unwrap();
    assert_eq!(std::fs::read(&checkpoint).unwrap(), b"in-progress");

    // 병렬 분석을 끝까지 마치면 지움 (내용이 맞지 않는 중간 저장은 무시하고 처음부터 분석)
    super::analyze_dump(&result, &test_config(&dump, 0), &corpus).unwrap();
    assert!(!checkpoint.exists());
    std::fs::remove_dir_all(&root).unwrap();
  }

  #[test]
  fn scoring_refuses_corpora_built_before_spaces_were_counted() {
    let dir = std::env::temp_dir().join(format!("p2anut-v3-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("corpus.bin");
    let stats = crate::test_util::stats();
    stats.save(path.to_str().unwrap(), super::MIN_THUMB_PARSER - 1).unwrap();
    let err = super::load_corpus(&path, super::MIN_THUMB_PARSER).unwrap_err().to_string();
    assert!(err.contains("does not count spaces") && err.contains("analyze --reset"), "{err}");
    // 통계만 볼 때(analyze)는 읽힘
    assert_eq!(super::load_corpus(&path, 0).unwrap(), stats);
    // 엄지 방식은 v4부터, 스페이스를 배열에 두면 v6부터
    stats.save(path.to_str().unwrap(), super::MIN_THUMB_PARSER).unwrap();
    assert_eq!(super::load_corpus(&path, super::MIN_THUMB_PARSER).unwrap(), stats);
    assert!(super::load_corpus(&path, super::MIN_GRID_PARSER).is_err());
    stats.save(path.to_str().unwrap(), crate::dump::PARSER_VERSION).unwrap();
    assert_eq!(super::load_corpus(&path, super::MIN_GRID_PARSER).unwrap(), stats);
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn punctuation_and_newlines_break_sequences() {
    let f = run("ab,cd\nef");
    assert_eq!((bi(&f, "bc"), bi(&f, "de"), bi(&f, "cd"), bi(&f, "ef")), (0, 0, 1, 1));
  }
}
