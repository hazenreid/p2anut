mod code;
mod config;
mod corpus;
mod download;
mod dump;
mod keycorpus;
mod layout;
mod optimizer;
mod precomp;
mod score;
mod sensitivity;
mod symbols;
mod wiki;
#[cfg(test)]
mod test_util;

use std::path::Path;
use std::fs;
use anyhow::Context;
use std::sync::atomic::{AtomicBool, Ordering};

pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);

use config::{load_config, Config};
use corpus::{CorpusArrays, CorpusStats};
use precomp::Precomp;
use score::ScoreNorm;
use layout::{Layout, save_results, load_results, sort_unique_layouts};
use optimizer::Optimizer;

// 스페이스 키 번호 (a-z 다음). 글자 코퍼스를 세는 규칙은 wiki.rs의 analyze_bytes
pub const SPACE_ID: u8 = 26;

// 채점·최적화에 쓸 수 있는 가장 오래된 파서 버전: 단어 사이 공백만 세는 6부터 (스페이스 키도 배열에 두므로.
// 5는 문장부호 옆 공백까지 세서 누름 횟수가 다르고, 그 전에는 공백을 세지 않음)
const MIN_SCORING_PARSER: u32 = 6;

// 저장된 코퍼스 불러오기 (전체 통계). min_parser보다 오래된 파서로 만든 것은 거부 (0 = 통계만 볼 때). 그 밖에 파서가 다르면 경고
fn load_corpus(corpus_path: &Path, min_parser: u32) -> anyhow::Result<CorpusStats> {
  println!("[Load Corpus] {}", corpus_path.display());
  let (stats, parser_version) = CorpusStats::load(corpus_path.to_str().context("Invalid corpus path")?)?;
  if parser_version < min_parser {
    anyhow::bail!("{} was built with parser v{parser_version}, which does not count spaces the way scoring needs (v{min_parser}+); rebuild it with `analyze --reset`",
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
  if !reset && corpus_path.exists() { load_corpus(&corpus_path, min_parser) } else { wiki::analyze_dump(result_path, config, &corpus_path) }
}

// 최적화·채점용: 빈도가 낮은 빅/트라이그램을 config 기준으로 잘라 냄
fn for_scoring(mut stats: CorpusStats, config: &Config) -> CorpusStats {
  stats.trim_by_coverage(config.bigram_coverage, config.trigram_coverage);
  stats
}

// 글자 배열에 섞을 러스트 코드: 위키와 같은 공백 규칙으로 센 코드 코퍼스 (code_share가 0이면 None). 코드 코퍼스가 없거나 오래됐으면 만듦
fn code_letters(result_path: &Path, config: &Config) -> anyhow::Result<Option<CorpusStats>> {
  if config.code_share <= 0.0 { return Ok(None); }
  Ok(Some(code::load_or_build(result_path, config)?.letters))
}

// 채점 기준: 코퍼스 배열과, 지금 설정으로 만든 표와 총점 기준
struct Scoring {
  arrays: CorpusArrays,
  precomp: Precomp,
  norm: ScoreNorm,
  // 코드를 섞었으면 영어만·코드만의 기준과 코드 비율 (비교 출력용)
  parts: Vec<(&'static str, Scoring)>,
  share: f64,
}

impl Scoring {
  // 코퍼스 하나 (빈도가 낮은 n-gram은 잘라 냄)
  fn of(stats: &CorpusStats, config: &Config) -> Self {
    let arrays = for_scoring(stats.clone(), config).to_arrays();
    let (precomp, norm) = layout::scoring(&arrays, config);
    Scoring { arrays, precomp, norm, parts: Vec::new(), share: 0.0 }
  }

  // 글자 배열의 기준: 위키 영어(전체 통계)에 코드를 code_share만큼(글자 수 기준) 섞음
  fn new(english: &CorpusStats, code: Option<&CorpusStats>, config: &Config) -> Self {
    let Some(code) = code else { return Self::of(english, config) };
    let mut s = Self::of(&english.mixed(code, config.code_share), config);
    s.parts = vec![("english", Self::of(english, config)), ("code", Self::of(code, config))];
    s.share = config.code_share;
    s
  }

  // 저장된 코퍼스로 (code_share가 0보다 크면 코드 코퍼스도)
  fn load(result_path: &Path, config: &Config) -> anyhow::Result<Self> {
    let english = load_corpus(&result_path.join(&config.corpus_file), MIN_SCORING_PARSER)?;
    Ok(Self::new(&english, code_letters(result_path, config)?.as_ref(), config))
  }

  // 코퍼스 파일이 있을 때만
  fn load_if_present(result_path: &Path, config: &Config) -> anyhow::Result<Option<Self>> {
    if result_path.join(&config.corpus_file).exists() { Self::load(result_path, config).map(Some) } else { Ok(None) }
  }

  fn score(&self, l: &mut Layout) {
    l.calc_scores(&self.arrays, &self.precomp);
    l.calc_total_score(&self.norm);
  }

  // 글자 100개당 항목별 비용. 코드를 섞었으면 영어만·코드만의 총비용도 다음 줄에
  fn print_goodness(&self, l: &Layout) {
    l.print_goodness(&self.norm, &self.arrays, &self.precomp);
    if self.parts.is_empty() { return; }
    let total = |s: &Scoring| {
      let mut x = *l;
      s.score(&mut x);
      if x.total <= layout::PENALTY_SCORE { "disqualified".to_string() } else { format!("{:.1}", score::cost_per_100(x.total)) }
    };
    println!("  {}  (the total mixes them {:.0}:{:.0} by letters)", self.parts.iter().map(|(name, s)| format!("{name}={}", total(s))).collect::<Vec<_>>().join("  "),
      (1.0 - self.share) * 100.0, self.share * 100.0);
  }
}

// 저장된 결과 (결과 파일이 없으면 빈 목록). 채점 기준이 있으면 그 코퍼스와 키 구성이 같은 것만 지금 설정으로 다시 채점해서
// 정렬 (results 명령의 순서)
fn saved_results(result_path: &Path, config: &Config, scoring: Option<&Scoring>) -> anyhow::Result<Vec<Layout>> {
  let file = config.result_file_in(result_path);
  if !file.exists() { return Ok(Vec::new()); }
  let mut results = load_results(file.to_str().context("Invalid result file path")?)?;
  if let Some(s) = scoring {
    results.retain(|l| l.key_count() == s.arrays.keys.len());
    for l in &mut results { s.score(l); }
    layout::sort_layouts(&mut results);
  }
  Ok(results)
}

// 결과 파일이 없을 때의 안내. 코드를 섞으면 결과 파일이 따로라서, 섞기 전의 결과가 있어도 optimize를 먼저 돌려야 함
fn no_results(result_path: &Path, config: &Config) -> String {
  let mut msg = format!("no results at {}: run optimize first", config.result_file_in(result_path).display());
  let plain = result_path.join(&config.result_file);
  if config.code_share > 0.0 && plain.exists() {
    msg += &format!(" (code_share {} keeps its own results; the results without code are in {} and show with code_share = 0)", config.code_share, plain.display());
  }
  msg
}

// 배열 파일 읽기: 행마다 10칸씩 3행 (스페이스 키는 '_', 빈칸은 '.')
fn read_layout(file: &str) -> anyhow::Result<Layout> {
  Layout::parse(&fs::read_to_string(file).with_context(|| format!("reading {file}"))?).with_context(|| format!("parsing {file}"))
}

// 채점할 배열 파일들: 키 구성이 코퍼스와 같아야 함. center_rare를 어기면 알리기만 함
fn layout_files(files: &[String], s: &Scoring, config: &Config, tag: &str) -> anyhow::Result<Vec<(String, Layout)>> {
  let center_mask = s.arrays.rare_mask(config.center_rare);
  files.iter().map(|f| {
    let l = read_layout(f)?;
    anyhow::ensure!(l.key_count() == s.arrays.keys.len(), "{f} has {} letters, the corpus has {}", l.key_count(), s.arrays.keys.len());
    if !l.center_ok(center_mask) { println!("[{tag}] note: {f} does not follow center_rare ({} letters)", config.center_rare); }
    Ok((f.clone(), l))
  }).collect()
}

fn print_results(results: &[Layout], scoring: Option<&Scoring>) {
  for (i, l) in results.iter().enumerate() {
    println!("[{}]", i + 1);
    l.print();
    if let Some(s) = scoring { s.print_goodness(l); }
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
  let result_file = config.result_file_in(result_path);
  if !result_file.exists() {
    eprintln!("{}", no_results(result_path, config));
    std::process::exit(1);
  }
  println!("[Results] {}", result_file.display());
  let scoring = Scoring::load_if_present(result_path, config)?;
  print_results(&saved_results(result_path, config, scoring.as_ref())?, scoring.as_ref());
  Ok(())
}

// 배열 파일(행마다 10칸, 빈칸 '.')을 저장된 결과 1등과 같은 기준(같은 정규화 표본)으로 채점해 나란히 출력
fn cmd_score(result_path: &Path, config: &Config, files: &[String]) -> anyhow::Result<()> {
  anyhow::ensure!(!files.is_empty(), "usage: <result_path> score <layout_file>...");
  let english = load_corpus(&result_path.join(&config.corpus_file), MIN_SCORING_PARSER)?;
  let code = code_letters(result_path, config)?;
  // 행·손가락 통계는 채점과 같은 코퍼스(코드를 섞었으면 섞은 것)의 전체 통계로
  let full = code.as_ref().map_or_else(|| english.clone(), |c| english.mixed(c, config.code_share));
  let (full_arrays, windows) = (full.to_arrays(), full.windows());
  let s = Scoring::new(&english, code.as_ref(), config);
  let mut entries: Vec<(String, Layout)> = saved_results(result_path, config, Some(&s))?.into_iter().take(1)
    .map(|l| ("best saved result".to_string(), l)).collect();
  for (f, mut l) in layout_files(files, &s, config, "Score")? {
    s.score(&mut l);
    entries.push((f, l));
  }
  for (name, l) in &entries {
    println!("[{name}]");
    l.print();
    s.print_goodness(l);
    l.print_stats(&full_arrays, &s.precomp, &windows);
    println!();
  }
  Ok(())
}

// 점수 설정값을 하나씩 흔들어 저장된 결과와 배열 파일들의 1등이 바뀌는지 점검 (설정·결과 파일은 바꾸지 않음)
fn cmd_sensitivity(result_path: &Path, config: &Config, files: &[String]) -> anyhow::Result<()> {
  let s = Scoring::load(result_path, config)?;
  let mut cands: Vec<(String, Layout)> = saved_results(result_path, config, Some(&s))?.into_iter().enumerate()
    .map(|(i, l)| (format!("saved #{}", i + 1), l)).collect();
  cands.extend(layout_files(files, &s, config, "Sensitivity")?);
  anyhow::ensure!(!cands.is_empty(), "no layouts to check: run optimize first or pass layout files");
  sensitivity::run(&s.arrays, config, cands, s.arrays.rare_mask(config.center_rare));
  Ok(())
}

// english: 위키 영어 코퍼스의 전체 통계 (code_share가 0보다 크면 코드를 섞어서 최적화)
fn run_optimize(result_path: &Path, config: &Config, english: &CorpusStats, reset: bool) -> anyhow::Result<()> {
  let s = Scoring::new(english, code_letters(result_path, config)?.as_ref(), config);
  let result_file = config.result_file_in(result_path);
  if config.code_share > 0.0 {
    println!("[Corpus] Wikipedia English with {:.0}% Rust code (by letters) -> {}", config.code_share * 100.0, result_file.display());
  }
  let center_mask = s.arrays.rare_mask(config.center_rare);
  if center_mask != 0 {
    let letters: String = (0..s.arrays.keys.len()).filter(|&i| center_mask >> i & 1 == 1).map(|i| s.arrays.keys[i]).collect();
    println!("[Center] least frequent {} letters forced into the center columns: {letters}", config.center_rare);
  }
  let result_file_str = result_file.to_str().context("Invalid result file path")?;
  let base_layouts = if !reset && result_file.exists() {
    let mut layouts = load_results(result_file_str)?;
    // 키 구성이 다른 결과(스페이스 방식이 다를 때)는 시드로 쓰지 않음
    let before = layouts.len();
    layouts.retain(|l| l.key_count() == s.arrays.keys.len());
    if layouts.len() < before {
      println!("[Results] skipped {} saved layouts with a different key set (made with the old thumb space key)", before - layouts.len());
    }
    // center_rare를 어기는 배열은 구역 안에서만 움직이는 SA로는 못 고치므로 시드로 쓰지 않음
    let before = layouts.len();
    layouts.retain(|l| l.center_ok(center_mask));
    if layouts.len() < before {
      println!("[Results] skipped {} saved layouts that break center_rare", before - layouts.len());
    }
    for l in &mut layouts { s.score(l); }
    layouts
  } else { vec![] };
  let optimizer = Optimizer::new(s.arrays.clone(), &s.precomp, &s.norm, config.elites, config.min_home_diff, config.sa.clone(), center_mask);
  println!("[Optimize]");
  let layouts = optimizer.run(&base_layouts, result_file_str);
  let mut combined = base_layouts;
  combined.extend(layouts);
  let best = sort_unique_layouts(combined, config.elites, &[], config.min_home_diff);
  save_results(&best, result_file_str)?;
  println!("\n[Results]");
  print_results(&best, Some(&s));
  Ok(())
}

fn cmd_run(result_path: &Path, config: &Config, reset: bool) -> anyhow::Result<()> {
  let english = load_or_analyze(result_path, config, reset, MIN_SCORING_PARSER)?;
  run_optimize(result_path, config, &english, reset)
}

// 심볼 레이어 최적화: 저장된 결과 N번(기본 1, results와 같은 순서) 또는 배열 파일을 베이스로. --flip이면 베이스를 좌우로 뒤집어서
fn cmd_symbols(result_path: &Path, config: &Config, args: &[String]) -> anyhow::Result<()> {
  let flip = args.iter().any(|a| a == "--flip");
  let target = args.iter().find(|a| !a.starts_with("--")).map_or("1", String::as_str);
  let (mut base, label, what) = if let Ok(n) = target.parse::<usize>() {
    let result_file = config.result_file_in(result_path);
    anyhow::ensure!(result_file.exists(), "{}, or pass a layout file", no_results(result_path, config));
    let results = saved_results(result_path, config, Scoring::load_if_present(result_path, config)?.as_ref())?;
    anyhow::ensure!((1..=results.len()).contains(&n), "there are {} saved results; pick 1..={}", results.len(), results.len());
    // 코드를 섞은 결과면 그 표시를 붙임 (symbols-code50-1.txt): 비율마다 N번 배열이 다름
    let label = config.share_tag().map_or_else(|| n.to_string(), |t| format!("{t}-{n}"));
    (results[n - 1], label, format!("saved result #{n} of {}", result_file.display()))
  } else {
    let stem = Path::new(target).file_stem().map_or_else(|| "layout".to_string(), |s| s.to_string_lossy().into_owned());
    (read_layout(target)?, stem, target.to_string())
  };
  anyhow::ensure!(base.key_count() == 27, "the base layout must hold a-z and the space key (_)");
  if flip { base = base.mirrored(); }
  println!("[Symbols] base: {what}{}", if flip { ", mirrored" } else { "" });
  symbols::optimize(result_path, config, &base, &format!("{label}{}", if flip { "-flip" } else { "" }))
}

const USAGE: &str = "[analyze|optimize|run|results|score <layout_file>...|sensitivity [layout_file]...|analyze-code|symbols [N|layout_file] [--flip]|symbols-score <symbols_file>...] [--reset]";

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
    eprintln!("Usage: {} <result_path> {USAGE}", args[0]);
    std::process::exit(1);
  }
  let result_path = Path::new(&args[1]);
  let cmd = args.get(2).map(std::string::String::as_str);
  let reset = args.iter().any(|a| a == "--reset") && !matches!(cmd, Some("results" | "score" | "sensitivity" | "analyze-code" | "symbols" | "symbols-score"));
  let config = load_config(Path::new("config.toml"))?;
  // 최적화의 모든 난수(시작 배열, 담금질, 교차)가 이 시드에서 나옴
  let seed = config.seed.unwrap_or_else(|| fastrand::u64(..));
  println!("[Seed] {seed}");
  fastrand::seed(seed);
  if reset { fs::create_dir_all(result_path)?; }
  match cmd {
    Some("analyze")  => cmd_analyze(result_path, &config, reset),
    Some("optimize") => run_optimize(result_path, &config, &load_corpus(&result_path.join(&config.corpus_file), MIN_SCORING_PARSER)?, reset),
    Some("run")      => cmd_run(result_path, &config, reset),
    Some("results")  => cmd_results(result_path, &config),
    Some("score")    => cmd_score(result_path, &config, &args[3..].iter().filter(|a| *a != "--reset").cloned().collect::<Vec<_>>()),
    Some("sensitivity") => cmd_sensitivity(result_path, &config, &args[3..].iter().filter(|a| *a != "--reset").cloned().collect::<Vec<_>>()),
    Some("analyze-code") => code::build_corpus(result_path, &config).map(|_| ()),
    Some("symbols") => cmd_symbols(result_path, &config, &args[3..]),
    Some("symbols-score") => symbols::score_files(result_path, &config, &args[3..]),
    _ => {
      eprintln!("Usage: {} <result_path> {USAGE}", args[0]);
      std::process::exit(1);
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::layout::{self, Layout};

  #[test]
  fn scoring_refuses_corpora_built_before_spaces_were_counted() {
    let dir = std::env::temp_dir().join(format!("p2anut-v3-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("corpus.bin");
    let stats = crate::test_util::stats();
    stats.save(path.to_str().unwrap(), super::MIN_SCORING_PARSER - 1).unwrap();
    let err = super::load_corpus(&path, super::MIN_SCORING_PARSER).unwrap_err().to_string();
    assert!(err.contains("does not count spaces") && err.contains("analyze --reset"), "{err}");
    // 통계만 볼 때(analyze)는 읽힘
    assert_eq!(super::load_corpus(&path, 0).unwrap(), stats);
    stats.save(path.to_str().unwrap(), crate::dump::PARSER_VERSION).unwrap();
    assert_eq!(super::load_corpus(&path, super::MIN_SCORING_PARSER).unwrap(), stats);
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn saved_results_are_rescored_and_sorted_when_a_corpus_is_given() {
    let dir = std::env::temp_dir().join(format!("p2anut-saved-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = crate::config::load_config(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml"))).unwrap();
    let s = super::Scoring::new(&crate::test_util::stats(), None, &config);
    // 결과 파일이 없으면 빈 목록
    assert!(super::saved_results(&dir, &config, Some(&s)).unwrap().is_empty());
    // 무작위 배열 5개와, 키 구성이 다른 배열 하나 (스페이스 키를 뺀 것)
    let mut ls: Vec<Layout> = (0..5).map(|_| Layout::make_random(&s.arrays.keys, &[], 0)).collect();
    let mut odd = ls[0];
    odd.slots[odd.pos[crate::SPACE_ID as usize] as usize] = layout::EMPTY;
    odd.pos[crate::SPACE_ID as usize] = layout::EMPTY;
    ls.push(odd);
    layout::save_results(&ls, config.result_file_in(&dir).to_str().unwrap()).unwrap();
    // 채점 기준이 없으면 파일 그대로
    let slots = |v: &[Layout]| v.iter().map(|l| l.slots).collect::<Vec<_>>();
    assert_eq!(slots(&super::saved_results(&dir, &config, None).unwrap()), slots(&ls));
    // 있으면 키 구성이 같은 것만 지금 설정으로 다시 채점해서 높은 점수부터
    let sorted = super::saved_results(&dir, &config, Some(&s)).unwrap();
    assert_eq!(sorted.len(), 5);
    for l in &sorted { let mut again = *l; s.score(&mut again); assert_eq!(again.total, l.total); }
    assert!(sorted.windows(2).all(|w| w[0].total >= w[1].total));
    assert!(sorted.iter().all(|l| ls[..5].iter().any(|o| o.slots == l.slots)));
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn missing_results_point_to_optimize_and_the_results_without_code() {
    let dir = std::env::temp_dir().join(format!("p2anut-none-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = crate::config::load_config(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml"))).unwrap();
    // config.toml의 비율과 상관없이 코드를 절반 섞는 설정으로
    config.code_share = 0.5;
    let msg = super::no_results(&dir, &config);
    assert!(msg.contains("result-grid-code50.bin: run optimize first") && !msg.contains("without code"), "{msg}");
    // 섞기 전 결과가 있으면 그 파일과 보는 법도 알려 줌
    std::fs::write(dir.join(&config.result_file), b"").unwrap();
    let msg = super::no_results(&dir, &config);
    assert!(msg.contains("without code are in") && msg.contains("result-grid.bin") && msg.contains("code_share = 0"), "{msg}");
    config.code_share = 0.0;
    assert!(!super::no_results(&dir, &config).contains("without code"));
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn code_is_mixed_in_by_letters_and_shown_by_part() {
    let mut config = crate::config::load_config(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml"))).unwrap();
    (config.bigram_coverage, config.trigram_coverage) = (1.0, 1.0);
    // 손 비율 감점은 섞은 뒤의 비율로 매겨 평균이 되지 않으므로 끔
    (config.min_hand_usage, config.hand_hard_limit) = (0.0, Some(0.0));
    let english = crate::test_util::stats();
    let mut f = crate::corpus::FastCorpus::new();
    crate::wiki::analyze_bytes(&b"pub fn new self let mut value impl from str usize index expect unwrap\n".repeat(50), &mut f);
    let code = f.into_stats();
    // 섞지 않으면 영어만
    let plain = super::Scoring::new(&english, None, &config);
    assert!(plain.parts.is_empty());
    let s = super::Scoring::new(&english, Some(&code), &config);
    assert_eq!(s.parts.iter().map(|p| p.0).collect::<Vec<_>>(), ["english", "code"]);
    assert_eq!(s.parts[0].1.arrays.freq, plain.arrays.freq);
    // 글자 수 기준으로 섞으므로, 섞은 비용은 영어만·코드만의 비용을 code_share로 평균한 것 (반올림 오차 안에서)
    let l = Layout::parse("kfdbq.uolp\nrns_x.ehtc\nymviz.ajgw\n").unwrap();
    let cost = |x: &super::Scoring| { let mut m = l; x.score(&mut m); crate::score::cost_per_100(m.total) };
    let (mix, e, c) = (cost(&s), cost(&s.parts[0].1), cost(&s.parts[1].1));
    let want = (1.0 - config.code_share) * e + config.code_share * c;
    assert!((mix - want).abs() < 0.01 * want, "{mix} vs {want} ({e}, {c})");
    assert!((e - c).abs() > 1.0);
  }
}
