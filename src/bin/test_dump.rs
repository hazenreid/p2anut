// 위키 덤프 파싱 결과를 확인하는 도구.
// 사용법: test_dump [result_path] [articles]
//   - config.toml 기준 덤프 파일(wikipedia_download 또는 wikipedia_dump_file)이 result_path 아래 있으면 그 파일을,
//     아니면 wikipedia_dump_url을 스트리밍으로 읽음. 문서 선별 규칙과 wikipedia_sample은 분석기와 같음
//   - 첫 본문 기사의 원문과 정리 결과를 출력
//   - articles(기본 2000)개 기사를 정리해서, 분석기와 같은 규칙으로 끊은 단어의 상위 목록과
//     마크업 찌꺼기로 의심되는 토큰의 빈도를 출력 (0이면 첫 기사만 출력)
#[path = "../dump.rs"]
#[allow(dead_code)] // 분석기 전용 항목(PARSER_VERSION 등)은 여기서 안 씀
mod dump;
#[path = "../config.rs"]
#[allow(dead_code)]
mod config;

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;
use reqwest::blocking::Client;
use std::time::Duration;
use bzip2::read::MultiBzDecoder;

const TOP_WORDS: usize = 60;

// 정상 문장에는 거의 안 나오고 마크업이 새면 많이 나오는 토큰
const SUSPICIOUS: [&str; 40] = [
  "nbsp", "ndash", "mdash", "minus", "times", "frac", "sqrt", "mathrm", "mathbf", "displaystyle", "le", "ge",
  "jpg", "jpeg", "png", "svg", "gif", "ogg", "thumb", "px", "upright",
  "http", "https", "www", "html", "htm", "php", "isbn", "doi", "issn", "oclc", "pmid",
  "wikitable", "colspan", "rowspan", "bgcolor", "notoc", "defaultsort", "reflist", "infobox",
];

fn open_source(cfg: &config::Config, result_path: Option<&str>) -> anyhow::Result<Box<dyn Read>> {
  if let Some(path) = result_path.and_then(|dir| cfg.dump_path(Path::new(dir))) {
    if path.exists() {
      println!("Reading {}", path.display());
      return Ok(Box::new(std::fs::File::open(path)?));
    }
    println!("{} not found, streaming instead", path.display());
  }
  let client = Client::builder()
    .connect_timeout(Duration::from_secs(30))
    .timeout(None)
    .user_agent("KeyboardLayoutOptimizer/1.0 (corpus builder)")
    .build()?;
  println!("Connecting to {}...", cfg.wikipedia_dump_url);
  Ok(Box::new(client.get(&cfg.wikipedia_dump_url).send()?.error_for_status()?))
}

// 분석기(analyze_bytes)와 같은 경계: a-z 연속만 한 단어, 대문자는 소문자로
fn count_words(text: &[u8], words: &mut HashMap<Vec<u8>, u64>) {
  for w in text.split(|b| !b.is_ascii_alphabetic()).filter(|w| !w.is_empty()) {
    *words.entry(w.to_ascii_lowercase()).or_insert(0) += 1;
  }
}

fn print_preview(title: &str, raw: &[u8], text: &[u8]) {
  println!("=== ARTICLE: {title} ===\n");
  let raw_preview = String::from_utf8_lossy(&raw[..raw.len().min(1000)]); // 글자 중간에서 잘려도 표시되도록 lossy
  println!("--- RAW (first 1000 chars) ---\n{raw_preview}\n");
  let stripped_preview = String::from_utf8_lossy(&text[..text.len().min(2000)]);
  println!("--- STRIPPED (first 2000 chars) ---\n{stripped_preview}\n");
}

#[allow(clippy::cast_precision_loss)]
fn print_report(counts: &dump::ScanCounts, tail_cut: usize, words: &HashMap<Vec<u8>, u64>) {
  let total: u64 = words.values().sum();
  let per_million = |n: u64| if total == 0 { 0.0 } else { n as f64 * 1e6 / total as f64 };
  let articles = counts.articles as usize;
  let pct = |n: usize| if articles == 0 { 0.0 } else { n as f64 * 100.0 / articles as f64 };

  println!("=== REPORT: {articles} articles, {total} words, {} distinct ===", words.len());
  println!("skipped {} redirects, {} list/disambiguation pages, {} outside sample",
    counts.redirects, counts.lists, counts.sampled_out);
  println!("tail sections cut in {tail_cut} articles ({:.1}%)\n", pct(tail_cut));

  let mut top: Vec<(&Vec<u8>, &u64)> = words.iter().collect();
  top.sort_unstable_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
  println!("--- top {TOP_WORDS} words (per million words) ---");
  for (rank, &(w, &n)) in top.iter().take(TOP_WORDS).enumerate() {
    println!("{:>3}. {:<14} {:>10.0}", rank + 1, String::from_utf8_lossy(w), per_million(n));
  }

  // 일부(minus, times, thumb, le 등)는 정상 단어이기도 하므로 절대값보다 수정 전후 비교에 쓰기
  println!("\n--- suspicious tokens (per million words; normal words like 'times' also appear here) ---");
  let mut sus: Vec<(&str, u64)> = SUSPICIOUS.iter()
    .map(|&s| (s, words.get(s.as_bytes()).copied().unwrap_or(0)))
    .collect();
  sus.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
  for (s, n) in sus {
    println!("{s:<14} {:>10.1}  ({n})", per_million(n));
  }
}

fn main() -> anyhow::Result<()> {
  let cfg = config::load_config(Path::new("config.toml"))?;
  let args: Vec<String> = std::env::args().collect();
  let result_path = args.get(1).map(String::as_str);
  let limit: usize = match args.get(2) {
    Some(s) => s.parse().map_err(|_| anyhow::anyhow!("articles must be a number: {s}"))?,
    None => 2000,
  };
  let source = open_source(&cfg, result_path)?;
  let decoder = MultiBzDecoder::new(BufReader::new(source));

  let mut cleaner = dump::ArticleCleaner::new();
  let mut tail_cut = 0usize;
  let mut words: HashMap<Vec<u8>, u64> = HashMap::new();
  let mut n = 0usize;

  // 분석기와 같은 문서 선별(본문만, 넘겨주기·목록·동음이의 제외, wikipedia_sample 표본)
  let counts = dump::scan_articles(BufReader::new(decoder), cfg.wikipedia_sample, |title, raw| {
    let text = cleaner.clean(raw);
    if n == 0 { print_preview(&String::from_utf8_lossy(title), raw, text); }
    count_words(text, &mut words);
    if cleaner.tail_cut { tail_cut += 1; }
    n += 1;
    Ok(n < limit.max(1))
  })?;

  if limit > 0 { print_report(&counts, tail_cut, &words); }
  Ok(())
}
