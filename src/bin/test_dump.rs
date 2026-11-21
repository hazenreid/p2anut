// 위키 덤프 파싱 결과를 확인하는 도구.
// 사용법: test_dump [result_path] [articles]
//   - config.toml의 wikipedia_dump_file이 result_path 아래 있으면 그 파일을, 아니면 wikipedia_dump_url을 스트리밍으로 읽음
//   - 첫 본문 기사의 원문과 정리 결과를 출력
//   - articles(기본 2000)개 기사를 정리해서, 분석기와 같은 규칙으로 끊은 단어의 상위 목록과
//     마크업 찌꺼기로 의심되는 토큰의 빈도를 출력 (0이면 첫 기사만 출력)
#[path = "../dump.rs"]
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
use quick_xml::events::Event;
use quick_xml::reader::Reader;

const TOP_WORDS: usize = 60;

// 정상 문장에는 거의 안 나오고 마크업이 새면 많이 나오는 토큰
const SUSPICIOUS: [&str; 40] = [
  "nbsp", "ndash", "mdash", "minus", "times", "frac", "sqrt", "mathrm", "mathbf", "displaystyle", "le", "ge",
  "jpg", "jpeg", "png", "svg", "gif", "ogg", "thumb", "px", "upright",
  "http", "https", "www", "html", "htm", "php", "isbn", "doi", "issn", "oclc", "pmid",
  "wikitable", "colspan", "rowspan", "bgcolor", "notoc", "defaultsort", "reflist", "infobox",
];

fn open_source(cfg: &config::Config, result_path: Option<&str>) -> anyhow::Result<Box<dyn Read>> {
  if let (Some(dir), Some(file)) = (result_path, &cfg.wikipedia_dump_file) {
    let path = Path::new(dir).join(file);
    if path.exists() {
      println!("Reading {}", path.display());
      return Ok(Box::new(std::fs::File::open(path)?));
    }
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
fn print_report(articles: usize, tail_cut: usize, words: &HashMap<Vec<u8>, u64>) {
  let total: u64 = words.values().sum();
  let per_million = |n: u64| if total == 0 { 0.0 } else { n as f64 * 1e6 / total as f64 };
  let pct = |n: usize| if articles == 0 { 0.0 } else { n as f64 * 100.0 / articles as f64 };

  println!("=== REPORT: {articles} articles, {total} words, {} distinct ===", words.len());
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
  let mut reader = Reader::from_reader(BufReader::new(decoder));

  let mut buf = Vec::new();
  let mut cleaner = dump::ArticleCleaner::new();
  let mut in_text = false;
  let mut in_ns = false;
  let mut in_title = false;
  let mut is_main_ns = false;
  let mut current_title = String::new();

  let mut articles = 0usize;
  let mut tail_cut = 0usize;
  let mut words: HashMap<Vec<u8>, u64> = HashMap::new();

  loop {
    match reader.read_event_into(&mut buf) {
      Ok(Event::Start(e)) => match e.name().as_ref() {
        b"page"  => { is_main_ns = false; current_title.clear(); }
        b"ns"    => { in_ns = true; }
        b"title" => { in_title = true; }
        b"text" if is_main_ns => { in_text = true; }
        _ => {}
      },
      Ok(Event::Text(e)) => {
        if in_ns {
          is_main_ns = e.as_ref() as &[u8] == b"0";
          in_ns = false;
        } else if in_title {
          current_title = String::from_utf8_lossy(e.as_ref()).into_owned();
          in_title = false;
        } else if in_text {
          in_text = false;
          let raw = e.into_inner();
          if !dump::is_redirect(&raw) {
            let text = cleaner.clean(&raw);
            if articles == 0 { print_preview(&current_title, &raw, text); }
            count_words(text, &mut words);
            articles += 1;
            if cleaner.tail_cut { tail_cut += 1; }
            if articles >= limit.max(1) { break; }
          }
        }
      },
      Ok(Event::End(e)) => match e.name().as_ref() {
        b"text"  => { in_text = false; }
        b"title" => { in_title = false; }
        _ => {}
      },
      Ok(Event::Eof) => break,
      Err(e) => { eprintln!("XML error: {e}"); break; }
      _ => {}
    }
    buf.clear();
  }

  if limit > 0 { print_report(articles, tail_cut, &words); }
  Ok(())
}
