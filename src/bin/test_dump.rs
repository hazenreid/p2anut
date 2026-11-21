// 첫 본문 기사 하나의 원문과 마크업 제거 결과를 출력하는 확인용 도구.
// 사용법: test_dump [result_path] — config.toml의 wikipedia_dump_file이 result_path 아래 있으면 그 파일을,
// 아니면 wikipedia_dump_url을 스트리밍으로 읽음
#[path = "../dump.rs"]
mod dump;
#[path = "../config.rs"]
#[allow(dead_code)]
mod config;

use std::io::{BufReader, Read};
use std::path::Path;
use reqwest::blocking::Client;
use std::time::Duration;
use bzip2::read::MultiBzDecoder;
use quick_xml::events::Event;
use quick_xml::reader::Reader;

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

fn main() -> anyhow::Result<()> {
  let cfg = config::load_config(Path::new("config.toml"))?;
  let result_path = std::env::args().nth(1);
  let source = open_source(&cfg, result_path.as_deref())?;

  let decoder = MultiBzDecoder::new(BufReader::new(source));
  let mut reader = Reader::from_reader(BufReader::new(decoder));

  let mut buf = Vec::new();
  let mut cleaner = dump::ArticleCleaner::new();
  let mut in_text = false;
  let mut in_ns = false;
  let mut in_title = false;
  let mut is_main_ns = false;
  let mut current_title = String::new();

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
          let raw = e.into_inner();
          if dump::is_redirect(&raw) {
            in_text = false;
            buf.clear(); // continue로 루프 끝의 clear를 건너뛰면 buf가 계속 커짐
            continue;
          }

          println!("=== ARTICLE: {} ===\n", current_title);

          let raw_preview = String::from_utf8_lossy(&raw[..raw.len().min(1000)]); // 글자 중간에서 잘려도 표시되도록 lossy
          println!("--- RAW (first 1000 chars) ---\n{raw_preview}\n");

          let text = cleaner.clean(&raw);
          let stripped_preview = String::from_utf8_lossy(&text[..text.len().min(2000)]);
          println!("--- STRIPPED (first 2000 chars) ---\n{stripped_preview}\n");

          break;
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

  Ok(())
}
