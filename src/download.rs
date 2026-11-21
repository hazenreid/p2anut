// 덤프 다운로드: 여러 주소(공식·미러)의 속도를 재서 빠른 순으로 고르고, 덩어리 단위로 이어받기.
// 한 주소가 계속 실패하면 다음으로 빠른 주소로 넘어가 받은 데서부터 이어받음
#![allow(clippy::cast_precision_loss)]
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use anyhow::Context;
use reqwest::blocking::Client;
use crate::SHUTDOWN;

pub fn http_client() -> anyhow::Result<Client> {
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

// 속도 측정: 앞부분 PROBE_BYTES를 받아 보거나 PROBE_TIMEOUT까지 받은 양으로 잼 (연결·첫 응답 시간 포함)
const PROBE_BYTES: u64 = 4 << 20;
const PROBE_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone)]
pub struct Probe {
  pub url: String,
  pub total: Option<u64>,     // 파일 전체 크기 (Content-Range 또는 Content-Length)
  pub bytes_per_sec: f64,     // 첫 데이터 도착 뒤의 전송 속도
  pub latency: Duration,      // 요청부터 첫 데이터까지
}

pub fn probe(client: &Client, url: &str, bytes: u64, timeout: Duration) -> anyhow::Result<Probe> {
  let start = Instant::now();
  let mut resp = client.get(url)
    .header(reqwest::header::RANGE, format!("bytes=0-{}", bytes - 1))
    .timeout(timeout)
    .send()?;
  let status = resp.status();
  let total = if status == reqwest::StatusCode::PARTIAL_CONTENT {
    content_range_total(&resp)
  } else if status.is_success() {
    resp.content_length()
  } else {
    anyhow::bail!("HTTP {status}");
  };
  // 속도는 첫 데이터가 도착한 뒤부터 잼: 연결·첫 응답 지연(멀리 있는 미러일수록 큼)은 긴 다운로드에서는
  // 한 번뿐이라, 포함하면 멀지만 빠른 미러가 손해를 봄. 지연은 따로 기록
  let mut buf = vec![0u8; 64 << 10];
  let mut got = 0u64;
  let mut first: Option<(Instant, u64)> = None;
  while got < bytes {
    // 제한 시간이 지나 읽기가 끊기면 거기까지 받은 양으로 계산
    match std::io::Read::read(&mut resp, &mut buf) {
      Ok(0) | Err(_) => break,
      Ok(n) => {
        got += n as u64;
        if first.is_none() { first = Some((Instant::now(), got)); }
      }
    }
  }
  let end = Instant::now();
  let Some((t_first, first_bytes)) = first else { anyhow::bail!("no data within {}s", timeout.as_secs()); };
  let latency = t_first - start;
  // 첫 읽기 뒤에 받은 양 / 걸린 시간. 걸린 시간이 아주 짧으면 그만큼 빠른 것이므로 하한(0.1ms)만 둠.
  // 첫 읽기에 다 받아 버린 경우(아주 작은 파일)는 받은 전체 양으로
  let after = got - first_bytes;
  let measured = if after > 0 { after } else { got };
  let bytes_per_sec = measured as f64 / (end - t_first).as_secs_f64().max(1e-4);
  Ok(Probe { url: url.to_string(), total, bytes_per_sec, latency })
}

// 주소들의 속도를 동시에 재서 빠른 순으로 정렬. 실패한 주소, 다른 주소들과 파일 크기가 다른 주소(다른 날짜판,
// 아직 덜 받은 미러 등)는 뺌. 크기는 가장 많은 주소가 알려 준 값(동률이면 빠른 쪽)을 기준으로 함
pub fn rank_mirrors(client: &Client, urls: &[String], bytes: u64, timeout: Duration) -> anyhow::Result<Vec<Probe>> {
  let results: Vec<(String, anyhow::Result<Probe>)> = std::thread::scope(|s| {
    let handles: Vec<_> = urls.iter().map(|u| (u.clone(), s.spawn(|| probe(client, u, bytes, timeout)))).collect();
    handles.into_iter()
      .map(|(u, h)| (u, h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("probe thread panicked")))))
      .collect()
  });

  let mut ok: Vec<Probe> = Vec::new();
  for (url, r) in results {
    match r {
      Ok(p) => ok.push(p),
      Err(e) => println!("[Mirror] skipped {url}: {e}"),
    }
  }
  ok.sort_by(|a, b| b.bytes_per_sec.total_cmp(&a.bytes_per_sec));

  // 기준 크기: 가장 많이 나온 값, 동률이면 더 빠른 주소가 알려 준 값
  let mut sizes: Vec<(Option<u64>, usize, usize)> = Vec::new(); // (크기, 개수, 처음 나온 순위)
  for (rank, p) in ok.iter().enumerate() {
    match sizes.iter_mut().find(|s| s.0 == p.total) {
      Some(s) => s.1 += 1,
      None => sizes.push((p.total, 1, rank)),
    }
  }
  let expected = sizes.iter().max_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2))).map(|s| s.0);
  ok.retain(|p| {
    let same = Some(p.total) == expected;
    if !same { println!("[Mirror] skipped {}: size {:?} differs from {:?}", p.url, p.total, expected.flatten()); }
    same
  });
  anyhow::ensure!(!ok.is_empty(), "no dump URL responded");
  for p in &ok {
    println!("[Mirror] {:>8.2} MiB/s  {:>5} ms  {}", p.bytes_per_sec / f64::from(1 << 20), p.latency.as_millis(), p.url);
  }
  Ok(ok)
}

// 다운로드 동작 값 (테스트에서는 작게)
#[derive(Clone, Copy)]
pub struct DownloadOpts {
  // 한 번의 Range 요청으로 받는 크기와 그 요청 하나의 제한 시간.
  // reqwest 0.11 blocking에는 읽기 타임아웃이 없어서, 연결이 데이터 없이 멈추면 read가 영원히 막힘
  // (실제로 3GB 지점에서 ESTABLISHED 상태로 2시간 넘게 멈춤). 덩어리마다 요청 전체 제한 시간을 걸어
  // 멈춘 연결은 끊고 받은 데까지 이어받음. 8MiB를 2분 안에 못 받으면(약 70KB/s 미만) 멈춘 것으로 봄.
  // Wikimedia에서 실측한 속도(약 0.18MiB/s)로 한 덩어리에 약 45초 → 여유 약 2.7배.
  // 이보다 느린 회선이면 매 덩어리가 시간 초과되지만, 받은 만큼은 저장되므로 계속 진행됨
  pub chunk: u64,
  pub chunk_timeout: Duration,
  // 한 주소에서 진행 없이 연속으로 이만큼 실패하면 다음 주소로 넘어감
  pub max_failures: u32,
  // 재시도 대기: 실패 횟수 × retry_wait (최대 max_wait)
  pub retry_wait: Duration,
  pub max_wait: Duration,
  // 속도 측정
  pub probe_bytes: u64,
  pub probe_timeout: Duration,
  // 다 받은 뒤 Wikimedia가 공개하는 SHA-1과 비교 (여러 미러에서 섞어 받을 수 있으므로)
  pub verify: bool,
}

impl Default for DownloadOpts {
  fn default() -> Self {
    Self {
      chunk: 8 << 20,
      chunk_timeout: Duration::from_secs(120),
      max_failures: 10,
      retry_wait: Duration::from_secs(5),
      max_wait: Duration::from_secs(60),
      probe_bytes: PROBE_BYTES,
      probe_timeout: PROBE_TIMEOUT,
      verify: true,
    }
  }
}

// URL의 마지막 경로 부분 (쿼리·조각 제외)
fn url_file_name(url: &str) -> Option<&str> {
  url.split(['?', '#']).next()?.rsplit('/').next().filter(|s| !s.is_empty())
}

// 같은 디렉터리의 다른 파일 주소
fn sibling_url(url: &str, name: &str) -> Option<String> {
  let (dir, _) = url.split(['?', '#']).next()?.rsplit_once('/')?;
  Some(format!("{dir}/{name}"))
}

// "enwiki-20261101-pages-articles-multistream.xml.bz2" → "enwiki-20261101-sha1sums.txt"
fn sha1sums_name(file: &str) -> Option<String> {
  let mut parts = file.splitn(3, '-');
  let (wiki, date, _rest) = (parts.next()?, parts.next()?, parts.next()?);
  (!wiki.is_empty() && date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit()))
    .then(|| format!("{wiki}-{date}-sha1sums.txt"))
}

// "<sha1>  <파일 이름>" 줄들에서 file의 SHA-1
fn find_sha1(sums: &str, file: &str) -> Option<String> {
  sums.lines().find_map(|l| {
    let mut it = l.split_whitespace();
    let (hash, name) = (it.next()?, it.next()?);
    (name == file && hash.len() == 40 && hash.bytes().all(|b| b.is_ascii_hexdigit())).then(|| hash.to_ascii_lowercase())
  })
}

// 체크섬 파일을 받아 file의 SHA-1을 찾음. 공식 주소를 먼저 묻고, 안 되면 나머지 주소 순서대로
fn expected_sha1(client: &Client, urls: &[String], file: &str) -> Option<String> {
  let name = sha1sums_name(file)?;
  let official = |u: &&String| u.contains("://dumps.wikimedia.org/");
  let order = urls.iter().filter(official).chain(urls.iter().filter(|u| !official(u)));
  for url in order {
    let Some(sums_url) = sibling_url(url, &name) else { continue; };
    let text = client.get(&sums_url).timeout(Duration::from_secs(30)).send()
      .and_then(reqwest::blocking::Response::error_for_status)
      .and_then(reqwest::blocking::Response::text);
    match text {
      Ok(t) => match find_sha1(&t, file) {
        Some(h) => { println!("[Verify] expected sha1 {h} (from {sums_url})"); return Some(h); }
        None => println!("[Verify] {file} not listed in {sums_url}"),
      },
      Err(e) => println!("[Verify] could not get {sums_url}: {e}"),
    }
  }
  None
}

// 파일의 SHA-1 (16진 소문자). 1GiB마다 진행 표시
pub fn sha1_file(path: &Path) -> anyhow::Result<String> {
  use sha1::{Digest, Sha1};
  let mut f = fs::File::open(path)?;
  let len = f.metadata()?.len();
  let mut hasher = Sha1::new();
  let mut buf = vec![0u8; 4 << 20];
  let mut done = 0u64;
  loop {
    if SHUTDOWN.load(Ordering::SeqCst) { anyhow::bail!("verification interrupted (re-run to verify again)"); }
    let n = std::io::Read::read(&mut f, &mut buf)?;
    if n == 0 { break; }
    hasher.update(&buf[..n]);
    let before = done;
    done += n as u64;
    if done >> 30 != before >> 30 {
      print!("\r[Verify] {} / {} MiB          ", done >> 20, len >> 20);
      let _ = std::io::stdout().flush();
    }
  }
  Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

// 다 받은 part 파일을 공개된 SHA-1과 비교. 체크섬을 못 구하면 경고만 하고 통과.
// 다르면 "<path>.bad"로 옮기고 에러 (깨진 파일이 분석에 쓰이지 않도록, 최종 이름으로 바꾸기 전에 검사)
fn verify_download(client: &Client, urls: &[String], part: &Path, path: &Path) -> anyhow::Result<()> {
  let Some(file) = urls.first().and_then(|u| url_file_name(u)) else { return Ok(()); };
  let Some(expected) = expected_sha1(client, urls, file) else {
    println!("[Verify] no published sha1 found for {file}; skipping verification");
    return Ok(());
  };
  println!("[Verify] hashing {} ...", part.display());
  let actual = sha1_file(part)?;
  if actual != expected {
    let bad = std::path::PathBuf::from(format!("{}.bad", path.display()));
    fs::rename(part, &bad)?;
    anyhow::bail!("sha1 mismatch for {file}: expected {expected}, got {actual}. The corrupt file was moved to {}; delete it and re-run to download again", bad.display());
  }
  println!("\r[Verify] sha1 OK          ");
  Ok(())
}

// 덩어리 하나 받기. Ok(true) = 파일 끝까지 받음
fn fetch_chunk(client: &Client, url: &str, part: &Path, offset: u64, total: &mut Option<u64>, last_report: &mut u64, opts: &DownloadOpts) -> anyhow::Result<bool> {
  let range_end = offset + opts.chunk - 1;
  let mut resp = client.get(url)
    .header(reqwest::header::RANGE, format!("bytes={offset}-{range_end}"))
    .timeout(opts.chunk_timeout)
    .send()?;
  let status = resp.status();

  if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
    let t = content_range_total(&resp);
    if t == Some(offset) { return Ok(true); } // 이미 다 받음
    anyhow::bail!("server rejected range at {offset} (remote size {t:?})");
  }
  let (mut file, expected) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
    let t = content_range_total(&resp).context("206 response without Content-Range total")?;
    if let Some(prev) = *total { anyhow::ensure!(t == prev, "remote size changed ({prev} -> {t})"); }
    *total = Some(t);
    (fs::OpenOptions::new().create(true).append(true).open(part)?, range_end.min(t - 1) + 1 - offset)
  } else if status.is_success() {
    // Range 미지원 서버: 덩어리 없이 처음부터 전부 받음 (요청 제한 시간도 걸 수 없음)
    anyhow::ensure!(offset == 0, "server ignored Range at offset {offset}");
    println!("\n[Download] server ignored Range, downloading whole file without stall detection");
    *total = resp.content_length();
    let mut file = fs::File::create(part)?;
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
}

// 덤프를 path로 내려받음. 주소가 여럿이면 속도를 재서 빠른 순으로 쓰고, 한 주소가 계속 실패하면 다음 주소로.
// "<path>.part"에 덩어리 단위 Range 요청으로 이어 쓰고, 끝나면 rename
pub fn download_dump(client: &Client, urls: &[String], path: &Path, opts: &DownloadOpts) -> anyhow::Result<()> {
  anyhow::ensure!(!urls.is_empty(), "no dump URL");
  let (order, mut total): (Vec<String>, Option<u64>) = if urls.len() == 1 {
    (urls.to_vec(), None)
  } else {
    println!("[Mirror] measuring {} URLs...", urls.len());
    let ranked = rank_mirrors(client, urls, opts.probe_bytes, opts.probe_timeout)?;
    let total = ranked[0].total;
    (ranked.into_iter().map(|p| p.url).collect(), total)
  };

  let part = std::path::PathBuf::from(format!("{}.part", path.display()));
  let mut last_report = 0u64;
  let mut last_err = None;

  for (i, url) in order.iter().enumerate() {
    if i > 0 { println!("\n[Download] switching to {url}"); } else { println!("[Download] from {url}"); }
    let mut failures = 0u32;
    let finished = loop {
      let offset = fs::metadata(&part).map_or(0, |m| m.len());
      if let Some(t) = total {
        if offset == t { break true; }
        if offset > t { fs::remove_file(&part)?; anyhow::bail!("partial file larger than remote ({offset} > {t}); removed, re-run to restart"); }
      }
      match fetch_chunk(client, url, &part, offset, &mut total, &mut last_report, opts) {
        Ok(true) => break true,
        Ok(false) => failures = 0,
        Err(e) if SHUTDOWN.load(Ordering::SeqCst) => return Err(e),
        Err(e) => {
          // 진행이 있었으면 연속 실패 횟수 초기화
          if fs::metadata(&part).map_or(0, |m| m.len()) > offset { failures = 0; }
          failures += 1;
          if failures > opts.max_failures {
            eprintln!("\n[Download] {url} failed {} times in a row: {e:#}", opts.max_failures);
            last_err = Some(e);
            break false;
          }
          let wait = (opts.retry_wait * failures).min(opts.max_wait);
          eprintln!("\n[Download] {e:#} — retry {failures}/{} in {:.1}s", opts.max_failures, wait.as_secs_f32());
          std::thread::sleep(wait);
        }
      }
    };
    if finished {
      if opts.verify { verify_download(client, &order, &part, path)?; }
      fs::rename(&part, path)?;
      println!("\n[Download] done: {}", path.display());
      return Ok(());
    }
  }
  let err = last_err.unwrap_or_else(|| anyhow::anyhow!("download failed"));
  Err(err.context(format!("all {} dump URLs failed; received data is kept in {} for the next run (delete it to restart)", order.len(), part.display())))
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::io::{BufRead, BufReader};
  use std::net::TcpListener;
  use std::sync::Arc;
  use std::sync::atomic::AtomicUsize;

  // 테스트용 HTTP 서버 동작
  #[derive(Clone, Copy)]
  enum Mode {
    Ok { delay_per_64k_ms: u64 },   // Range 지원, 느리게 보낼 수 있음
    NotFound,
    BreakAfter(usize),             // 요청 이만큼 받은 뒤로는 연결을 바로 끊음
    SlowStart { first_ms: u64 },   // 응답 전에 한 번만 기다리고 그 뒤로는 빠르게 (멀리 있는 미러)
  }

  const DUMP: &str = "/enwiki-20260101-pages-articles.xml.bz2";

  // body를 DUMP 경로에서 Range 요청대로 보내는 서버. (덤프 주소, 덤프 요청 수)
  fn serve(body: Arc<Vec<u8>>, mode: Mode) -> (String, Arc<AtomicUsize>) {
    serve_files(vec![(DUMP.to_string(), body)], mode)
  }

  // 경로별 파일을 보내는 서버. 첫 파일이 덤프이고 mode·요청 수는 덤프에만 적용, 없는 경로는 404
  fn serve_files(files: Vec<(String, Arc<Vec<u8>>)>, mode: Mode) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let hits2 = hits.clone();
    let files = Arc::new(files);
    let url = format!("http://{addr}{}", files[0].0);
    std::thread::spawn(move || {
      for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue; };
        let files = files.clone();
        let hits = hits2.clone();
        std::thread::spawn(move || {
          let mut reader = BufReader::new(stream.try_clone().unwrap());
          let mut range: Option<(u64, u64)> = None;
          let mut path = String::new();
          loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 { return; }
            let l = line.trim_end();
            if l.is_empty() { break; }
            if path.is_empty() { path = l.split(' ').nth(1).unwrap_or("").to_string(); }
            if let Some(v) = l.to_ascii_lowercase().strip_prefix("range: bytes=") {
              let (a, b) = v.split_once('-').unwrap();
              range = Some((a.parse().unwrap(), b.parse().unwrap()));
            }
          }
          let not_found = |s: &mut std::net::TcpStream| {
            let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
          };
          let Some((_, body)) = files.iter().find(|(p, _)| *p == path) else { return not_found(&mut stream); };
          let mut delay = 0;
          if path == files[0].0 {
            let n = hits.fetch_add(1, Ordering::SeqCst);
            delay = match mode {
              Mode::NotFound => return not_found(&mut stream),
              Mode::BreakAfter(k) if n >= k => return,
              Mode::BreakAfter(_) => 0,
              Mode::Ok { delay_per_64k_ms } => delay_per_64k_ms,
              Mode::SlowStart { first_ms } => { std::thread::sleep(Duration::from_millis(first_ms)); 0 }
            };
          }
          let len = body.len() as u64;
          let (a, b) = range.unwrap_or((0, len.saturating_sub(1)));
          if a >= len {
            let _ = write!(stream, "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{len}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return;
          }
          let b = b.min(len - 1);
          let part = &body[a as usize..=b as usize];
          let _ = write!(stream, "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {a}-{b}/{len}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", part.len());
          for piece in part.chunks(64 << 10) {
            if delay > 0 { std::thread::sleep(Duration::from_millis(delay)); }
            if stream.write_all(piece).is_err() { return; }
          }
        });
      }
    });
    (url, hits)
  }

  fn sha1_hex(data: &[u8]) -> String {
    use sha1::{Digest, Sha1};
    Sha1::digest(data).iter().map(|b| format!("{b:02x}")).collect()
  }

  // 덤프와 체크섬 파일을 함께 보내는 서버
  fn serve_with_sums(body: Arc<Vec<u8>>, sums: &str) -> String {
    let sums_path = "/enwiki-20260101-sha1sums.txt".to_string();
    serve_files(vec![(DUMP.to_string(), body), (sums_path, Arc::new(sums.as_bytes().to_vec()))], Mode::Ok { delay_per_64k_ms: 0 }).0
  }

  fn client() -> Client {
    Client::builder().no_proxy().build().unwrap()
  }

  fn body(n: usize) -> Arc<Vec<u8>> {
    Arc::new((0..n).map(|i| (i * 31 % 251) as u8).collect())
  }

  fn opts() -> DownloadOpts {
    DownloadOpts {
      chunk: 100_000,
      chunk_timeout: Duration::from_secs(10),
      max_failures: 2,
      retry_wait: Duration::from_millis(10),
      max_wait: Duration::from_millis(20),
      probe_bytes: 256 << 10,
      probe_timeout: Duration::from_secs(5),
      verify: true,
    }
  }

  #[test]
  fn ranks_by_speed_and_drops_bad_mirrors() {
    let data = body(600_000);
    let (fast, _) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    let (slow, _) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 40 });
    let (other_size, _) = serve(body(500_000), Mode::Ok { delay_per_64k_ms: 0 });
    let (missing, _) = serve(data.clone(), Mode::NotFound);
    let urls = vec![slow.clone(), missing, other_size, fast.clone()];
    let ranked = rank_mirrors(&client(), &urls, 256 << 10, Duration::from_secs(5)).unwrap();
    let order: Vec<&str> = ranked.iter().map(|p| p.url.as_str()).collect();
    assert_eq!(order, vec![fast.as_str(), slow.as_str()]);
    assert!(ranked.iter().all(|p| p.total == Some(600_000)));
  }

  #[test]
  fn downloads_from_fastest_and_fails_over() {
    let dir = std::env::temp_dir().join(format!("p2anut-dl-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let data = body(650_000);

    // 가장 빠른 주소가 덩어리 두 개 뒤(속도 측정 요청 1번 + 덩어리 2번)에 고장 → 느린 주소로 넘어가 이어받음
    let (breaking, hits_breaking) = serve(data.clone(), Mode::BreakAfter(3));
    let (slow, hits_slow) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 20 });
    let path = dir.join("a.bz2");
    download_dump(&client(), &[slow.clone(), breaking.clone()], &path, &opts()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);
    assert!(hits_breaking.load(Ordering::SeqCst) >= 3);
    // 느린 주소는 속도 측정 1번 + 남은 덩어리만 받음 (처음부터 다시 받지 않음)
    let slow_hits = hits_slow.load(Ordering::SeqCst);
    assert!(slow_hits <= 1 + 5, "slow mirror re-downloaded too much: {slow_hits} requests");
    assert!(!dir.join("a.bz2.part").exists());

    // 주소 하나: 속도 측정 없이 바로
    let (only, hits_only) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    let path = dir.join("b.bz2");
    download_dump(&client(), &[only], &path, &opts()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);
    assert_eq!(hits_only.load(Ordering::SeqCst), 7); // 100KB 덩어리 7개

    // 전부 실패하면 에러 (받던 파일은 남겨서 다음에 이어받기)
    let (dead, _) = serve(data.clone(), Mode::BreakAfter(0));
    let path = dir.join("c.bz2");
    let err = download_dump(&client(), &[dead], &path, &opts()).unwrap_err();
    assert!(format!("{err:#}").contains("all 1 dump URLs failed"), "{err:#}");
    assert!(!path.exists());
    fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn verifies_sha1_after_download() {
    let dir = std::env::temp_dir().join(format!("p2anut-sha-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let data = body(350_000);
    let file = &DUMP[1..];
    let good = format!("0000000000000000000000000000000000000000  enwiki-20260101-other.sql.gz\n{}  {file}\n", sha1_hex(&data));

    // 맞는 체크섬: 통과
    let url = serve_with_sums(data.clone(), &good);
    let path = dir.join("ok.bz2");
    download_dump(&client(), &[url], &path, &opts()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);

    // 틀린 체크섬: 에러, 최종 이름으로 두지 않고 .bad로 옮김
    let url = serve_with_sums(data.clone(), &format!("{}  {file}\n", "ab".repeat(20)));
    let path = dir.join("bad.bz2");
    let err = download_dump(&client(), &[url], &path, &opts()).unwrap_err();
    assert!(format!("{err:#}").contains("sha1 mismatch"), "{err:#}");
    assert!(!path.exists() && dir.join("bad.bz2.bad").exists());

    // 체크섬 파일이 없음: 경고만 하고 통과
    let (url, _) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    let path = dir.join("nosums.bz2");
    download_dump(&client(), &[url], &path, &opts()).unwrap();
    assert!(path.exists());

    // 체크섬은 공식 주소에서 먼저 구함: 미러의 체크섬이 틀려도 공식 것이 맞으면 통과
    // (공식 주소는 테스트에서 띄울 수 없으므로, 첫 주소의 체크섬을 못 구하면 다음 주소에서 구하는지로 확인)
    let (no_sums, _) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    let with_sums = serve_with_sums(data.clone(), &good);
    let path = dir.join("second.bz2");
    download_dump(&client(), &[no_sums, with_sums], &path, &opts()).unwrap();
    assert!(path.exists());
    fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn checksum_file_names_and_parsing() {
    assert_eq!(sha1sums_name("enwiki-20261101-pages-articles-multistream.xml.bz2").as_deref(), Some("enwiki-20261101-sha1sums.txt"));
    assert_eq!(sha1sums_name("dump.xml.bz2"), None);
    assert_eq!(sha1sums_name("enwiki-latest-pages-articles.xml.bz2"), None);
    assert_eq!(sibling_url("https://m.org/a/b/x.bz2?q=1", "s.txt").as_deref(), Some("https://m.org/a/b/s.txt"));
    let sums = "8083BEC4F852BFF427A88C1A652E5B2FA284485E  enwiki-1-pages-articles.xml.bz2\nbfdb7ead45e262d7b9d55d86c8958789e7efd1ae  enwiki-1-pages-articles-multistream.xml.bz2\n";
    assert_eq!(find_sha1(sums, "enwiki-1-pages-articles.xml.bz2").as_deref(), Some("8083bec4f852bff427a88c1a652e5b2fa284485e"));
    assert_eq!(find_sha1(sums, "enwiki-1-pages-articles-multistream.xml.bz2").as_deref(), Some("bfdb7ead45e262d7b9d55d86c8958789e7efd1ae"));
    assert_eq!(find_sha1(sums, "missing.bz2"), None);
  }

  #[test]
  fn probe_excludes_first_byte_latency() {
    // 첫 응답까지 0.5초 걸리지만 그 뒤로는 빠른 서버: 지연을 속도에 섞으면 1MiB / 0.5초 = 2MiB/s 이하로 나옴
    let data = body(1 << 20);
    let (url, _) = serve(data, Mode::SlowStart { first_ms: 500 });
    let p = probe(&client(), &url, 1 << 20, Duration::from_secs(5)).unwrap();
    assert!(p.latency >= Duration::from_millis(500), "{:?}", p.latency);
    assert!(p.bytes_per_sec > 10.0 * f64::from(1 << 20), "{} B/s", p.bytes_per_sec);

    // 그래도 전송 자체가 느린 서버는 느리게 잼
    let (slow, _) = serve(body(1 << 20), Mode::Ok { delay_per_64k_ms: 30 });
    let q = probe(&client(), &slow, 1 << 20, Duration::from_secs(5)).unwrap();
    assert!(q.bytes_per_sec < 3.0 * f64::from(1 << 20), "{} B/s", q.bytes_per_sec);
  }

  #[test]
  fn probe_reports_size_and_speed() {
    let data = body(300_000);
    let (url, _) = serve(data, Mode::Ok { delay_per_64k_ms: 0 });
    let p = probe(&client(), &url, 128 << 10, Duration::from_secs(5)).unwrap();
    assert_eq!(p.total, Some(300_000));
    assert!(p.bytes_per_sec > 0.0);
  }
}
