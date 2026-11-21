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
  pub bytes_per_sec: f64,
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
  let mut buf = vec![0u8; 64 << 10];
  let mut got = 0u64;
  while got < bytes {
    // 제한 시간이 지나 읽기가 끊기면 거기까지 받은 양으로 계산
    match std::io::Read::read(&mut resp, &mut buf) {
      Ok(0) | Err(_) => break,
      Ok(n) => got += n as u64,
    }
  }
  anyhow::ensure!(got > 0, "no data within {}s", timeout.as_secs());
  Ok(Probe { url: url.to_string(), total, bytes_per_sec: got as f64 / start.elapsed().as_secs_f64().max(1e-3) })
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
    println!("[Mirror] {:>8.2} MiB/s  {}", p.bytes_per_sec / f64::from(1 << 20), p.url);
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
    }
  }
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
  }

  // body를 Range 요청대로 보내는 서버. (주소, 받은 요청 수)
  fn serve(body: Arc<Vec<u8>>, mode: Mode) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let hits2 = hits.clone();
    std::thread::spawn(move || {
      for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue; };
        let body = body.clone();
        let n = hits2.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
          let mut reader = BufReader::new(stream.try_clone().unwrap());
          let mut range: Option<(u64, u64)> = None;
          loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 { return; }
            let l = line.trim_end();
            if l.is_empty() { break; }
            if let Some(v) = l.to_ascii_lowercase().strip_prefix("range: bytes=") {
              let (a, b) = v.split_once('-').unwrap();
              range = Some((a.parse().unwrap(), b.parse().unwrap()));
            }
          }
          let delay = match mode {
            Mode::NotFound => {
              let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
              return;
            }
            Mode::BreakAfter(k) if n >= k => return,
            Mode::BreakAfter(_) => 0,
            Mode::Ok { delay_per_64k_ms } => delay_per_64k_ms,
          };
          let len = body.len() as u64;
          let (a, b) = range.unwrap_or((0, len - 1));
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
    (format!("http://{addr}/dump.xml.bz2"), hits)
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
  fn probe_reports_size_and_speed() {
    let data = body(300_000);
    let (url, _) = serve(data, Mode::Ok { delay_per_64k_ms: 0 });
    let p = probe(&client(), &url, 128 << 10, Duration::from_secs(5)).unwrap();
    assert_eq!(p.total, Some(300_000));
    assert!(p.bytes_per_sec > 0.0);
  }
}
