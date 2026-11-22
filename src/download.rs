// 덤프 다운로드: 여러 주소(공식·미러)의 속도를 하나씩 재고, 빠른 미러 여러 곳에서 덩어리를 나눠 동시에 받음.
// 받은 덩어리는 <파일>.part의 제자리에 쓰고 <파일>.part.state에 기록해서, 중단해도 남은 덩어리만 이어받음.
// 미러 하나가 계속 실패하면 그 연결은 예비 주소로 넘어가고, 예비도 없으면 그 연결만 멈춤 (다른 미러는 계속)
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use anyhow::Context;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
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

// 속도 측정: 미러마다 하나씩 차례로(다른 미러와 회선을 나눠 쓰지 않게), 첫 데이터가 온 뒤 PROBE_WINDOW 동안
// 또는 PROBE_BYTES까지 받아 봄. 첫 데이터까지는 PROBE_TIMEOUT까지 기다림
const PROBE_BYTES: u64 = 32 << 20;
const PROBE_WINDOW: Duration = Duration::from_secs(8);
const PROBE_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone)]
pub struct Probe {
  pub url: String,
  pub total: Option<u64>,     // 파일 전체 크기 (Content-Range 또는 Content-Length)
  pub bytes_per_sec: f64,     // 안정된 구간(첫 데이터 뒤 걸린 시간의 뒤쪽 절반)의 전송 속도
  pub latency: Duration,      // 요청부터 첫 데이터까지
}

// 받은 기록 (첫 데이터 뒤 경과 초, 그 뒤 받은 양)에서 뒤쪽 절반 시간 동안의 속도.
// 연결 초반에는 TCP가 속도를 천천히 올려서(slow start) 멀리 있는 서버일수록 앞부분이 느리게 나오므로 뺌
fn steady_speed(samples: &[(f64, u64)]) -> Option<f64> {
  let &(t_end, b_end) = samples.last()?;
  let half = t_end / 2.0;
  let (t_h, b_h) = samples.iter().rev().find(|s| s.0 <= half).copied().unwrap_or((0.0, 0));
  Some((b_end - b_h) as f64 / (t_end - t_h).max(1e-4))
}

pub fn probe(client: &Client, url: &str, bytes: u64, window: Duration, timeout: Duration) -> anyhow::Result<Probe> {
  let start = Instant::now();
  let mut resp = client.get(url)
    .header(reqwest::header::RANGE, format!("bytes=0-{}", bytes - 1))
    .timeout(timeout + window + Duration::from_secs(5))
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
  let mut first: Option<Instant> = None;
  let mut samples: Vec<(f64, u64)> = Vec::new();
  let mut after = 0u64;
  while got < bytes {
    // 제한 시간이 지나 읽기가 끊기면 거기까지 받은 양으로 계산
    match resp.read(&mut buf) {
      Ok(0) | Err(_) => break,
      Ok(n) => {
        got += n as u64;
        match first {
          None => first = Some(Instant::now()),
          Some(t0) => {
            after += n as u64;
            let t = t0.elapsed();
            samples.push((t.as_secs_f64(), after));
            if t >= window { break; }
          }
        }
      }
    }
  }
  let Some(t_first) = first else { anyhow::bail!("no data within {}s", timeout.as_secs()); };
  let latency = t_first - start;
  // 첫 읽기에 다 받아 버린 경우(아주 작은 파일)는 받은 전체 양으로. 걸린 시간이 아주 짧으면 그만큼 빠른 것이므로 하한(0.1ms)만 둠
  let bytes_per_sec = steady_speed(&samples).unwrap_or_else(|| got as f64 / t_first.elapsed().as_secs_f64().max(1e-4));
  Ok(Probe { url: url.to_string(), total, bytes_per_sec, latency })
}

// 주소들의 속도를 하나씩 재서 빠른 순으로 정렬. 실패한 주소, 다른 주소들과 파일 크기가 다른 주소(다른 날짜판,
// 아직 덜 받은 미러 등)는 뺌. 크기는 가장 많은 주소가 알려 준 값(동률이면 빠른 쪽)을 기준으로 함
pub fn rank_mirrors(client: &Client, urls: &[String], opts: &DownloadOpts) -> anyhow::Result<Vec<Probe>> {
  let mut results: Vec<(String, anyhow::Result<Probe>)> = Vec::new();
  for (i, u) in urls.iter().enumerate() {
    print!("\r[Mirror] measuring {}/{}: {}          ", i + 1, urls.len(), host(u));
    let _ = std::io::stdout().flush();
    results.push((u.clone(), probe(client, u, opts.probe_bytes, opts.probe_window, opts.probe_timeout)));
    if SHUTDOWN.load(Ordering::SeqCst) { anyhow::bail!("interrupted while measuring mirrors"); }
  }
  println!();

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

// 주소의 호스트 이름 (표시용)
fn host(url: &str) -> &str {
  url.split_once("://").map_or(url, |(_, r)| r).split(['/', '?', '#']).next().unwrap_or(url)
}

// 주소가 가리키는 (IP, 포트)들. 이름을 풀지 못하면 빈 목록
fn resolve(url: &str) -> Vec<SocketAddr> {
  let h = host(url);
  let default_port = if url.starts_with("https") { 443 } else { 80 };
  let (name, port) = match h.rsplit_once(':') {
    Some((n, p)) if !n.contains(':') && p.parse::<u16>().is_ok() => (n, p.parse().unwrap_or(default_port)),
    _ => (h, default_port),
  };
  let mut addrs: Vec<SocketAddr> = (name, port).to_socket_addrs().map(Iterator::collect).unwrap_or_default();
  addrs.sort_unstable();
  addrs.dedup();
  addrs
}

// 동시에 받을 미러와 예비를 나눔 (ranked는 빠른 순). 가장 빠른 곳의 min_share 이상만 바로 쓰고,
// 같은 (IP, 포트)를 쓰는 주소(같은 서버의 다른 이름)는 빠른 쪽 하나만 씀. 나머지는 빠른 순으로 예비:
// 바로 쓰는 미러가 계속 실패하면 그 연결이 예비 주소로 넘어감
fn split_mirrors(ranked: &[Probe], min_share: f64, addrs: impl Fn(&str) -> Vec<SocketAddr>) -> (Vec<String>, Vec<String>) {
  let fastest = ranked.first().map_or(0.0, |p| p.bytes_per_sec);
  let mut used: Vec<Vec<SocketAddr>> = Vec::new();
  let (mut primary, mut backup) = (Vec::new(), Vec::new());
  for p in ranked {
    let a = addrs(&p.url);
    let same_server = !a.is_empty() && used.iter().any(|u| u.iter().any(|x| a.contains(x)));
    if primary.is_empty() || (p.bytes_per_sec >= fastest * min_share && !same_server) {
      primary.push(p.url.clone());
      used.push(a);
    } else {
      backup.push(p.url.clone());
    }
  }
  (primary, backup)
}

// 다운로드 동작 값 (테스트에서는 작게)
#[derive(Clone, Copy)]
pub struct DownloadOpts {
  // 한 번의 Range 요청으로 받는 크기와 그 요청 하나의 제한 시간.
  // reqwest 0.11 blocking에는 읽기 타임아웃이 없어서, 연결이 데이터 없이 멈추면 read가 영원히 막힘
  // (실제로 3GB 지점에서 ESTABLISHED 상태로 2시간 넘게 멈춤). 덩어리마다 요청 전체 제한 시간을 걸어
  // 멈춘 연결은 끊고 그 덩어리를 다시 받음. 8MiB를 2분 안에 못 받으면(약 70KB/s 미만) 멈춘 것으로 봄.
  // Wikimedia에서 실측한 속도(약 0.18MiB/s)로 한 덩어리에 약 45초 → 여유 약 2.7배
  pub chunk: u64,
  pub chunk_timeout: Duration,
  // 한 연결이 진행 없이 연속으로 이만큼 실패하면 예비 주소로 넘어가거나 멈춤
  pub max_failures: u32,
  // 재시도 대기: 실패 횟수 × retry_wait (최대 max_wait)
  pub retry_wait: Duration,
  pub max_wait: Duration,
  // 속도 측정 (미러마다 차례로): 최대 크기, 첫 데이터 뒤 재는 시간, 첫 데이터까지 기다리는 시간
  pub probe_bytes: u64,
  pub probe_window: Duration,
  pub probe_timeout: Duration,
  // 다 받은 뒤 Wikimedia가 공개하는 SHA-1과 비교 (여러 미러에서 섞어 받으므로)
  pub verify: bool,
  // 여러 미러에서 동시에 받기 (false면 가장 빠른 한 곳에서만, 실패하면 다음 곳으로)
  pub parallel: bool,
  // 동시에 받을 때 바로 쓰는 미러의 최소 속도 (가장 빠른 곳 대비). 이보다 느린 곳은 예비로만
  pub min_share: f64,
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
      probe_window: PROBE_WINDOW,
      probe_timeout: PROBE_TIMEOUT,
      verify: true,
      parallel: true,
      min_share: 0.2,
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


// 받다 만 파일(<파일>.part)의 진행 기록: 덩어리마다 받았는지. 파일 이름·크기·덩어리 크기가 같을 때만 이어서 씀
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct PartState {
  file: String,
  total: u64,
  chunk: u64,
  done: Vec<bool>,
}

fn state_path(part: &Path) -> PathBuf { PathBuf::from(format!("{}.state", part.display())) }

// 쓰다가 끊겨도 기록이 깨지지 않게 임시 파일에 쓰고 바꿔치기
fn save_state(path: &Path, s: &PartState) -> anyhow::Result<()> {
  let tmp = PathBuf::from(format!("{}.tmp", path.display()));
  fs::write(&tmp, bincode::serialize(s)?)?;
  fs::rename(&tmp, path)?;
  Ok(())
}

// part를 전체 크기로 만들어 두고(덩어리를 제자리에 씀) 받은 덩어리 목록을 돌려줌.
// 기록이 맞지 않으면 처음부터. 기록 없이 part만 있으면 예전 방식(앞에서부터 이어 붙임)으로 받다 만 파일로 보고
// 그 길이까지(덩어리 경계로 버림) 받은 것으로 침
fn prepare_part(part: &Path, file: &str, total: u64, chunk: u64) -> anyhow::Result<Vec<bool>> {
  let n = total.div_ceil(chunk) as usize;
  let sp = state_path(part);
  let mut done = vec![false; n];
  let part_len = fs::metadata(part).map(|m| m.len()).ok();
  let state: Option<PartState> = fs::read(&sp).ok().and_then(|b| bincode::deserialize(&b).ok());
  match (state, part_len) {
    (Some(s), Some(len)) if s.file == file && s.total == total && s.chunk == chunk && s.done.len() == n && len == total => {
      done = s.done;
      let have = done.iter().filter(|d| **d).count();
      if have > 0 { println!("[Download] resuming: {have} of {n} chunks already received"); }
    }
    (Some(_), _) => {
      println!("[Download] progress record in {} does not match this download; starting over", sp.display());
      if part_len.is_some() { fs::remove_file(part)?; }
    }
    (None, Some(len)) if len > total => {
      println!("[Download] partial file larger than remote ({len} > {total}); starting over");
      fs::remove_file(part)?;
    }
    (None, Some(len)) => {
      let full = if len == total { n } else { (len / chunk) as usize };
      for d in done.iter_mut().take(full) { *d = true; }
      if full > 0 { println!("[Download] resuming an earlier partial file: {} of {} MiB", (full as u64 * chunk).min(total) >> 20, total >> 20); }
    }
    (None, None) => {}
  }
  fs::OpenOptions::new().create(true).truncate(false).write(true).open(part)?.set_len(total)?;
  save_state(&sp, &PartState { file: file.to_string(), total, chunk, done: done.clone() })?;
  Ok(done)
}

// 덩어리 [offset, offset+len)를 받아 메모리로. 206과 전체 크기·길이가 맞아야 함
fn fetch_range(client: &Client, url: &str, offset: u64, len: u64, total: u64, timeout: Duration, received: &AtomicU64) -> anyhow::Result<Vec<u8>> {
  let mut resp = client.get(url)
    .header(reqwest::header::RANGE, format!("bytes={offset}-{}", offset + len - 1))
    .timeout(timeout)
    .send()?;
  let status = resp.status();
  if status != reqwest::StatusCode::PARTIAL_CONTENT {
    if status.is_success() { anyhow::bail!("server ignored Range"); }
    anyhow::bail!("HTTP {status}");
  }
  let t = content_range_total(&resp).context("206 response without Content-Range total")?;
  anyhow::ensure!(t == total, "remote size differs ({t} vs {total})");
  let mut data = Vec::with_capacity(len as usize);
  let mut buf = vec![0u8; 1 << 20];
  loop {
    if SHUTDOWN.load(Ordering::SeqCst) { anyhow::bail!("interrupted"); }
    let n = resp.read(&mut buf)?;
    if n == 0 { break; }
    data.extend_from_slice(&buf[..n]);
    received.fetch_add(n as u64, Ordering::Relaxed);
    anyhow::ensure!(data.len() as u64 <= len, "server sent more than requested");
  }
  anyhow::ensure!(data.len() as u64 == len, "chunk ended early at {}/{len} bytes", data.len());
  Ok(data)
}

fn write_at(file: &mut fs::File, offset: u64, data: &[u8]) -> std::io::Result<()> {
  file.seek(SeekFrom::Start(offset))?;
  file.write_all(data)?;
  // 기록에 "받음"으로 적기 전에 디스크에 내려 둠 (갑자기 꺼져도 기록과 내용이 어긋나지 않게)
  file.sync_data()
}

enum Job { Chunk(usize), Wait, Finished }

// 받을 덩어리 대기열과 진행 상황 (작업자들이 함께 씀)
struct Work {
  done: Vec<bool>,
  remaining: usize,
  pending: VecDeque<usize>,
  // 받는 중인 덩어리: (번호, 시작 시각, 받는 연결 수)
  flying: Vec<(usize, Instant, u8)>,
  state: PartState,
  state_path: PathBuf,
  save_error: Option<String>,
}

impl Work {
  fn next(&mut self) -> Job {
    if let Some(i) = self.pending.pop_front() {
      self.flying.push((i, Instant::now(), 1));
      return Job::Chunk(i);
    }
    if self.remaining == 0 { return Job::Finished; }
    // 대기열이 비었으면 가장 오래 걸리고 있는 덩어리를 같이 받음 (끝부분이 느린 미러 하나에 묶이지 않게). 먼저 끝난 쪽만 씀
    let oldest = self.flying.iter().enumerate()
      .filter(|(_, f)| f.2 == 1 && !self.done[f.0])
      .min_by_key(|(_, f)| f.1).map(|(k, _)| k);
    if let Some(k) = oldest {
      self.flying[k].2 += 1;
      return Job::Chunk(self.flying[k].0);
    }
    Job::Wait
  }

  // 이 덩어리를 받던 연결 하나가 손을 뗌. 남은 연결 수
  fn release(&mut self, i: usize) -> u8 {
    let Some(k) = self.flying.iter().position(|f| f.0 == i) else { return 0; };
    self.flying[k].2 -= 1;
    let left = self.flying[k].2;
    if left == 0 { self.flying.swap_remove(k); }
    left
  }

  fn finish(&mut self, i: usize) {
    self.release(i);
    if self.done[i] { return; }
    self.done[i] = true;
    self.state.done[i] = true;
    self.remaining -= 1;
    if let Err(e) = save_state(&self.state_path, &self.state) { self.save_error = Some(format!("{e:#}")); }
  }

  // 받다가 실패: 다른 연결도 받고 있지 않으면 대기열 맨 앞에 되돌림
  fn fail(&mut self, i: usize) {
    if self.release(i) == 0 && !self.done[i] && !self.pending.contains(&i) { self.pending.push_front(i); }
  }
}

struct Engine<'a> {
  client: &'a Client,
  part: &'a Path,
  total: u64,
  opts: &'a DownloadOpts,
  work: Mutex<Work>,
  backups: Mutex<VecDeque<String>>,
  // 연결마다 받은 바이트 (속도 표시용)와 지금 쓰는 주소
  received: Vec<AtomicU64>,
  current: Vec<Mutex<String>>,
  live: AtomicUsize,
  last_error: Mutex<Option<anyhow::Error>>,
}

impl Engine<'_> {
  fn worker(&self, slot: usize, first_url: &str) {
    let mut url = first_url.to_string();
    let mut file = match fs::OpenOptions::new().write(true).open(self.part) {
      Ok(f) => f,
      Err(e) => { *self.last_error.lock().unwrap() = Some(e.into()); return; }
    };
    let chunk = self.opts.chunk;
    let mut failures = 0u32;
    loop {
      if SHUTDOWN.load(Ordering::SeqCst) { return; }
      let job = self.work.lock().unwrap().next();
      let i = match job {
        Job::Finished => return,
        Job::Wait => { std::thread::sleep(Duration::from_millis(100)); continue; }
        Job::Chunk(i) => i,
      };
      let offset = i as u64 * chunk;
      let len = chunk.min(self.total - offset);
      match fetch_range(self.client, &url, offset, len, self.total, self.opts.chunk_timeout, &self.received[slot]) {
        Ok(data) => {
          failures = 0;
          let already = self.work.lock().unwrap().done[i];
          let written = if already { Ok(()) } else { write_at(&mut file, offset, &data) };
          match written {
            Ok(()) => self.work.lock().unwrap().finish(i),
            Err(e) => {
              // 디스크 오류는 다른 미러로 바꿔도 안 낫기 때문에 이 연결을 멈춤
              self.work.lock().unwrap().fail(i);
              *self.last_error.lock().unwrap() = Some(anyhow::Error::from(e).context(format!("writing {}", self.part.display())));
              return;
            }
          }
        }
        Err(e) => {
          self.work.lock().unwrap().fail(i);
          if SHUTDOWN.load(Ordering::SeqCst) { return; }
          failures += 1;
          if failures > self.opts.max_failures {
            let next = self.backups.lock().unwrap().pop_front();
            eprintln!("\n[Download] {} failed {} times in a row: {e:#}", host(&url), self.opts.max_failures);
            *self.last_error.lock().unwrap() = Some(e);
            let Some(next) = next else { return; };
            println!("[Download] switching this connection to {next}");
            url = next;
            *self.current[slot].lock().unwrap() = host(&url).to_string();
            failures = 0;
            continue;
          }
          let wait = (self.opts.retry_wait * failures).min(self.opts.max_wait);
          eprintln!("\n[Download] {}: {e:#} — retry {failures}/{} in {:.1}s", host(&url), self.opts.max_failures, wait.as_secs_f32());
          std::thread::sleep(wait);
        }
      }
    }
  }
}

// part에 남은 덩어리를 primary 미러마다 연결 하나씩 동시에 받음. 연결이 계속 실패하면 backups에서 다음 주소를 가져감
fn download_chunks(client: &Client, primary: &[String], backups: Vec<String>, part: &Path, file: &str, total: u64, opts: &DownloadOpts) -> anyhow::Result<()> {
  let done = prepare_part(part, file, total, opts.chunk)?;
  let remaining = done.iter().filter(|d| !**d).count();
  if remaining == 0 { return Ok(()); }
  let chunk_len = |i: usize| opts.chunk.min(total - i as u64 * opts.chunk);
  let done_bytes: u64 = (0..done.len()).filter(|&i| done[i]).map(chunk_len).sum();
  println!("[Download] {} of {} MiB to go, from {} at once{}", (total - done_bytes) >> 20, total >> 20,
    primary.iter().map(|u| host(u)).collect::<Vec<_>>().join(", "),
    if backups.is_empty() { String::new() } else { format!(" (backup: {})", backups.iter().map(|u| host(u)).collect::<Vec<_>>().join(", ")) });
  let engine = Engine {
    client, part, total, opts,
    work: Mutex::new(Work {
      pending: (0..done.len()).filter(|&i| !done[i]).collect(),
      remaining, flying: Vec::new(),
      state: PartState { file: file.to_string(), total, chunk: opts.chunk, done: done.clone() },
      state_path: state_path(part), save_error: None, done,
    }),
    backups: Mutex::new(backups.into()),
    received: primary.iter().map(|_| AtomicU64::new(0)).collect(),
    current: primary.iter().map(|u| Mutex::new(host(u).to_string())).collect(),
    live: AtomicUsize::new(primary.len()),
    last_error: Mutex::new(None),
  };

  std::thread::scope(|s| {
    for (slot, url) in primary.iter().enumerate() {
      let e = &engine;
      s.spawn(move || { e.worker(slot, url); e.live.fetch_sub(1, Ordering::SeqCst); });
    }
    // 1초마다 진행 표시: 받은 양, 전체 속도, 연결마다 속도
    let mut last = Instant::now();
    let mut last_bytes: Vec<u64> = vec![0; primary.len()];
    while engine.live.load(Ordering::SeqCst) > 0 {
      std::thread::sleep(Duration::from_millis(100));
      if last.elapsed() < Duration::from_secs(1) { continue; }
      let dt = last.elapsed().as_secs_f64();
      last = Instant::now();
      let mib = f64::from(1 << 20);
      let speeds: Vec<f64> = engine.received.iter().zip(&mut last_bytes)
        .map(|(r, prev)| { let now = r.load(Ordering::Relaxed); let v = (now - *prev) as f64 / dt / mib; *prev = now; v })
        .collect();
      let have: u64 = { let w = engine.work.lock().unwrap(); (0..w.done.len()).filter(|&i| w.done[i]).map(chunk_len).sum() };
      let per: Vec<String> = engine.current.iter().zip(&speeds).map(|(h, v)| format!("{} {v:.1}", h.lock().unwrap())).collect();
      print!("\r[Download] {} / {} MiB  {:.1} MiB/s  ({})          ", have >> 20, total >> 20, speeds.iter().sum::<f64>(), per.join(", "));
      let _ = std::io::stdout().flush();
    }
  });
  println!();

  let work = engine.work.into_inner().unwrap();
  if let Some(e) = work.save_error { anyhow::bail!("could not save download progress to {}: {e}", state_path(part).display()); }
  if work.remaining == 0 { return Ok(()); }
  if SHUTDOWN.load(Ordering::SeqCst) {
    anyhow::bail!("download interrupted with {} of {} chunks left (re-run to resume)", work.remaining, work.done.len());
  }
  let err = engine.last_error.into_inner().unwrap().unwrap_or_else(|| anyhow::anyhow!("download failed"));
  Err(err.context(format!("all {} dump URLs failed; received data is kept in {} for the next run (delete it and its .state file to restart)",
    primary.len() + engine.backups.into_inner().unwrap().len(), part.display())))
}

// 파일 크기 (Range 0-0 요청). None = 서버가 Range를 무시함. 실패하면 대기했다가 다시 물음
fn remote_size(client: &Client, url: &str, opts: &DownloadOpts) -> anyhow::Result<Option<u64>> {
  let mut failures = 0u32;
  loop {
    let r = client.get(url).header(reqwest::header::RANGE, "bytes=0-0").timeout(Duration::from_secs(30)).send();
    let res = match r {
      Ok(resp) if resp.status() == reqwest::StatusCode::PARTIAL_CONTENT => content_range_total(&resp).context("206 response without Content-Range total").map(Some),
      Ok(resp) if resp.status().is_success() => Ok(None),
      Ok(resp) => Err(anyhow::anyhow!("HTTP {}", resp.status())),
      Err(e) => Err(e.into()),
    };
    match res {
      Ok(v) => return Ok(v),
      Err(e) if SHUTDOWN.load(Ordering::SeqCst) => return Err(e),
      Err(e) => {
        failures += 1;
        if failures > opts.max_failures { return Err(e.context("all 1 dump URLs failed")); }
        let wait = (opts.retry_wait * failures).min(opts.max_wait);
        eprintln!("[Download] {e:#} — retry {failures}/{} in {:.1}s", opts.max_failures, wait.as_secs_f32());
        std::thread::sleep(wait);
      }
    }
  }
}

// Range를 지원하지 않는 서버: 덩어리 없이 처음부터 전부 받음 (멈춤 감지·이어받기 불가)
fn download_whole(client: &Client, url: &str, part: &Path) -> anyhow::Result<()> {
  println!("[Download] server ignored Range, downloading whole file without stall detection or resume");
  let mut resp = client.get(url).send()?.error_for_status()?;
  let total = resp.content_length();
  let mut file = fs::File::create(part)?;
  let n = std::io::copy(&mut resp, &mut file)?;
  file.flush()?;
  if let Some(t) = total { anyhow::ensure!(n == t, "connection closed at {n}/{t} bytes"); }
  Ok(())
}

// 덤프를 path로 내려받음. 주소가 여럿이면 하나씩 속도를 재서, 동시에 받기면 빠른 미러들(같은 서버는 하나)에서 나눠 받고,
// 아니면 가장 빠른 곳에서 받다가 계속 실패하면 다음 곳으로. "<path>.part"에 받고, 검증 뒤 rename
pub fn download_dump(client: &Client, urls: &[String], path: &Path, opts: &DownloadOpts) -> anyhow::Result<()> {
  anyhow::ensure!(!urls.is_empty(), "no dump URL");
  let part = PathBuf::from(format!("{}.part", path.display()));
  let file = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
  if urls.len() == 1 {
    match remote_size(client, &urls[0], opts)? {
      Some(total) => download_chunks(client, &urls[..1], Vec::new(), &part, &file, total, opts)?,
      None => download_whole(client, &urls[0], &part)?,
    }
  } else {
    println!("[Mirror] measuring {} URLs one at a time...", urls.len());
    let ranked = rank_mirrors(client, urls, opts)?;
    let total = ranked[0].total.context("mirrors did not report the file size")?;
    let (primary, backups) = if opts.parallel {
      split_mirrors(&ranked, opts.min_share, resolve)
    } else {
      (vec![ranked[0].url.clone()], ranked[1..].iter().map(|p| p.url.clone()).collect())
    };
    download_chunks(client, &primary, backups, &part, &file, total, opts)?;
  }
  if opts.verify && let Err(e) = verify_download(client, urls, &part, path) {
    let _ = fs::remove_file(state_path(&part));
    return Err(e);
  }
  fs::rename(&part, path)?;
  let _ = fs::remove_file(state_path(&part));
  println!("[Download] done: {}", path.display());
  Ok(())
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

  // 여러 서버에 걸쳐 덤프 요청을 동시에 몇 개 처리 중인지 (지금, 최대)
  #[derive(Default)]
  struct Tracker { active: AtomicUsize, max: AtomicUsize }
  struct Active(Arc<Tracker>);
  impl Drop for Active { fn drop(&mut self) { self.0.active.fetch_sub(1, Ordering::SeqCst); } }

  // body를 DUMP 경로에서 Range 요청대로 보내는 서버. (덤프 주소, 덤프 요청 수)
  fn serve(body: Arc<Vec<u8>>, mode: Mode) -> (String, Arc<AtomicUsize>) {
    serve_files(vec![(DUMP.to_string(), body)], mode)
  }

  fn serve_files(files: Vec<(String, Arc<Vec<u8>>)>, mode: Mode) -> (String, Arc<AtomicUsize>) {
    serve_tracked(files, mode, Arc::new(Tracker::default()))
  }

  // 경로별 파일을 보내는 서버. 첫 파일이 덤프이고 mode·요청 수는 덤프에만 적용, 없는 경로는 404
  fn serve_tracked(files: Vec<(String, Arc<Vec<u8>>)>, mode: Mode, tracker: Arc<Tracker>) -> (String, Arc<AtomicUsize>) {
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
        let tracker = tracker.clone();
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
          let mut _active = None;
          if path == files[0].0 {
            let now = tracker.active.fetch_add(1, Ordering::SeqCst) + 1;
            tracker.max.fetch_max(now, Ordering::SeqCst);
            _active = Some(Active(tracker.clone()));
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
      probe_window: Duration::from_secs(2),
      probe_timeout: Duration::from_secs(5),
      verify: true,
      // 예전 테스트(가장 빠른 곳에서 받다가 다음 곳으로)는 동시에 받기를 끄고 확인
      parallel: false,
      min_share: 0.0,
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
    let ranked = rank_mirrors(&client(), &urls, &opts()).unwrap();
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

    assert!(!dir.join("a.bz2.part.state").exists());

    // 주소 하나: 속도 측정 없이 크기만 묻고 바로
    let (only, hits_only) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    let path = dir.join("b.bz2");
    download_dump(&client(), &[only], &path, &opts()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);
    assert_eq!(hits_only.load(Ordering::SeqCst), 1 + 7); // 크기 요청 + 100KB 덩어리 7개

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
    let p = probe(&client(), &url, 1 << 20, Duration::from_secs(5), Duration::from_secs(5)).unwrap();
    assert!(p.latency >= Duration::from_millis(500), "{:?}", p.latency);
    assert!(p.bytes_per_sec > 10.0 * f64::from(1 << 20), "{} B/s", p.bytes_per_sec);

    // 그래도 전송 자체가 느린 서버는 느리게 잼
    let (slow, _) = serve(body(1 << 20), Mode::Ok { delay_per_64k_ms: 30 });
    let q = probe(&client(), &slow, 1 << 20, Duration::from_secs(5), Duration::from_secs(5)).unwrap();
    assert!(q.bytes_per_sec < 3.0 * f64::from(1 << 20), "{} B/s", q.bytes_per_sec);
  }

  #[test]
  fn probe_reports_size_and_speed() {
    let data = body(300_000);
    let (url, _) = serve(data, Mode::Ok { delay_per_64k_ms: 0 });
    let p = probe(&client(), &url, 128 << 10, Duration::from_secs(5), Duration::from_secs(5)).unwrap();
    assert_eq!(p.total, Some(300_000));
    assert!(p.bytes_per_sec > 0.0);
  }

  #[test]
  fn steady_speed_skips_the_slow_start() {
    // 1초 동안 0.1MB씩 천천히, 그 뒤 1초 동안 1MB씩: 평균은 1.1MB/s지만 뒤쪽 절반은 2MB/s
    let samples: Vec<(f64, u64)> = (1..=10u32).map(|k| (f64::from(k) * 0.1, u64::from(k) * 10_000))
      .chain((1..=10u32).map(|k| (1.0 + f64::from(k) * 0.1, 100_000 + u64::from(k) * 100_000))).collect();
    let v = steady_speed(&samples).unwrap();
    assert!((v - 1_000_000.0).abs() < 1.0, "{v}");
    assert_eq!(steady_speed(&[]), None);
    assert!((steady_speed(&[(0.5, 1000)]).unwrap() - 2000.0).abs() < 1e-6);
  }

  #[test]
  fn probe_stops_after_the_window() {
    // 아주 느리게 보내는 서버: 정해진 시간만 재고 멈춤 (전부 받으면 수십 초)
    let (url, _) = serve(body(4 << 20), Mode::Ok { delay_per_64k_ms: 50 });
    let t = Instant::now();
    let p = probe(&client(), &url, 4 << 20, Duration::from_millis(500), Duration::from_secs(5)).unwrap();
    assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
    assert!(p.bytes_per_sec > 0.0 && p.bytes_per_sec < 4.0 * f64::from(1 << 20), "{}", p.bytes_per_sec);
  }

  #[test]
  fn mirrors_are_measured_one_at_a_time() {
    let tracker = Arc::new(Tracker::default());
    let urls: Vec<String> = (0..3).map(|_| serve_tracked(vec![(DUMP.to_string(), body(400_000))], Mode::Ok { delay_per_64k_ms: 5 }, tracker.clone()).0).collect();
    let ranked = rank_mirrors(&client(), &urls, &opts()).unwrap();
    assert_eq!(ranked.len(), 3);
    assert_eq!(tracker.max.load(Ordering::SeqCst), 1);
  }

  fn probe_of(url: &str, mib_s: f64) -> Probe {
    Probe { url: url.to_string(), total: Some(1), bytes_per_sec: mib_s, latency: Duration::ZERO }
  }

  #[test]
  fn split_uses_one_address_per_server_and_keeps_slow_ones_as_backup() {
    let ip = |n: u8| SocketAddr::from(([10, 0, 0, n], 443));
    let addrs = |u: &str| match u {
      "https://a/x" => vec![ip(1), ip(2)],
      "https://b/x" => vec![ip(2)],          // a와 같은 서버
      "https://c/x" => vec![ip(3)],
      "https://d/x" => vec![ip(4)],          // 너무 느림
      _ => vec![],                           // 이름을 못 풂: 다른 서버로 봄
    };
    let ranked = [probe_of("https://a/x", 10.0), probe_of("https://b/x", 9.0), probe_of("https://c/x", 5.0),
      probe_of("https://e/x", 3.0), probe_of("https://d/x", 1.5)];
    let (primary, backup) = split_mirrors(&ranked, 0.2, addrs);
    assert_eq!(primary, ["https://a/x", "https://c/x", "https://e/x"]);
    assert_eq!(backup, ["https://b/x", "https://d/x"]);
    // 가장 빠른 곳은 기준과 상관없이 항상 씀
    let (p, b) = split_mirrors(&ranked[..1], 0.99, addrs);
    assert_eq!((p.len(), b.len()), (1, 0));
    assert_eq!(host("https://m.org:8443/a/b?x"), "m.org:8443");
    assert_eq!(resolve("http://127.0.0.1:8080/x"), vec![SocketAddr::from(([127, 0, 0, 1], 8080))]);
  }

  fn par() -> DownloadOpts { DownloadOpts { parallel: true, ..opts() } }

  #[test]
  fn parallel_download_splits_chunks_across_mirrors() {
    let dir = std::env::temp_dir().join(format!("p2anut-par-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let data = body(1_050_000); // 100KB 덩어리 11개 (마지막은 50KB)
    let servers: Vec<(String, Arc<AtomicUsize>)> = (0..3).map(|_| serve(data.clone(), Mode::Ok { delay_per_64k_ms: 10 })).collect();
    let urls: Vec<String> = servers.iter().map(|s| s.0.clone()).collect();
    let path = dir.join("p.bz2");
    download_dump(&client(), &urls, &path, &par()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);
    // 미러마다 측정 1번 + 덩어리 1개 이상
    for (_, hits) in &servers { assert!(hits.load(Ordering::SeqCst) >= 2, "{}", hits.load(Ordering::SeqCst)); }
    assert!(!dir.join("p.bz2.part").exists() && !dir.join("p.bz2.part.state").exists());

    // 미러 하나가 중간에 고장 나도 나머지로 끝까지
    let (broken, _) = serve(data.clone(), Mode::BreakAfter(3));
    let (ok1, _) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 10 });
    let (ok2, _) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 10 });
    let path = dir.join("q.bz2");
    download_dump(&client(), &[broken, ok1, ok2], &path, &par()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);

    // 크기가 다른 미러는 빠짐 (크기는 가장 많은 주소가 알려 준 값)
    let (other_size, other_hits) = serve(body(900_000), Mode::Ok { delay_per_64k_ms: 0 });
    let (good1, _) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    let (good2, _) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    let path = dir.join("r.bz2");
    download_dump(&client(), &[other_size, good1, good2], &path, &par()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);
    assert_eq!(other_hits.load(Ordering::SeqCst), 1); // 측정만
    fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn resumes_only_missing_chunks() {
    let dir = std::env::temp_dir().join(format!("p2anut-resume-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let data = body(650_000); // 덩어리 7개
    let path = dir.join("r.bz2");
    let part = dir.join("r.bz2.part");

    // 크기 요청 + 덩어리 3개 뒤 고장 → 실패, 받은 덩어리와 기록은 남음
    let (dying, _) = serve(data.clone(), Mode::BreakAfter(4));
    let err = download_dump(&client(), &[dying], &path, &opts()).unwrap_err();
    assert!(format!("{err:#}").contains("all 1 dump URLs failed"), "{err:#}");
    let st: PartState = bincode::deserialize(&fs::read(state_path(&part)).unwrap()).unwrap();
    assert_eq!(st.done.iter().filter(|d| **d).count(), 3);
    assert_eq!(fs::metadata(&part).unwrap().len(), 650_000);

    // 다시 받으면 남은 4개만 요청 (+ 크기 요청)
    let (healthy, hits) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    download_dump(&client(), &[healthy], &path, &opts()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);
    assert_eq!(hits.load(Ordering::SeqCst), 1 + 4);

    // 기록이 다른 다운로드(덩어리 크기가 다름)의 것이면 처음부터
    let path2 = dir.join("s.bz2");
    let part2 = dir.join("s.bz2.part");
    fs::write(&part2, vec![0u8; 650_000]).unwrap();
    save_state(&state_path(&part2), &PartState { file: "s.bz2".into(), total: 650_000, chunk: 1234, done: vec![true; 527] }).unwrap();
    let (fresh, hits) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    download_dump(&client(), &[fresh], &path2, &opts()).unwrap();
    assert_eq!(fs::read(&path2).unwrap(), *data);
    assert_eq!(hits.load(Ordering::SeqCst), 1 + 7);
    fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn continues_an_old_style_partial_file() {
    let dir = std::env::temp_dir().join(format!("p2anut-old-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let data = body(650_000);
    let path = dir.join("o.bz2");
    // 예전 방식(앞에서부터 이어 붙임)으로 250KB까지 받다 만 파일, 기록 없음 → 덩어리 2개(200KB)는 받은 것으로
    fs::write(dir.join("o.bz2.part"), &data[..250_000]).unwrap();
    let (url, hits) = serve(data.clone(), Mode::Ok { delay_per_64k_ms: 0 });
    download_dump(&client(), &[url], &path, &opts()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), *data);
    assert_eq!(hits.load(Ordering::SeqCst), 1 + 5);
    fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn idle_connections_help_with_the_slowest_chunk() {
    let mut w = Work {
      done: vec![false; 3], remaining: 3, pending: VecDeque::from([0, 1, 2]), flying: Vec::new(),
      state: PartState { file: String::new(), total: 3, chunk: 1, done: vec![false; 3] },
      state_path: std::env::temp_dir().join(format!("p2anut-work-{}.state", std::process::id())), save_error: None,
    };
    assert!(matches!(w.next(), Job::Chunk(0)));
    assert!(matches!(w.next(), Job::Chunk(1)));
    assert!(matches!(w.next(), Job::Chunk(2)));
    w.finish(2);
    // 대기열이 비면 가장 오래된 덩어리(0)를 같이 받고, 그다음은 1, 그 뒤로는 기다림
    assert!(matches!(w.next(), Job::Chunk(0)));
    assert!(matches!(w.next(), Job::Chunk(1)));
    assert!(matches!(w.next(), Job::Wait));
    // 한쪽이 끝내면 끝, 다른 쪽이 나중에 끝내도 두 번 세지 않음
    w.finish(0); w.finish(0);
    assert_eq!(w.remaining, 1);
    // 같이 받던 둘이 모두 실패하면 대기열로 되돌아감
    w.fail(1); w.fail(1);
    assert_eq!(w.pending, VecDeque::from([1]));
    assert!(matches!(w.next(), Job::Chunk(1)));
    w.finish(1);
    assert!(matches!(w.next(), Job::Finished));
    let _ = fs::remove_file(&w.state_path);
  }
}
