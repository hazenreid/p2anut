#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use std::collections::HashMap;
use anyhow::Context;
use serde::{Serialize, Deserialize};
use crate::layout::CHARS;

#[derive(Clone)]
pub struct CorpusArrays {
  pub keys: Vec<char>,
  pub freq: Box<[u64]>,
  pub nz_bigrams: Vec<(u8, u8, u64)>,
  pub nz_trigrams: Vec<(u8, u8, u8, u64)>,
  pub bigram_flat: Vec<usize>,
  pub bigram_offsets: [usize; CHARS + 1],
  pub trigram_flat: Vec<usize>,
  pub trigram_offsets: [usize; CHARS + 1],
}

// 글자 id마다 그 글자가 들어간 n-gram 번호 목록을 이어 붙인 것과 시작 위치 (최적화의 점수 차이 계산용)
fn index_lists(members: impl Iterator<Item = Vec<u8>>) -> (Vec<usize>, [usize; CHARS + 1]) {
  let mut lists: Vec<Vec<usize>> = vec![Vec::new(); CHARS];
  for (i, mut ids) in members.enumerate() {
    ids.sort_unstable();
    ids.dedup();
    for id in ids { lists[id as usize].push(i); }
  }
  let mut flat = Vec::new();
  let mut offsets = [0usize; CHARS + 1];
  for id in 0..CHARS {
    offsets[id] = flat.len();
    flat.extend_from_slice(&lists[id]);
  }
  offsets[CHARS] = flat.len();
  (flat, offsets)
}

impl CorpusArrays {
  // 빈도 최하위 n글자의 id 비트마스크 (동률은 id 순). n = 0이면 0 (제약 없음)
  pub fn rare_mask(&self, n: usize) -> u32 {
    let mut ids: Vec<usize> = (0..self.keys.len()).collect();
    ids.sort_by_key(|&i| (self.freq[i], i));
    ids.iter().take(n).fold(0, |m, &i| m | 1 << i)
  }
}

const CORPUS_MAGIC: &[u8] = b"P2ANUT-CORPUS\0";
const CORPUS_FORMAT: u32 = 1;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct CorpusStats {
  pub letters: HashMap<char, u64>,
  pub bigrams: HashMap<(char, char), u64>,
  pub trigrams: HashMap<(char, char, char), u64>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct FastCorpus {
  pub letters: Box<[u64; CHARS]>,
  pub bigrams: Box<[u64; CHARS * CHARS]>,
  pub trigrams: Box<[u64]>,
  // 공백(연속 공백은 하나로)을 양옆에 따라 나눈 개수. 확인용이며 코퍼스 파일에는 저장하지 않음 (SPACE_CASES 순서). 첫 경우만 셈
  pub space_cases: [u64; SPACE_CASES.len()],
}

// 공백 경우: 첫 경우(단어 사이)만 누름과 n-gram으로 셈, 나머지는 개수만 보여 줌
pub const SPACE_CASES: [&str; 5] = [
  "letter _ letter (counted)",
  "letter _ punctuation/digit or the reverse (not counted)",
  "punctuation/digit _ punctuation/digit (not counted)",
  "double spaces (not counted)",
  "line start/end, markup or other symbols (not counted)",
];

impl FastCorpus {
  pub fn new() -> Self {
    Self {
      letters: Box::new([0; CHARS]),
      bigrams: Box::new([0; CHARS * CHARS]),
      trigrams: vec![0u64; CHARS * CHARS * CHARS].into_boxed_slice(),
      space_cases: [0; SPACE_CASES.len()],
    }
  }

  #[inline]
  pub fn add_letter(&mut self, id: u8) {
    self.letters[id as usize] += 1;
  }

  #[inline]
  pub fn add_bigram(&mut self, p: u8, c: u8) {
    self.bigrams[p as usize * CHARS + c as usize] += 1;
  }

  #[inline]
  pub fn add_trigram(&mut self, pp: u8, p: u8, c: u8) {
    self.trigrams[(pp as usize * CHARS + p as usize) * CHARS + c as usize] += 1;
  }

  // 병렬 분석에서 덩어리별 결과 합치기
  pub fn merge(&mut self, other: &FastCorpus) {
    for (a, b) in self.letters.iter_mut().zip(other.letters.iter()) { *a += b; }
    for (a, b) in self.bigrams.iter_mut().zip(other.bigrams.iter()) { *a += b; }
    for (a, b) in self.trigrams.iter_mut().zip(other.trigrams.iter()) { *a += b; }
    for (a, b) in self.space_cases.iter_mut().zip(other.space_cases.iter()) { *a += b; }
  }

  // 중간 저장용 (고정 길이 배열은 serde가 직접 못 다뤄서 Vec으로)
  pub fn to_vecs(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    (self.letters.to_vec(), self.bigrams.to_vec(), self.trigrams.to_vec(), self.space_cases.to_vec())
  }

  pub fn from_vecs(letters: &[u64], bigrams: &[u64], trigrams: &[u64], space_cases: &[u64]) -> anyhow::Result<Self> {
    anyhow::ensure!(letters.len() == CHARS && bigrams.len() == CHARS * CHARS && trigrams.len() == CHARS * CHARS * CHARS
      && space_cases.len() == SPACE_CASES.len(), "corpus array sizes do not match");
    let mut f = Self::new();
    f.letters.copy_from_slice(letters);
    f.bigrams.copy_from_slice(bigrams);
    f.trigrams.copy_from_slice(trigrams);
    f.space_cases.copy_from_slice(space_cases);
    Ok(f)
  }

  // 분석 뒤 공백 경우별 개수 출력 (공백 규칙이 생각대로 동작하는지 확인용)
  pub fn print_space_cases(&self) {
    let letters: u64 = self.letters[..26].iter().sum();
    let per = |n: u64| n as f64 * 100.0 / letters.max(1) as f64;
    println!("[Spaces] presses per 100 letters by neighbours:");
    for (name, &n) in SPACE_CASES.iter().zip(&self.space_cases) {
      println!("  {name:<55} {:>6.2}  ({n})", per(n));
    }
  }

  pub fn into_stats(self) -> CorpusStats {
    let mut stats = CorpusStats::new();
    let to_char = |idx: usize| -> char {
      if idx < 26 { (idx as u8 + b'a') as char } else { ' ' }
    };

    for (i, &cnt) in self.letters.iter().enumerate() {
      if cnt > 0 { stats.letters.insert(to_char(i), cnt); }
    }
    for (i, &cnt) in self.bigrams.iter().enumerate() {
      if cnt > 0 { stats.bigrams.insert((to_char(i / CHARS), to_char(i % CHARS)), cnt); }
    }
    for (i, &cnt) in self.trigrams.iter().enumerate() {
      if cnt > 0 {
        let pp = to_char(i / (CHARS * CHARS));
        let p = to_char((i / CHARS) % CHARS);
        let c = to_char(i % CHARS);
        stats.trigrams.insert((pp, p, c), cnt);
      }
    }
    stats
  }
}

impl CorpusStats {
  pub fn new() -> Self {
    Self {
      letters: HashMap::new(),
      bigrams: HashMap::new(),
      trigrams: HashMap::new(),
    }
  }

  // 배치할 키는 a-z와 공백(id 26, 27번째 키). 공백이 낀 n-gram도 글자끼리의 것과 똑같이 씀
  pub fn to_arrays(&self) -> CorpusArrays {
    let keys: Vec<char> = ('a'..='z').chain(std::iter::once(' ')).collect();
    let lu = |ch: char| if ch == ' ' { Some(26) } else { ch.is_ascii_lowercase().then(|| ch as u8 - b'a') };

    let mut freq = vec![0u64; CHARS].into_boxed_slice();
    for (&ch, &cnt) in &self.letters {
      if let Some(i) = lu(ch) { freq[i as usize] = cnt; }
    }

    let mut nz_bigrams: Vec<(u8, u8, u64)> = self.bigrams.iter()
      .filter_map(|(&(a, b), &cnt)| {
        Some((lu(a)?, lu(b)?, cnt))
      })
      .collect();
    nz_bigrams.sort_unstable_by_key(|&(a, b, _)| (a, b));

    let mut nz_trigrams: Vec<(u8, u8, u8, u64)> = self.trigrams.iter()
      .filter_map(|(&(a, b, c), &cnt)| {
        Some((lu(a)?, lu(b)?, lu(c)?, cnt))
      })
      .collect();
    nz_trigrams.sort_unstable_by_key(|&(a, b, c, _)| (a, b, c));

    let (bigram_flat, bigram_offsets) = index_lists(nz_bigrams.iter().map(|&(a, b, _)| vec![a, b]));
    let (trigram_flat, trigram_offsets) = index_lists(nz_trigrams.iter().map(|&(a, b, c, _)| vec![a, b, c]));

    CorpusArrays { keys, freq, nz_bigrams, nz_trigrams, bigram_flat, bigram_offsets, trigram_flat, trigram_offsets }
  }

  // 모든 트라이그램 창 (공백 = id 26). 결과의 창 기준 통계 표시용
  pub fn windows(&self) -> Vec<(u8, u8, u8, u64)> {
    let id = |ch: char| if ch == ' ' { Some(26u8) } else { ch.is_ascii_lowercase().then(|| ch as u8 - b'a') };
    self.trigrams.iter().filter_map(|(&(a, b, c), &n)| Some((id(a)?, id(b)?, id(c)?, n))).collect()
  }

  // 빈도 상위부터 누적 비율이 bigram/trigram 기준에 닿을 때까지만 보존 (1.0 = 전부)
  // 최적화 속도를 위해 불러올 때 적용. 파일에는 전체 통계를 저장해서 기준을 바꿔도 재분석이 필요 없음.
  // 묶음마다 따로 자름: 글자끼리 / 가운데가 공백인 것("a _ c") / 나머지 공백이 낀 것.
  // 빈도가 큰 공백 n-gram이 섞여도 글자끼리의 n-gram은 늘 같은 것이 남음
  // other를 글자(a-z) 수 기준 share 비율로 섞은 통계 (self는 1 - share). 글자 수는 self와 같게 맞춤
  pub fn mixed(&self, other: &CorpusStats, share: f64) -> CorpusStats {
    fn blend<K: Eq + std::hash::Hash + Copy>(a: &HashMap<K, u64>, ka: f64, b: &HashMap<K, u64>, kb: f64) -> HashMap<K, u64> {
      let mut m: HashMap<K, f64> = HashMap::new();
      for (k, &n) in a { *m.entry(*k).or_default() += n as f64 * ka; }
      for (k, &n) in b { *m.entry(*k).or_default() += n as f64 * kb; }
      #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
      m.into_iter().map(|(k, v)| (k, v.round() as u64)).filter(|&(_, n)| n > 0).collect()
    }
    let letters = |s: &CorpusStats| s.letters.iter().filter(|(c, _)| c.is_ascii_lowercase()).map(|(_, &n)| n).sum::<u64>() as f64;
    let (la, lb) = (letters(self), letters(other));
    let (ka, kb) = (1.0 - share, if lb > 0.0 { share * la / lb } else { 0.0 });
    CorpusStats {
      letters: blend(&self.letters, ka, &other.letters, kb),
      bigrams: blend(&self.bigrams, ka, &other.bigrams, kb),
      trigrams: blend(&self.trigrams, ka, &other.trigrams, kb),
    }
  }

  pub fn trim_by_coverage(&mut self, bigram: f64, trigram: f64) {
    fn trim<K: Eq + std::hash::Hash + Clone + Ord>(map: &mut HashMap<K, u64>, threshold: f64) {
      if map.is_empty() || threshold >= 1.0 { return; }
      let total: u64 = map.values().sum();
      let limit = (total as f64 * threshold) as u64;
      let mut items: Vec<(K, u64)> = map.iter().map(|(k, &v)| (k.clone(), v)).collect();
      // 동률은 키 순으로 정렬: HashMap 순회 순서는 실행마다 달라서, 경계에서 잘리는 항목이 바뀌면 재현이 깨짐
      items.sort_by(|(ka, ca), (kb, cb)| cb.cmp(ca).then_with(|| ka.cmp(kb)));

      let mut cumulative = 0u64;
      let mut keep_count = 0;
      for (_, count) in &items {
        cumulative += count;
        keep_count += 1;
        if cumulative >= limit { break; }
      }
      items.truncate(keep_count);
      *map = items.into_iter().collect();
    }
    // group: 키마다 묶음 번호 (0..n)
    fn trim_groups<K: Eq + std::hash::Hash + Clone + Ord>(map: &mut HashMap<K, u64>, threshold: f64, n: usize, group: impl Fn(&K) -> usize) {
      let mut groups: Vec<HashMap<K, u64>> = vec![HashMap::new(); n];
      for (k, v) in map.drain() { groups[group(&k)].insert(k, v); }
      for g in &mut groups { trim(g, threshold); map.extend(g.drain()); }
    }
    trim_groups(&mut self.bigrams, bigram, 2, |&(a, b)| usize::from(a == ' ' || b == ' '));
    trim_groups(&mut self.trigrams, trigram, 3, |&(a, b, c)| {
      if b == ' ' { 1 } else if a == ' ' || c == ' ' { 2 } else { 0 }
    });
  }

  // 파일 형식: CORPUS_MAGIC + 파일 형식 버전(u32) + 파서 버전(u32) + bincode(CorpusStats)
  pub fn save(&self, path: &str, parser_version: u32) -> anyhow::Result<()> {
    self.save_with(path, CORPUS_MAGIC, parser_version)
  }

  // 머리표가 다른 코퍼스(심볼 레이어용 코드 코퍼스)도 같은 형식으로 저장
  pub fn save_with(&self, path: &str, magic: &[u8], parser_version: u32) -> anyhow::Result<()> {
    let mut bytes = magic.to_vec();
    bytes.extend_from_slice(&CORPUS_FORMAT.to_le_bytes());
    bytes.extend_from_slice(&parser_version.to_le_bytes());
    bytes.extend_from_slice(&bincode::serialize(self)?);
    // 쓰다가 끊겨도 기존 파일이 깨지지 않게 임시 파일에 쓰고 바꿔치기
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
  }

  // (통계, 만들 때의 파서 버전)
  pub fn load(path: &str) -> anyhow::Result<(Self, u32)> {
    Self::load_with(path, CORPUS_MAGIC)
  }

  pub fn load_with(path: &str, magic: &[u8]) -> anyhow::Result<(Self, u32)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let header = magic.len() + 8;
    if bytes.len() < header || !bytes.starts_with(magic) {
      if magic == CORPUS_MAGIC { anyhow::bail!("{path} is an old-format corpus (trimmed at save time); rebuild it with `analyze --reset`"); }
      anyhow::bail!("{path} is not a {} file", String::from_utf8_lossy(magic.strip_suffix(b"\0").unwrap_or(magic)));
    }
    let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().expect("4 bytes"));
    let format = word(magic.len());
    anyhow::ensure!(format == CORPUS_FORMAT, "{path} has corpus format {format}, expected {CORPUS_FORMAT}; rebuild it with `analyze --reset`");
    let parser_version = word(magic.len() + 4);
    Ok((bincode::deserialize(&bytes[header..])?, parser_version))
  }

  // 글자 비율은 a-z끼리, 공백은 글자 100개당 횟수로 따로. n-gram은 공백 없는 것과 공백이 낀 것을 따로 (공백은 _로 표시)
  pub fn print(&self) {
    let show = |c: char| if c == ' ' { '_' } else { c };
    let total_letters: u64 = self.letters.iter().filter(|(c, _)| **c != ' ').map(|(_, n)| n).sum();
    let mut letters: Vec<_> = self.letters.iter().filter(|(c, _)| **c != ' ').collect();
    letters.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!("\n[Letters]");
    for (ch, count) in &letters {
      println!("  {}: {:>12}  ({:.2}%)", ch, count, **count as f64 / total_letters as f64 * 100.0);
    }
    let spaces = self.letters.get(&' ').copied().unwrap_or(0);
    let between: u64 = self.trigrams.iter().filter(|(k, _)| k.1 == ' ' && k.0 != ' ' && k.2 != ' ').map(|(_, n)| n).sum();
    let per = |n: u64| n as f64 / total_letters.max(1) as f64 * 100.0;
    println!("  space presses: {spaces:>12}  ({:.1} per 100 letters; between two letters {:.1})", per(spaces), per(between));

    fn top<K: Ord + Clone>(map: &HashMap<K, u64>, pick: impl Fn(&K) -> bool) -> Vec<(K, u64)> {
      let mut v: Vec<(K, u64)> = map.iter().filter(|(k, _)| pick(k)).map(|(k, &n)| (k.clone(), n)).collect();
      v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
      v.truncate(20);
      v
    }
    for spaced in [false, true] {
      let label = if spaced { "with space" } else { "letters only" };
      println!("\n[Bigrams] top 20, {label}");
      for ((c1, c2), count) in top(&self.bigrams, |&(a, b)| (a == ' ' || b == ' ') == spaced) {
        println!("  {}{}: {count:>12}", show(c1), show(c2));
      }
      println!("\n[Trigrams] top 20, {label}");
      for ((c1, c2, c3), count) in top(&self.trigrams, |&(a, b, c)| (a == ' ' || b == ' ' || c == ' ') == spaced) {
        println!("  {}{}{}: {count:>12}", show(c1), show(c2), show(c3));
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn trim_breaks_ties_by_key() {
    // 빈도가 모두 같으면 90% 경계에서 어떤 항목이 남는지가 HashMap 순서에 좌우되면 안 됨
    for _ in 0..20 {
      let mut s = CorpusStats::new();
      for (i, c) in ('a'..='j').enumerate() { s.bigrams.insert((c, 'z'), 10); s.trigrams.insert((c, 'y', 'z'), 10 + (i as u64 % 2)); }
      s.trim_by_coverage(0.99, 0.90);
      let mut kept: Vec<char> = s.bigrams.keys().map(|k| k.0).collect();
      kept.sort_unstable();
      assert_eq!(kept, ('a'..='j').collect::<Vec<_>>()); // 99% → 10개 모두
      let mut tri: Vec<char> = s.trigrams.keys().map(|k| k.0).collect();
      tri.sort_unstable();
      // 빈도 11인 b,d,f,h,j 먼저, 그다음 빈도 10은 키 순 a,c,e,g
      assert_eq!(tri, vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j']);
    }
  }

  #[test]
  fn space_ngrams_do_not_change_letter_only_trimming() {
    let with = crate::test_util::stats();
    assert!(with.letters[&' '] > 0 && with.trigrams.keys().any(|k| k.1 == ' '));
    let letters_only = |s: &CorpusStats| {
      let mut o = s.clone();
      o.bigrams.retain(|k, _| k.0 != ' ' && k.1 != ' ');
      o.trigrams.retain(|k, _| k.0 != ' ' && k.1 != ' ' && k.2 != ' ');
      o
    };
    let mut a = with.clone();
    a.trim_by_coverage(0.95, 0.80);
    let mut b = letters_only(&with);
    b.trim_by_coverage(0.95, 0.80);
    assert_eq!(letters_only(&a), b);
    // 채점에 쓰는 "a _ c" 창은 자기들끼리만 잘림 (다른 공백 창이 있든 없든 같은 것이 남음)
    let cross_only = |s: &CorpusStats| { let mut o = s.clone(); o.bigrams.clear(); o.trigrams.retain(|k, _| k.1 == ' '); o };
    let mut c = cross_only(&with);
    c.trim_by_coverage(0.95, 0.80);
    assert_eq!(cross_only(&a), c);
    assert!(c.trigrams.len() < cross_only(&with).trigrams.len());
    assert!(a.trigrams.len() < with.trigrams.len());
  }

  #[test]
  fn trim_full_coverage_keeps_everything() {
    let mut s = crate::test_util::stats();
    let before = s.clone();
    s.trim_by_coverage(1.0, 1.0);
    assert_eq!(s, before);
    s.trim_by_coverage(0.99, 0.90);
    assert!(s.trigrams.len() < before.trigrams.len());
  }

  #[test]
  fn save_load_roundtrip_and_rejects_old_format() {
    let dir = std::env::temp_dir().join(format!("p2anut-corpus-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.bin");
    let path = path.to_str().unwrap();
    let s = crate::test_util::stats();
    s.save(path, 7).unwrap();
    let (loaded, ver) = CorpusStats::load(path).unwrap();
    assert_eq!((loaded, ver), (s.clone(), 7));
    // 헤더 없는 예전 형식(bincode만)은 거부
    std::fs::write(path, bincode::serialize(&s).unwrap()).unwrap();
    assert!(CorpusStats::load(path).unwrap_err().to_string().contains("old-format"));
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn fast_corpus_merge_and_vec_roundtrip() {
    let mut a = FastCorpus::new();
    let mut b = FastCorpus::new();
    crate::wiki::analyze_bytes(b"hello world", &mut a);
    crate::wiki::analyze_bytes(b"other text", &mut b);
    let mut both = FastCorpus::new();
    // 줄바꿈은 연속을 끊으므로 두 덩어리를 따로 센 것과 같음
    crate::wiki::analyze_bytes(b"hello world\nother text", &mut both);
    a.merge(&b);
    assert_eq!(a, both);
    let (l, bi, tri, sp) = a.to_vecs();
    assert_eq!(FastCorpus::from_vecs(&l, &bi, &tri, &sp).unwrap(), a);
    assert!(FastCorpus::from_vecs(&l[..3], &bi, &tri, &sp).is_err());
  }

  #[test]
  fn mixing_weighs_by_letters() {
    let mut a = FastCorpus::new();
    crate::wiki::analyze_bytes(&b"the cat sat on the mat\n".repeat(10), &mut a);
    let mut b = FastCorpus::new();
    crate::wiki::analyze_bytes(&b"fn zz qq\n".repeat(100), &mut b);
    let (a, b) = (a.into_stats(), b.into_stats());
    let letters = |s: &CorpusStats| s.letters.iter().filter(|(c, _)| c.is_ascii_lowercase()).map(|(_, &n)| n).sum::<u64>();
    // 0이면 그대로, 글자 수는 앞쪽에 맞춤 (반올림 오차만큼 다를 수 있음)
    assert_eq!(a.mixed(&b, 0.0), a);
    let m = a.mixed(&b, 0.5);
    assert!(letters(&m).abs_diff(letters(&a)) <= 3);
    // 절반은 b의 글자: z는 b 글자의 1/3
    let z = m.letters[&'z'] as f64 / letters(&m) as f64;
    assert!((z - 0.5 / 3.0).abs() < 0.02, "{z}");
    assert_eq!(m.bigrams[&('t', 'h')], a.bigrams[&('t', 'h')] / 2);
    let whole = a.mixed(&b, 1.0);
    assert!(!whole.letters.contains_key(&'t') && whole.letters[&'z'] > 0);
  }

  #[test]
  fn space_is_the_27th_key() {
    let stats = crate::test_util::stats();
    let g = stats.to_arrays();
    assert_eq!(g.keys.len(), 27);
    assert_eq!(g.keys[26], ' ');
    assert_eq!(g.freq[26], stats.letters[&' ']);
    // 글자끼리와 공백이 낀 n-gram이 모두 일반 n-gram으로 들어감
    assert_eq!(g.nz_bigrams.len(), stats.bigrams.len());
    assert_eq!(g.nz_trigrams.len(), stats.trigrams.len());
    assert!(g.nz_trigrams.iter().any(|&(a, b, c, _)| b == 26 && a < 26 && c < 26));
    assert!(g.bigram_offsets[27] > g.bigram_offsets[26]);
  }
}
