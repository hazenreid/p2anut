#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use std::collections::HashMap;
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
}

impl FastCorpus {
  pub fn new() -> Self {
    Self {
      letters: Box::new([0; CHARS]),
      bigrams: Box::new([0; CHARS * CHARS]),
      trigrams: vec![0u64; CHARS * CHARS * CHARS].into_boxed_slice(),
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
  }

  // 중간 저장용 (고정 길이 배열은 serde가 직접 못 다뤄서 Vec으로)
  pub fn to_vecs(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    (self.letters.to_vec(), self.bigrams.to_vec(), self.trigrams.to_vec())
  }

  pub fn from_vecs(letters: &[u64], bigrams: &[u64], trigrams: &[u64]) -> anyhow::Result<Self> {
    anyhow::ensure!(letters.len() == CHARS && bigrams.len() == CHARS * CHARS && trigrams.len() == CHARS * CHARS * CHARS,
      "corpus array sizes do not match");
    let mut f = Self::new();
    f.letters.copy_from_slice(letters);
    f.bigrams.copy_from_slice(bigrams);
    f.trigrams.copy_from_slice(trigrams);
    Ok(f)
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

  // include_space=false: 공백을 엄지 키로 보고 배열에서 뺌 (a-z 26키만 배치, 공백이 낀 빅/트라이그램은 버림)
  pub fn to_arrays(&self, include_space: bool) -> CorpusArrays {
    let keys: Vec<char> = ('a'..='z').chain(include_space.then_some(' ')).collect();

    let lu = |ch: char| -> Option<u8> {
      if ch.is_ascii_lowercase() { Some(ch as u8 - b'a') }
      else if ch == ' ' && include_space { Some(26) }
      else { None }
    };

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

    let mut bg_lists: Vec<Vec<usize>> = vec![Vec::new(); CHARS];
    for (i, &(a, b, _)) in nz_bigrams.iter().enumerate() {
      let (a, b) = (a as usize, b as usize);
      bg_lists[a].push(i);
      if b != a { bg_lists[b].push(i); }
    }
    let mut bigram_flat = Vec::new();
    let mut bigram_offsets = [0usize; CHARS + 1];
    for id in 0..CHARS {
      bigram_offsets[id] = bigram_flat.len();
      bigram_flat.extend_from_slice(&bg_lists[id]);
    }
    bigram_offsets[CHARS] = bigram_flat.len();

    let mut tg_lists: Vec<Vec<usize>> = vec![Vec::new(); CHARS];
    for (i, &(a, b, c, _)) in nz_trigrams.iter().enumerate() {
      let (a, b, c) = (a as usize, b as usize, c as usize);
      tg_lists[a].push(i);
      if b != a { tg_lists[b].push(i); }
      if c != a && c != b { tg_lists[c].push(i); }
    }
    let mut trigram_flat = Vec::new();
    let mut trigram_offsets = [0usize; CHARS + 1];
    for id in 0..CHARS {
      trigram_offsets[id] = trigram_flat.len();
      trigram_flat.extend_from_slice(&tg_lists[id]);
    }
    trigram_offsets[CHARS] = trigram_flat.len();

    CorpusArrays { keys, freq, nz_bigrams, nz_trigrams, bigram_flat, bigram_offsets, trigram_flat, trigram_offsets }
  }

  // 빈도 상위부터 누적 비율이 bigram/trigram 기준에 닿을 때까지만 보존 (1.0 = 전부)
  // 최적화 속도를 위해 불러올 때 적용. 파일에는 전체 통계를 저장해서 기준을 바꿔도 재분석이 필요 없음.
  // 공백 없는 것과 공백이 낀 것은 따로 자름: 빈도가 큰 공백 n-gram이 섞여도 글자끼리의 n-gram은 예전과 똑같이 남음
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
    fn trim_split<K: Eq + std::hash::Hash + Clone + Ord>(map: &mut HashMap<K, u64>, threshold: f64, has_space: impl Fn(&K) -> bool) {
      let (mut spaced, mut plain): (HashMap<K, u64>, HashMap<K, u64>) = map.drain().partition(|(k, _)| has_space(k));
      trim(&mut plain, threshold);
      trim(&mut spaced, threshold);
      plain.extend(spaced);
      *map = plain;
    }
    trim_split(&mut self.bigrams, bigram, |&(a, b)| a == ' ' || b == ' ');
    trim_split(&mut self.trigrams, trigram, |&(a, b, c)| a == ' ' || b == ' ' || c == ' ');
  }

  // 파일 형식: CORPUS_MAGIC + 파일 형식 버전(u32) + 파서 버전(u32) + bincode(CorpusStats)
  pub fn save(&self, path: &str, parser_version: u32) -> anyhow::Result<()> {
    let mut bytes = CORPUS_MAGIC.to_vec();
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
    let bytes = std::fs::read(path)?;
    let header = CORPUS_MAGIC.len() + 8;
    if bytes.len() < header || !bytes.starts_with(CORPUS_MAGIC) {
      anyhow::bail!("{path} is an old-format corpus (trimmed at save time); rebuild it with `analyze --reset`");
    }
    let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().expect("4 bytes"));
    let format = word(CORPUS_MAGIC.len());
    anyhow::ensure!(format == CORPUS_FORMAT, "{path} has corpus format {format}, expected {CORPUS_FORMAT}; rebuild it with `analyze --reset`");
    let parser_version = word(CORPUS_MAGIC.len() + 4);
    Ok((bincode::deserialize(&bytes[header..])?, parser_version))
  }

  // 글자 비율은 a-z끼리, 공백(엄지)은 글자 100개당 횟수로 따로. n-gram은 공백 없는 것과 공백이 낀 것을 따로 (공백은 _로 표시)
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
    println!("  space between letters (thumb): {spaces:>12}  ({:.1} per 100 letters)", spaces as f64 / total_letters.max(1) as f64 * 100.0);

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
    // 공백이 낀 것도 자기들끼리 같은 기준으로 잘림
    assert!(a.trigrams.keys().any(|k| k.1 == ' '));
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
    crate::analyze_bytes(b"hello world", &mut a);
    crate::analyze_bytes(b"other text", &mut b);
    let mut both = FastCorpus::new();
    // 줄바꿈은 연속을 끊으므로 두 덩어리를 따로 센 것과 같음
    crate::analyze_bytes(b"hello world\nother text", &mut both);
    a.merge(&b);
    assert_eq!(a, both);
    let (l, bi, tri) = a.to_vecs();
    assert_eq!(FastCorpus::from_vecs(&l, &bi, &tri).unwrap(), a);
    assert!(FastCorpus::from_vecs(&l[..3], &bi, &tri).is_err());
  }

  #[test]
  fn to_arrays_can_drop_space() {
    let stats = crate::test_util::stats();
    let with = stats.to_arrays(true);
    let without = stats.to_arrays(false);
    assert_eq!(with.keys.len(), 27);
    assert_eq!(without.keys.len(), 26);
    assert!(!without.keys.contains(&' '));
    assert_eq!(without.freq[26], 0);
    assert!(without.nz_bigrams.iter().all(|&(a, b, _)| a < 26 && b < 26));
    assert!(without.nz_trigrams.iter().all(|&(a, b, c, _)| a < 26 && b < 26 && c < 26));
    // 공백 없는 n-gram은 그대로 남아야 함
    let letters_only = with.nz_bigrams.iter().filter(|&&(a, b, _)| a < 26 && b < 26).count();
    assert_eq!(without.nz_bigrams.len(), letters_only);
    assert_eq!(without.bigram_offsets[26], without.bigram_offsets[27]);
  }
}
