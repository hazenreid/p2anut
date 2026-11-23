// 심볼 레이어용 코드 코퍼스: 키는 a–z(0..26), 공백(26), ASCII 기호 32개(27..59). 주석을 뺀 러스트 코드를 셈:
// - 글자는 소문자로 (Shift는 따로 세지 않음), 기호는 그대로 한 키
// - 공백은 두 키 사이의 공백 하나만 셈. 연속 공백, 줄 처음·끝의 공백(들여쓰기 포함)은 세지 않고 연속을 끊음
// - 숫자, 탭, 줄바꿈, 그 밖의 문자는 연속을 끊음
// - 같은 키 연속은 그 쌍이 낀 트라이그램을 세지 않음 (글자 코퍼스와 같음). 빅그램도 글자는 세지 않지만, 기호는 셈
//   (::, //, )) 등: 원샷 레이어 키를 누른 채로 이어 치는지 알려면 필요)
#![allow(clippy::cast_possible_truncation)]
use crate::corpus::CorpusStats;

pub const SYMBOLS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
pub const NSYM: usize = 32;
pub const SPACE: u8 = 26;
pub const FIRST_SYM: u8 = 27;
pub const KEYS: usize = FIRST_SYM as usize + NSYM;

// 코드 코퍼스 파일의 머리표와 세는 규칙의 버전 (규칙을 바꾸면 올림)
pub const CODE_MAGIC: &[u8] = b"P2ANUT-CODE\0";
// 2: 크레이트마다 본체 코드를 모두 읽고 같은 무게로 더함
// 3: 기호의 같은 키 연속(::)도 빅그램으로 셈
pub const CODE_VERSION: u32 = 3;

// 기호의 번호 (0..32)
pub fn sym_index(ch: char) -> Option<usize> { SYMBOLS.find(ch).filter(|_| ch.is_ascii()) }

// 바이트 하나의 키 번호 (글자는 대소문자 무관). 공백과 그 밖의 문자는 None
fn key_of(b: u8) -> Option<u8> {
  if b.is_ascii_alphabetic() { return Some(b.to_ascii_lowercase() - b'a'); }
  SYMBOLS.as_bytes().iter().position(|&s| s == b).map(|i| FIRST_SYM + i as u8)
}

// 키 번호 → 문자 (공백은 ' ')
pub fn key_char(id: u8) -> char {
  match id {
    0..=25 => (b'a' + id) as char,
    SPACE => ' ',
    _ => SYMBOLS.as_bytes()[(id - FIRST_SYM) as usize] as char,
  }
}

// 문자 → 키 번호
pub fn char_key(ch: char) -> Option<u8> {
  if ch == ' ' { Some(SPACE) } else if ch.is_ascii() { key_of(ch as u8) } else { None }
}

#[derive(Clone, PartialEq, Debug)]
pub struct KeyCorpus {
  pub unigrams: Box<[u64]>,
  pub bigrams: Box<[u64]>,
  pub trigrams: Box<[u64]>,
}

impl KeyCorpus {
  pub fn new() -> Self {
    Self {
      unigrams: vec![0; KEYS].into_boxed_slice(),
      bigrams: vec![0; KEYS * KEYS].into_boxed_slice(),
      trigrams: vec![0; KEYS * KEYS * KEYS].into_boxed_slice(),
    }
  }

  pub fn merge(&mut self, other: &KeyCorpus) {
    for (a, b) in self.unigrams.iter_mut().zip(other.unigrams.iter()) { *a += b; }
    for (a, b) in self.bigrams.iter_mut().zip(other.bigrams.iter()) { *a += b; }
    for (a, b) in self.trigrams.iter_mut().zip(other.trigrams.iter()) { *a += b; }
  }

  pub fn into_stats(self) -> CorpusStats {
    let mut s = CorpusStats::new();
    for (i, &n) in self.unigrams.iter().enumerate() { if n > 0 { s.letters.insert(key_char(i as u8), n); } }
    for (i, &n) in self.bigrams.iter().enumerate() {
      if n > 0 { s.bigrams.insert((key_char((i / KEYS) as u8), key_char((i % KEYS) as u8)), n); }
    }
    for (i, &n) in self.trigrams.iter().enumerate() {
      if n > 0 {
        s.trigrams.insert((key_char((i / (KEYS * KEYS)) as u8), key_char((i / KEYS % KEYS) as u8), key_char((i % KEYS) as u8)), n);
      }
    }
    s
  }
}

// 키 연속을 이어 가며 셈
struct Chain<'a> {
  c: &'a mut KeyCorpus,
  hist: [u8; 3],
  len: usize,
}

impl Chain<'_> {
  fn commit(&mut self, id: u8) {
    self.c.unigrams[id as usize] += 1;
    self.hist = [self.hist[1], self.hist[2], id];
    self.len += 1;
    let h = self.hist.map(usize::from);
    if self.len >= 2 && (h[1] != h[2] || h[2] >= FIRST_SYM as usize) { self.c.bigrams[h[1] * KEYS + h[2]] += 1; }
    if self.len >= 3 && h[1] != h[2] && h[0] != h[1] { self.c.trigrams[(h[0] * KEYS + h[1]) * KEYS + h[2]] += 1; }
  }

  fn reset(&mut self) { self.len = 0; self.hist = [255; 3]; }
}

// 주석을 뺀 코드를 키 연속으로 세어 더함
pub fn count_text(bytes: &[u8], corpus: &mut KeyCorpus) {
  let mut c = Chain { c: corpus, hist: [255; 3], len: 0 };
  let mut spaces = 0usize;
  let mut i = 0;
  while i < bytes.len() {
    let b = bytes[i];
    if b == b' ' { spaces += 1; i += 1; continue; }
    i += 1;
    let Some(k) = key_of(b) else { c.reset(); spaces = 0; continue; };
    // 공백 하나는 앞뒤 키 사이의 누름. 줄 처음(앞에 이어지는 키가 없음)이나 연속 공백은 세지 않고 끊음
    if spaces == 1 && c.len > 0 { c.commit(SPACE); } else if spaces > 0 { c.reset(); }
    spaces = 0;
    c.commit(k);
  }
}

// 기호 빈도 요약 (글자 100개당): 기호별 누름과 기호가 낀 상위 바이그램
pub fn print_summary(stats: &CorpusStats) {
  let letters: u64 = stats.letters.iter().filter(|(c, _)| c.is_ascii_lowercase()).map(|(_, n)| n).sum();
  #[allow(clippy::cast_precision_loss)]
  let per = |n: u64| n as f64 * 100.0 / letters.max(1) as f64;
  let sym = |c: char| sym_index(c).is_some();
  let mut syms: Vec<(char, u64)> = stats.letters.iter().filter(|(c, _)| sym(**c)).map(|(&c, &n)| (c, n)).collect();
  syms.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
  let total: u64 = syms.iter().map(|e| e.1).sum();
  println!("  symbols per 100 letters: {:.1} (space {:.1})", per(total), per(stats.letters.get(&' ').copied().unwrap_or(0)));
  println!("  {}", syms.iter().map(|(c, n)| format!("{c} {:.2}", per(*n))).collect::<Vec<_>>().join("  "));
  let show = |c: char| if c == ' ' { '\u{2423}' } else { c };
  let mut bi: Vec<((char, char), u64)> = stats.bigrams.iter().filter(|((a, b), _)| sym(*a) || sym(*b)).map(|(&k, &n)| (k, n)).collect();
  bi.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
  println!("  top bigrams: {}", bi.iter().take(20).map(|((a, b), n)| format!("{}{} {:.2}", show(*a), show(*b), per(*n))).collect::<Vec<_>>().join("  "));
}

#[cfg(test)]
mod tests {
  use super::*;

  fn run(text: &str) -> KeyCorpus { let mut c = KeyCorpus::new(); count_text(text.as_bytes(), &mut c); c }
  fn id(ch: char) -> usize { char_key(ch).unwrap() as usize }
  fn bi(c: &KeyCorpus, s: &str) -> u64 { let v: Vec<char> = s.chars().collect(); c.bigrams[id(v[0]) * KEYS + id(v[1])] }
  fn tri(c: &KeyCorpus, s: &str) -> u64 {
    let v: Vec<char> = s.chars().collect();
    c.trigrams[(id(v[0]) * KEYS + id(v[1])) * KEYS + id(v[2])]
  }

  #[test]
  fn keys_cover_letters_space_and_32_symbols() {
    assert_eq!(SYMBOLS.len(), NSYM);
    assert_eq!(KEYS, 59);
    for id in 0..KEYS as u8 { assert_eq!(char_key(key_char(id)), Some(id)); }
    assert_eq!((char_key('A'), char_key('1'), char_key('é')), (Some(0), None, None));
    assert_eq!(sym_index('('), Some(7));
    assert_eq!(sym_index('a'), None);
  }

  #[test]
  fn symbols_and_the_spaces_between_keys_are_counted() {
    let c = run("fn f<'a>(x: &'a str) -> Vec<u8> { x.len(); }");
    // "u8> {"의 숫자에서 끊긴 뒤에도 "> "는 셈
    assert_eq!((bi(&c, "f<"), bi(&c, "<'"), bi(&c, "'a"), bi(&c, ": "), bi(&c, " &"), bi(&c, "->"), bi(&c, "> "), bi(&c, "()")), (1, 1, 2, 1, 1, 1, 2, 1));
    assert_eq!((tri(&c, "();"), tri(&c, "x.l"), tri(&c, " ->"), bi(&c, "u>")), (1, 1, 1, 0));
    assert_eq!(c.unigrams[id('\'')], 2);
  }

  #[test]
  fn indentation_double_spaces_digits_and_line_ends_break() {
    let c = run("    let x = y;\n  a  b 1 c\td\u{e9}e");
    // 들여쓰기는 세지 않고, 줄 안의 공백 하나는 셈
    assert_eq!((bi(&c, "t "), bi(&c, " x"), bi(&c, "= "), bi(&c, "y;")), (1, 1, 1, 1));
    assert_eq!((bi(&c, ";a"), bi(&c, "a "), bi(&c, " b"), bi(&c, "b "), bi(&c, " c"), bi(&c, "cd"), bi(&c, "de")), (0, 0, 0, 0, 0, 0, 0));
    assert_eq!(c.unigrams[SPACE as usize], 3);
  }

  #[test]
  fn repeated_keys_follow_the_letter_corpus_rule() {
    // 같은 기호 연속(::)은 빅그램으로 세고, 그 쌍이 낀 트라이그램은 세지 않음
    let c = run("a::b");
    assert_eq!((bi(&c, "::"), tri(&c, "a::"), tri(&c, "::b"), bi(&c, "a:"), bi(&c, ":b")), (1, 0, 0, 1, 1));
    assert_eq!(c.unigrams[id(':')], 2);
    // 같은 글자 연속(ll)은 글자 코퍼스처럼 세지 않음
    let c = run("all in");
    assert_eq!((bi(&c, "ll"), tri(&c, "all"), bi(&c, "l "), tri(&c, "l i")), (0, 0, 1, 1));
  }

  #[test]
  fn merge_and_stats() {
    let (a, b) = (run("x = (y);"), run("it's ok"));
    let mut m = a.clone();
    m.merge(&b);
    let mut both = KeyCorpus::new();
    count_text(b"x = (y);\nit's ok", &mut both);
    assert_eq!(m, both);
    let s = m.into_stats();
    assert_eq!((s.letters[&'('], s.bigrams[&('(', 'y')], s.trigrams[&('t', '\'', 's')]), (1, 1, 1));
  }
}
