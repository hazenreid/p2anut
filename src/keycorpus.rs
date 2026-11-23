// 심볼 레이어용 코퍼스: 키는 a–z(0..26), 공백(26), ASCII 기호 32개(27..59).
// 위키 본문(영어 문장부호)과 러스트 코드(코드 코퍼스)를 같은 규칙으로 셈:
// - 글자는 소문자로 (Shift는 따로 세지 않음), 기호는 그대로 한 키
// - 공백은 두 키 사이의 공백 하나만 셈. 연속 공백, 줄 처음·끝의 공백(들여쓰기 포함)은 세지 않고 연속을 끊음
// - 숫자, 탭, 줄바꿈, 그 밖의 문자는 연속을 끊음. 둥근 따옴표·대시(‘ ’ “ ” – —)는 ASCII(' " -)로 바꿔 셈
// - 같은 키 연속은 빅그램으로 세지 않고 그 쌍이 낀 트라이그램도 세지 않음 (글자 코퍼스와 같음)
// - 영어 본문(count_prose)에서 '는 축약형의 아포스트로피일 때만 키로 셈 (is_contraction). 따옴표로 쓴 '(‘term’),
//   소유격(John's, students'), 이름(O'Brien)은 세지 않고 연속을 끊음. 코드(count_code)의 '는 수명·문자 리터럴이라 모두 셈
#![allow(clippy::cast_possible_truncation)]
use crate::corpus::CorpusStats;

pub const SYMBOLS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
pub const NSYM: usize = 32;
pub const SPACE: u8 = 26;
pub const FIRST_SYM: u8 = 27;
pub const KEYS: usize = FIRST_SYM as usize + NSYM;

// 영어 문장부호 코퍼스와 코드 코퍼스 파일의 머리표와 세는 규칙의 버전 (규칙을 바꾸면 올림)
pub const ENGLISH_MAGIC: &[u8] = b"P2ANUT-ENGLISH-PUNCT\0";
// 2: '는 축약형일 때만 셈
pub const ENGLISH_VERSION: u32 = 2;
pub const CODE_MAGIC: &[u8] = b"P2ANUT-CODE\0";
pub const CODE_VERSION: u32 = 1;

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

  // 중간 저장용
  pub fn to_vecs(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    (self.unigrams.to_vec(), self.bigrams.to_vec(), self.trigrams.to_vec())
  }

  pub fn from_vecs(u: &[u64], b: &[u64], t: &[u64]) -> anyhow::Result<Self> {
    anyhow::ensure!(u.len() == KEYS && b.len() == KEYS * KEYS && t.len() == KEYS * KEYS * KEYS, "punctuation corpus array sizes do not match");
    Ok(Self { unigrams: u.into(), bigrams: b.into(), trigrams: t.into() })
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
    if self.len >= 2 && h[1] != h[2] {
      self.c.bigrams[h[1] * KEYS + h[2]] += 1;
      if self.len >= 3 && h[0] != h[1] { self.c.trigrams[(h[0] * KEYS + h[1]) * KEYS + h[2]] += 1; }
    }
  }

  fn reset(&mut self) { self.len = 0; self.hist = [255; 3]; }
}

// 둥근 따옴표·대시의 UTF-8 (E2 80 xx) → ASCII
fn typographic(rest: &[u8]) -> Option<u8> {
  if rest.len() < 3 || rest[0] != 0xE2 || rest[1] != 0x80 { return None; }
  match rest[2] {
    0x98 | 0x99 => Some(b'\''),
    0x9C | 0x9D => Some(b'"'),
    0x93 | 0x94 => Some(b'-'),
    _ => None,
  }
}

// at의 '(너비 width 바이트)가 축약형의 아포스트로피인지: 앞뒤가 글자이고
// n't, 대명사 + 're/'ve/'ll/'d, I'm, it·he·she·that·what·there 등 + 's 중 하나.
// 's는 소유격과 모양이 같아서 축약형이 분명한 앞말일 때만 셈 (John's는 소유격으로 봄)
pub fn is_contraction(bytes: &[u8], at: usize, width: usize) -> bool {
  let start = bytes[..at].iter().rposition(|b| !b.is_ascii_alphabetic()).map_or(0, |p| p + 1);
  let tail = at + width;
  let end = bytes[tail..].iter().position(|b| !b.is_ascii_alphabetic()).map_or(bytes.len(), |p| tail + p);
  let (word, suffix) = (bytes[start..at].to_ascii_lowercase(), bytes[tail..end].to_ascii_lowercase());
  if word.is_empty() { return false; }
  let w = word.as_slice();
  match suffix.as_slice() {
    b"t" => w.ends_with(b"n"),
    b"re" => matches!(w, b"you" | b"we" | b"they" | b"who" | b"what" | b"there" | b"here" | b"where"),
    b"ve" => matches!(w, b"i" | b"you" | b"we" | b"they" | b"who" | b"what" | b"could" | b"would" | b"should" | b"might" | b"must"),
    b"ll" => matches!(w, b"i" | b"you" | b"he" | b"she" | b"it" | b"we" | b"they" | b"that" | b"there" | b"who" | b"what" | b"this"),
    b"d" => matches!(w, b"i" | b"you" | b"he" | b"she" | b"it" | b"we" | b"they" | b"that" | b"there" | b"who" | b"what"),
    b"m" => w == b"i",
    b"s" => matches!(w, b"it" | b"he" | b"she" | b"that" | b"what" | b"there" | b"here" | b"who" | b"where" | b"how" | b"let"
      | b"when" | b"why" | b"this" | b"everyone" | b"everybody" | b"someone" | b"somebody" | b"nobody" | b"nothing" | b"something" | b"everything"),
    _ => false,
  }
}

// 주석을 뺀 코드를 키 연속으로 세어 더함 (')는 모두 셈)
pub fn count_code(bytes: &[u8], corpus: &mut KeyCorpus) { count(bytes, corpus, false); }

// 영어 본문을 키 연속으로 세어 더함 (')는 축약형일 때만)
pub fn count_prose(bytes: &[u8], corpus: &mut KeyCorpus) { count(bytes, corpus, true); }

const APOSTROPHE: u8 = FIRST_SYM + 6;

fn count(bytes: &[u8], corpus: &mut KeyCorpus, prose: bool) {
  let mut c = Chain { c: corpus, hist: [255; 3], len: 0 };
  let mut spaces = 0usize;
  let mut i = 0;
  while i < bytes.len() {
    let b = bytes[i];
    if b == b' ' { spaces += 1; i += 1; continue; }
    let (key, width) = match typographic(&bytes[i..]) {
      Some(a) => (key_of(a), 3),
      None => (key_of(b), 1),
    };
    let at = i;
    i += width;
    let key = key.filter(|&k| !(prose && k == APOSTROPHE && !is_contraction(bytes, at, width)));
    let Some(k) = key else { c.reset(); spaces = 0; continue; };
    // 공백 하나는 앞뒤 키 사이의 누름. 줄 처음(앞에 이어지는 키가 없음)이나 연속 공백은 세지 않고 끊음
    if spaces == 1 && c.len > 0 { c.commit(SPACE); } else if spaces > 0 { c.reset(); }
    spaces = 0;
    c.commit(k);
  }
}

// 기호 빈도 요약 (글자 100개당): 기호별 누름과 기호가 낀 상위 바이그램. only가 있으면 그 기호만 (나머지가 낀 것은 뺌)
pub fn print_summary(stats: &CorpusStats, only: Option<&str>) {
  let letters: u64 = stats.letters.iter().filter(|(c, _)| c.is_ascii_lowercase()).map(|(_, n)| n).sum();
  #[allow(clippy::cast_precision_loss)]
  let per = |n: u64| n as f64 * 100.0 / letters.max(1) as f64;
  let sym = |c: char| sym_index(c).is_some();
  let keep = |c: char| sym(c) && only.is_none_or(|o| o.contains(c));
  let mut syms: Vec<(char, u64)> = stats.letters.iter().filter(|(c, _)| keep(**c)).map(|(&c, &n)| (c, n)).collect();
  syms.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
  let total: u64 = syms.iter().map(|e| e.1).sum();
  println!("  symbols per 100 letters: {:.1} (space {:.1})", per(total), per(stats.letters.get(&' ').copied().unwrap_or(0)));
  println!("  {}", syms.iter().map(|(c, n)| format!("{c} {:.2}", per(*n))).collect::<Vec<_>>().join("  "));
  let show = |c: char| if c == ' ' { '\u{2423}' } else { c };
  let mut bi: Vec<((char, char), u64)> = stats.bigrams.iter()
    .filter(|((a, b), _)| (keep(*a) || keep(*b)) && !(sym(*a) && !keep(*a)) && !(sym(*b) && !keep(*b)))
    .map(|(&k, &n)| (k, n)).collect();
  bi.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
  println!("  top bigrams: {}", bi.iter().take(20).map(|((a, b), n)| format!("{}{} {:.2}", show(*a), show(*b), per(*n))).collect::<Vec<_>>().join("  "));
}

#[cfg(test)]
mod tests {
  use super::*;

  fn run(text: &str) -> KeyCorpus { let mut c = KeyCorpus::new(); count_prose(text.as_bytes(), &mut c); c }
  fn run_code(text: &str) -> KeyCorpus { let mut c = KeyCorpus::new(); count_code(text.as_bytes(), &mut c); c }
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
    assert_eq!(key_char(APOSTROPHE), '\'');
    assert_eq!(sym_index('a'), None);
  }

  #[test]
  fn contractions_and_sentence_punctuation_are_counted() {
    let c = run("It's done. Don't stop, ok?");
    assert_eq!((bi(&c, "t'"), bi(&c, "n'"), bi(&c, "'s"), tri(&c, "n't"), tri(&c, "t's")), (1, 1, 1, 1, 1));
    // 문장 끝 부호 뒤 공백, 쉼표 뒤 공백도 셈 (글자 코퍼스와 달리)
    assert_eq!((bi(&c, ". "), tri(&c, "e. "), tri(&c, ". d"), bi(&c, ", "), bi(&c, "k?")), (1, 1, 1, 1, 1));
    assert_eq!(c.unigrams[SPACE as usize], 4);
  }

  #[test]
  fn only_contraction_apostrophes_count_in_prose() {
    let c = run("It's done. Don't stop, we're fine; I'm ok, you'll see what's here. They'd go, can't and won't.");
    assert_eq!(c.unigrams[id('\'')], 9);
    assert_eq!((tri(&c, "n't"), bi(&c, "'r"), bi(&c, "'m"), bi(&c, "'l"), bi(&c, "'d")), (3, 1, 1, 1, 1));
    // 따옴표, 소유격, 이름, 'n'은 세지 않고 끊음 (앞뒤 글자도 이어지지 않음)
    let c = run("The 'term' is John's, the students' books, O'Brien, rock 'n' roll, \u{2018}quoted\u{2019} and it\u{2019}s");
    assert_eq!(c.unigrams[id('\'')], 1);
    assert_eq!((bi(&c, "n'"), bi(&c, "ns"), bi(&c, " t"), bi(&c, "t'"), bi(&c, "'s")), (0, 0, 1, 1, 1));
    // 코드에서는 모두 셈 (수명, 문자 리터럴)
    let c = run_code("fn f<'a>(x: &'a str) -> char { 'x' }");
    assert_eq!(c.unigrams[id('\'')], 4);
    assert!(is_contraction(b"don't", 3, 1) && is_contraction(b"I'm", 1, 1) && !is_contraction(b"'em", 0, 1) && !is_contraction(b"Ka'b", 2, 1));
  }

  #[test]
  fn typographic_quotes_and_dashes_become_ascii() {
    let c = run("it\u{2019}s \u{201c}quoted\u{201d} well\u{2014}known");
    assert_eq!((bi(&c, "'s"), bi(&c, " \""), bi(&c, "\"q"), bi(&c, "d\""), bi(&c, "l-"), bi(&c, "-k")), (1, 1, 1, 1, 1, 1));
    // 다른 비ASCII 문자는 끊음
    let c = run("caf\u{e9} au");
    assert_eq!((bi(&c, "af"), bi(&c, "f "), bi(&c, " a")), (1, 0, 0));
  }

  #[test]
  fn indentation_double_spaces_digits_and_line_ends_break() {
    let c = run_code("    let x = y;\n  a  b 1 c\td");
    // 들여쓰기는 세지 않고, 줄 안의 공백 하나는 셈
    assert_eq!((bi(&c, "t "), bi(&c, " x"), bi(&c, "= "), bi(&c, "y;")), (1, 1, 1, 1));
    assert_eq!((bi(&c, ";a"), bi(&c, "a "), bi(&c, " b"), bi(&c, "b "), bi(&c, " c"), bi(&c, "cd")), (0, 0, 0, 0, 0, 0));
    assert_eq!(c.unigrams[SPACE as usize], 3);
  }

  #[test]
  fn repeated_keys_follow_the_letter_corpus_rule() {
    let c = run_code("a::b");
    assert_eq!((bi(&c, "::"), tri(&c, "a::"), tri(&c, "::b"), bi(&c, "a:"), bi(&c, ":b")), (0, 0, 0, 1, 1));
    assert_eq!(c.unigrams[id(':')], 2);
  }

  #[test]
  fn merge_vecs_and_stats_roundtrip() {
    let (a, b) = (run("x = (y);"), run("it's ok"));
    let mut m = a.clone();
    m.merge(&b);
    let mut both = KeyCorpus::new();
    count_prose(b"x = (y);\nit's ok", &mut both);
    assert_eq!(m, both);
    let (u, bi, t) = m.to_vecs();
    assert_eq!(KeyCorpus::from_vecs(&u, &bi, &t).unwrap(), m);
    assert!(KeyCorpus::from_vecs(&u[1..], &bi, &t).is_err());
    let s = m.into_stats();
    assert_eq!((s.letters[&'('], s.bigrams[&('(', 'y')], s.trigrams[&('t', '\'', 's')]), (1, 1, 1));
  }
}
