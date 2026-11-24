// 심볼 레이어 최적화 (symbols)와 채점 (symbols-score).
//
// 베이스 배열(저장된 결과나 배열 파일)은 고정하고, ASCII 기호 32개를 원샷 심볼 레이어 30칸과 베이스의 빈칸에 배치함.
// 채점은 배열 최적화와 같은 점수 모델(노력값, SFB·스킵그램, 시저스·LSB, 롤·리다이렉트, [costs] 단위)에
// 러스트 코드 코퍼스(analyze-code)의 n-gram을 씀. 비용은 코드의 글자 100개당이고 기호가 낀 n-gram만 셈
// (글자·공백끼리는 베이스가 고정이라 어느 배치에서나 같음).
//
// 원샷 레이어: 심볼 레이어 기호를 치기 전에 엄지로 레이어 키를 누름(탭). 심볼 레이어 기호가 이어지면(::, ();, ->)
// 레이어 키를 누른 채로 이어 치므로 탭은 처음 한 번만 (원샷 키를 누르고 있으면 보통 레이어처럼 동작하는 펌웨어:
// QMK OSL, ZMK sticky layer). 엄지가 새로 끼는 두 손가락 누름은 사이에 다른 손가락이 끼는 경우와 같게 봄:
// 같은 손가락이면 스킵그램(sfs), 인접 손가락 두 줄 차이면 스킵 시저스만 셈 (롤·SFB·시저스 없음). 이어 치는 기호끼리와
// 기호 다음 키는 엄지 없이 바로 이어지므로 보통 바이그램·트라이그램. 엄지가 새로 끼는 트라이그램은 롤·리다이렉트로
// 치지 않음. 탭 한 번의 비용은 노력값 표에서 가장 작은 값 (× [costs] effort).
// 베이스 빈칸의 기호는 탭이 없는 대신 노력값에 base_slot_extra를 더함 (탭 비용보다 크면 심볼 레이어가 모자랄 때만 씀).
//
// 짝: pairs(순서 있음)는 같은 줄에만. 한 손 안이면 떨어져 있어도 되고 첫 글자가 왼쪽, 양손에 나뉘면 좌우 대칭(같은 손가락)에
// 첫 글자가 왼손. loose_pairs(순서 없음)는 거기에 더해 한 손 같은 열(떨어져 있어도 됨)도 되고 어느 쪽이 먼저든.
// 짝은 심볼 레이어 안에서만.
// 짝 맞춤 (외우기 쉽게): 순서 있는 짝과 없는 짝을 가리지 않고 모양이 같은 짝끼리 자리를 맞춤 (오른손은 거울 자리로 봄).
// 한 손 가로 짝은 모두 같은 두 열 (양쪽으로 한 칸씩 넓은 짝은 같은 줄에서 그 짝을 감쌀 때만), 한 손 세로 짝은 모두 같은
// 두 줄, 양손 대칭 짝은 모두 같은 열이거나 모두 같은 줄 (꺾인 모양은 바로 옆끼리 이어질 때만). 담금질은 어긴 쌍마다
// GROUP_PENALTY를 더한 채 거쳐 갈 수 있고, 결과는 지키는 배치에서만 고름.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::collections::HashMap;
use std::path::Path;
use anyhow::Context;
use rayon::prelude::*;
use crate::config::{Config, SaParams, SymbolsConfig};
use crate::corpus::CorpusStats;
use crate::keycorpus::{self, FIRST_SYM, KEYS, NSYM, SPACE};
use crate::layout::{Layout, CHARS, COLS, EMPTY, ROWS};
use crate::precomp::{finger, Precomp};

const N: usize = ROWS * COLS;
pub const EMPTY_TOKEN: char = '·';
pub const SPACE_TOKEN: char = '␣';
// 담금질 온도 (글자 100개당 비용 단위): 처음에는 이 정도 악화를 자주 받아들이고 끝에는 거의 안 받아들임
const T_START: f64 = 3.0;
const T_END: f64 = 0.0005;
// 짝 맞춤 규칙을 어긴 쌍 하나의 벌점 (글자 100개당 비용 단위). 담금질 초반에는 어긴 배치를 거쳐 갈 수 있고 끝에는 거의 못 감.
// 결과는 규칙을 지키는 배치에서만 고르므로 벌점은 탐색에만 씀
const GROUP_PENALTY: f64 = 5.0;

// 30칸 물리 위치의 비용 표 (비용 단위: 검지 홈 행 키 한 번 = 1). 배열 최적화의 Precomp에서 그대로 가져옴
pub struct Tables {
  eff: [f64; N],
  sfb: Vec<f64>,
  sc: Vec<f64>,
  roll: Vec<f64>,
  sfs: Vec<f64>,
  ssc: Vec<f64>,
  roll3: Vec<f64>,
  xsfs: Vec<f64>,
  xssc: Vec<f64>,
  bg: Vec<f64>,
  tg: Vec<f64>,
  x: Vec<f64>,
}

impl Tables {
  pub fn new(config: &Config) -> Self {
    let precomp = Precomp::new(&[0; CHARS], &config.weights, &config.costs, &config.effort, 0.0);
    let (a_sfb, a_sc) = crate::layout::event_units(&precomp, &config.costs);
    let grid = config.effort.grid();
    let mut t = Tables {
      eff: [0.0; N], sfb: vec![0.0; N * N], sc: vec![0.0; N * N], roll: vec![0.0; N * N],
      sfs: vec![0.0; N * N * N], ssc: vec![0.0; N * N * N], roll3: vec![0.0; N * N * N],
      xsfs: vec![0.0; N * N], xssc: vec![0.0; N * N], bg: vec![0.0; N * N], tg: vec![0.0; N * N * N], x: vec![0.0; N * N],
    };
    for p in 0..N { t.eff[p] = f64::from(grid[p / COLS][p % COLS]) * f64::from(config.costs.effort); }
    for p1 in 0..N {
      for p2 in 0..N {
        let i = p1 * N + p2;
        let e = precomp.bigram(p1, p2);
        t.sfb[i] = f64::from(e.sfb_base) * a_sfb;
        t.sc[i] = f64::from(e.scissors_base) * a_sc;
        t.roll[i] = f64::from(e.rolling_base);
        t.bg[i] = t.sfb[i] + t.sc[i] - t.roll[i];
        let c = precomp.cross(p1, p2);
        t.xsfs[i] = f64::from(c.sfs_base) * a_sfb;
        t.xssc[i] = f64::from(c.skip_scissors_base) * a_sc;
        t.x[i] = t.xsfs[i] + t.xssc[i];
        for p3 in 0..N {
          let j = i * N + p3;
          let e = precomp.trigram(p1, p2, p3);
          t.sfs[j] = f64::from(e.sfs_base) * a_sfb;
          t.ssc[j] = f64::from(e.skip_scissors_base) * a_sc;
          t.roll3[j] = f64::from(e.roll_base);
          t.tg[j] = t.sfs[j] + t.ssc[j] - t.roll3[j];
        }
      }
    }
    t
  }
}

// 코드 코퍼스의 n-gram (글자 100개당 횟수). 기호가 하나라도 낀 것만
pub struct Ngrams {
  uni: [f64; KEYS],
  bi: Vec<(u8, u8, f64)>,
  tri: Vec<(u8, u8, u8, f64)>,
  bi_of: Vec<Vec<u32>>,
  tri_of: Vec<Vec<u32>>,
}

fn is_sym(id: u8) -> bool { id >= FIRST_SYM }

// 빈도 상위부터 누적 비율이 coverage에 닿을 때까지 남김 (동률은 키 순)
fn trim<K: Ord + Copy>(v: &mut Vec<(K, f64)>, coverage: f64) {
  v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
  if coverage >= 1.0 { return; }
  let total: f64 = v.iter().map(|e| e.1).sum();
  let mut acc = 0.0;
  let keep = v.iter().position(|e| { acc += e.1; acc >= total * coverage }).map_or(v.len(), |i| i + 1);
  v.truncate(keep);
}

impl Ngrams {
  pub fn new(stats: &CorpusStats, bigram_coverage: f64, trigram_coverage: f64) -> Self {
    let mut uni = [0.0; KEYS];
    let mut bi: HashMap<(u8, u8), f64> = HashMap::new();
    let mut tri: HashMap<(u8, u8, u8), f64> = HashMap::new();
    let letters: u64 = stats.letters.iter().filter(|(c, _)| c.is_ascii_lowercase()).map(|(_, n)| n).sum();
    let f = 100.0 / letters.max(1) as f64;
    let key = keycorpus::char_key;
    for (&c, &n) in &stats.letters { if let Some(k) = key(c) { uni[k as usize] += n as f64 * f; } }
    for (&(a, b), &n) in &stats.bigrams {
      // 같은 기호 연속(::)도 둠: 비용은 없지만 레이어 키를 누른 채로 이어 치는 횟수에 들어감
      if let (Some(x), Some(y)) = (key(a), key(b)) && (x != y || is_sym(x)) && (is_sym(x) || is_sym(y)) { *bi.entry((x, y)).or_default() += n as f64 * f; }
    }
    for (&(a, b, c), &n) in &stats.trigrams {
      if let (Some(x), Some(y), Some(z)) = (key(a), key(b), key(c)) && (is_sym(x) || is_sym(y) || is_sym(z)) {
        *tri.entry((x, y, z)).or_default() += n as f64 * f;
      }
    }
    let mut bv: Vec<((u8, u8), f64)> = bi.into_iter().collect();
    let mut tv: Vec<((u8, u8, u8), f64)> = tri.into_iter().collect();
    trim(&mut bv, bigram_coverage);
    trim(&mut tv, trigram_coverage);
    let bi: Vec<(u8, u8, f64)> = bv.into_iter().map(|((a, b), w)| (a, b, w)).collect();
    let tri: Vec<(u8, u8, u8, f64)> = tv.into_iter().map(|((a, b, c), w)| (a, b, c, w)).collect();
    let mut bi_of = vec![Vec::new(); NSYM];
    let mut tri_of = vec![Vec::new(); NSYM];
    for (i, &(a, b, _)) in bi.iter().enumerate() {
      for k in [a, b] { if is_sym(k) && bi_of[(k - FIRST_SYM) as usize].last() != Some(&(i as u32)) { bi_of[(k - FIRST_SYM) as usize].push(i as u32); } }
    }
    for (i, &(a, b, c, _)) in tri.iter().enumerate() {
      let mut ks = [a, b, c];
      ks.sort_unstable();
      for (j, &k) in ks.iter().enumerate() {
        if is_sym(k) && (j == 0 || ks[j - 1] != k) { tri_of[(k - FIRST_SYM) as usize].push(i as u32); }
      }
    }
    Ngrams { uni, bi, tri, bi_of, tri_of }
  }

  // 기호 누름 수 (글자 100개당)
  pub fn symbol_presses(&self) -> f64 { self.uni[FIRST_SYM as usize..].iter().sum() }
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Slot {
  pos: u8,
  // 심볼 레이어 칸이면 true (누를 때 탭이 앞에 낌)
  tap: bool,
}

struct Pair {
  a: usize,
  b: usize,
  // 놓을 수 있는 (a의 칸, b의 칸). 순서 없는 짝은 두 방향 모두 들어 있음
  places: Vec<(u8, u8)>,
}

// 배치 상태. 항목은 기호 0..32와 빈칸 표시(32..칸 수)
#[derive(Clone)]
pub struct State {
  slot_of: Vec<u8>,
  item_at: Vec<u8>,
  pos: [u8; KEYS],
  tap: [bool; KEYS],
}

type Move = Vec<(usize, u8)>;

pub struct Problem<'a> {
  t: &'a Tables,
  g: &'a Ngrams,
  key_pos: [u8; KEYS],
  slots: Vec<Slot>,
  pairs: Vec<Pair>,
  pair_of: Vec<Option<usize>>,
  singles: Vec<usize>,
  tap_cost: f64,
  base_extra: f64,
}

struct Work {
  bmark: Vec<u32>,
  tmark: Vec<u32>,
  stamp: u32,
  bl: Vec<u32>,
  tl: Vec<u32>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Breakdown {
  pub effort: f64,
  pub taps: f64,
  pub sfb: f64,
  pub scissors: f64,
  pub rolling: f64,
}

impl Breakdown {
  pub fn total(&self) -> f64 { self.effort + self.taps + self.sfb + self.scissors - self.rolling }
}

// 짝을 놓을 수 있는 칸 (심볼 레이어 칸 번호 = 위치): 한 손 같은 줄(떨어져 있어도 됨)과 좌우 대칭(양손 같은 손가락·같은 줄).
// 순서 있는 짝은 첫 글자가 왼쪽·왼손인 방향만. 순서 없는 짝은 한 손 같은 열(떨어져 있어도 됨)도 되고 두 방향 모두
fn placements(ordered: bool) -> Vec<(u8, u8)> {
  let at = |r: usize, c: usize| (r * COLS + c) as u8;
  let left = |c: usize| c < COLS / 2;
  let mut v: Vec<(u8, u8)> = Vec::new();
  for r in 0..ROWS {
    for c in 0..COLS {
      for c2 in c + 1..COLS { if left(c) == left(c2) { v.push((at(r, c), at(r, c2))); } }
      if !ordered { for r2 in r + 1..ROWS { v.push((at(r, c), at(r2, c))); } }
    }
    for c in 0..COLS / 2 { v.push((at(r, c), at(r, COLS - 1 - c))); }
  }
  if !ordered {
    let rev: Vec<(u8, u8)> = v.iter().map(|&(a, b)| (b, a)).collect();
    v.extend(rev);
  }
  v
}

// 짝 하나의 모양 (위치는 칸 번호 = 줄 × 10 + 열, 두 기호의 순서는 상관없음). 열은 왼손 기준 (오른손은 거울 자리로 바꿈)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
  // 한 손 가로: 손(0 왼손, 1 오른손), 줄, 두 열
  Row { hand: usize, row: usize, cols: (usize, usize) },
  // 한 손 세로: 위아래 두 줄, 열
  Col { rows: (usize, usize), col: usize },
  // 양손 대칭: 줄, 열
  Mirror { row: usize, col: usize },
}

impl Shape {
  // 맞춤 묶음: 0 한 손 가로, 1 한 손 세로, 2 양손 대칭
  fn group(self) -> usize { match self { Shape::Row { .. } => 0, Shape::Col { .. } => 1, Shape::Mirror { .. } => 2 } }
}

// 짝 규칙에 맞지 않는 자리(대각선, 양손인데 대칭이 아님)는 None
fn pair_shape(a: usize, b: usize) -> Option<Shape> {
  let half = COLS / 2;
  let left = |c: usize| if c < half { c } else { COLS - 1 - c };
  let ((ra, ca), (rb, cb)) = ((a / COLS, a % COLS), (b / COLS, b % COLS));
  if ra == rb && ca != cb && (ca < half) == (cb < half) {
    let (x, y) = (left(ca), left(cb));
    Some(Shape::Row { hand: usize::from(ca >= half), row: ra, cols: (x.min(y), x.max(y)) })
  } else if ra == rb && ca + cb == COLS - 1 {
    Some(Shape::Mirror { row: ra, col: ca.min(cb) })
  } else if ca == cb && ra != rb {
    Some(Shape::Col { rows: (ra.min(rb), ra.max(rb)), col: left(ca) })
  } else {
    None
  }
}

// f가 같은 가장 많은 항목 수
fn most<T, K: PartialEq>(ks: &[T], f: impl Fn(&T) -> K) -> usize {
  ks.iter().map(|k| ks.iter().filter(|j| f(j) == f(k)).count()).max().unwrap_or(0)
}

// linked로 이어지는 가장 큰 덩어리의 항목 수 (이어진 두 항목의 덩어리 번호를 작은 쪽으로 맞추기를 바뀌지 않을 때까지)
fn largest_group<T>(ks: &[T], linked: impl Fn(&T, &T) -> bool) -> usize {
  let mut label: [usize; NSYM / 2] = std::array::from_fn(|i| i);
  loop {
    let mut changed = false;
    for (i, a) in ks.iter().enumerate() {
      for (j, b) in ks.iter().enumerate().skip(i + 1) {
        if label[i] != label[j] && linked(a, b) {
          let m = label[i].min(label[j]);
          (label[i], label[j]) = (m, m);
          changed = true;
        }
      }
    }
    if !changed { break; }
  }
  most(&label[..ks.len()], |l| *l)
}

// 한 손 가로 짝 (손, 줄, 두 열): 모두 같은 두 열. 양쪽으로 한 칸씩 넓은 짝은 같은 손·같은 줄에서 그 두 열의 짝을 감쌀 때만 됨
fn row_violations(ks: &[(usize, usize, (usize, usize))]) -> usize {
  let fits = |base: (usize, usize)| ks.iter().filter(|p| p.2 == base
    || (p.2.0 + 1 == base.0 && p.2.1 == base.1 + 1 && ks.iter().any(|q| q.2 == base && (q.0, q.1) == (p.0, p.1)))).count();
  ks.len() - ks.iter().map(|k| fits(k.2)).max().unwrap_or(0)
}

// 양손 대칭 짝 (줄, 열): 모두 같은 열이거나 모두 같은 줄 (떨어져 있어도 됨). 꺾인 모양은 바로 옆(같은 줄의 옆 열, 같은 열의
// 옆 줄)끼리 이어서 한 덩어리일 때만 됨
fn mirror_violations(ks: &[(usize, usize)]) -> usize {
  let next = largest_group(ks, |a, b| (a.0 == b.0 && a.1.abs_diff(b.1) == 1) || (a.1 == b.1 && a.0.abs_diff(b.0) == 1));
  ks.len() - next.max(most(ks, |k| k.0)).max(most(ks, |k| k.1))
}

// 짝 맞춤 규칙: 모양마다 어긴 쌍 수 [한 손 가로, 한 손 세로 (모두 같은 두 줄), 양손 대칭]
fn shape_violations(places: &[(usize, usize)]) -> [usize; 3] {
  let mut row_ks = [(0usize, 0usize, (0usize, 0usize)); NSYM / 2];
  let mut col_ks = [(0usize, 0usize); NSYM / 2];
  let mut mir_ks = [(0usize, 0usize); NSYM / 2];
  let (mut nr, mut nc, mut nm) = (0, 0, 0);
  for &(a, b) in places {
    match pair_shape(a, b) {
      Some(Shape::Row { hand, row, cols }) => { row_ks[nr] = (hand, row, cols); nr += 1; }
      Some(Shape::Col { rows, .. }) => { col_ks[nc] = rows; nc += 1; }
      Some(Shape::Mirror { row, col }) => { mir_ks[nm] = (row, col); nm += 1; }
      None => {}
    }
  }
  [row_violations(&row_ks[..nr]), nc - most(&col_ks[..nc], |k| *k), mirror_violations(&mir_ks[..nm])]
}

impl<'a> Problem<'a> {
  pub fn new(t: &'a Tables, g: &'a Ngrams, base: &Layout, cfg: &SymbolsConfig) -> anyhow::Result<Self> {
    let mut key_pos = [0; KEYS];
    for (id, kp) in key_pos.iter_mut().enumerate().take(26) {
      anyhow::ensure!(base.pos[id] != EMPTY, "the base layout has no key {}", (b'a' + id as u8) as char);
      *kp = base.pos[id];
    }
    let p = base.pos[crate::SPACE_ID as usize];
    anyhow::ensure!(p != EMPTY, "the base layout has no space key (_)");
    key_pos[SPACE as usize] = p;
    let free: Vec<u8> = (0..N as u8).filter(|&p| base.slots[p as usize] == EMPTY).collect();
    anyhow::ensure!(N + free.len() >= NSYM, "the base layout leaves {} empty slots; 32 symbols need at least {}", free.len(), NSYM - N);
    let slots: Vec<Slot> = (0..N as u8).map(|p| Slot { pos: p, tap: true }).chain(free.iter().map(|&p| Slot { pos: p, tap: false })).collect();
    let mut pairs = Vec::new();
    for (list, ordered) in [(&cfg.pairs, true), (&cfg.loose_pairs, false)] {
      for p in list {
        let cs: Vec<char> = p.chars().collect();
        let idx = |c: char| keycorpus::sym_index(c).ok_or_else(|| anyhow::anyhow!("{c:?} is not an ASCII symbol"));
        pairs.push(Pair { a: idx(cs[0])?, b: idx(cs[1])?, places: placements(ordered) });
      }
    }
    let mut pair_of = vec![None; slots.len()];
    for (i, p) in pairs.iter().enumerate() { pair_of[p.a] = Some(i); pair_of[p.b] = Some(i); }
    let singles = (0..slots.len()).filter(|&i| pair_of[i].is_none()).collect();
    Ok(Problem { t, g, key_pos, slots, pairs, pair_of, singles, tap_cost: t.eff.iter().copied().fold(f64::INFINITY, f64::min), base_extra: f64::from(cfg.base_slot_extra) })
  }

  fn work(&self) -> Work {
    Work { bmark: vec![0; self.g.bi.len()], tmark: vec![0; self.g.tri.len()], stamp: 0, bl: Vec::new(), tl: Vec::new() }
  }

  // 항목 → 칸 배정에서 상태를 만듦 (slot_of 길이 = 칸 수)
  fn state_from(&self, slot_of: Vec<u8>) -> State {
    let mut st = State { item_at: vec![0; slot_of.len()], slot_of: vec![0; slot_of.len()], pos: self.key_pos, tap: [false; KEYS] };
    for (item, &s) in slot_of.iter().enumerate() { self.place(&mut st, item, s); }
    st
  }

  fn place(&self, st: &mut State, item: usize, slot: u8) {
    st.slot_of[item] = slot;
    st.item_at[slot as usize] = item as u8;
    if item < NSYM {
      let k = FIRST_SYM as usize + item;
      let s = self.slots[slot as usize];
      st.pos[k] = s.pos;
      st.tap[k] = s.tap;
    }
  }

  // 심볼 레이어 기호는 누를 때마다 탭 비용을 매기고, 이어 치는 기호(bi)에서 되돌려 줌
  fn uni(&self, st: &State, item: usize) -> f64 {
    if item >= NSYM { return 0.0; }
    let k = FIRST_SYM as usize + item;
    self.g.uni[k] * (self.t.eff[st.pos[k] as usize] + if st.tap[k] { self.tap_cost } else { self.base_extra })
  }

  // a 다음 b를 칠 때 엄지가 레이어 키를 새로 누르는지: b가 심볼 레이어 기호이고 a는 아닐 때
  fn tapped(st: &State, a: u8, b: u8) -> bool { st.tap[b as usize] && !st.tap[a as usize] }

  // 둘 다 심볼 레이어 기호면 레이어 키를 누른 채로 이어 침 (탭 하나를 아낌)
  fn held(st: &State, a: u8, b: u8) -> bool { st.tap[a as usize] && st.tap[b as usize] }

  // a 다음 b를 누르는 손가락 비용 (엄지가 새로 끼면 탭 너머 쌍). 탭 비용은 뺌
  fn pair(&self, st: &State, a: u8, b: u8) -> f64 {
    let i = st.pos[a as usize] as usize * N + st.pos[b as usize] as usize;
    if Self::tapped(st, a, b) { self.t.x[i] } else { self.t.bg[i] }
  }

  fn bi(&self, st: &State, a: u8, b: u8) -> f64 {
    self.pair(st, a, b) - if Self::held(st, a, b) { self.tap_cost } else { 0.0 }
  }

  fn tri(&self, st: &State, a: u8, b: u8, c: u8) -> f64 {
    if Self::tapped(st, a, b) || Self::tapped(st, b, c) { return 0.0; }
    let (pa, pb, pc) = (st.pos[a as usize] as usize, st.pos[b as usize] as usize, st.pos[c as usize] as usize);
    self.t.tg[(pa * N + pb) * N + pc]
  }

  pub fn total(&self, st: &State) -> f64 {
    let u: f64 = (0..NSYM).map(|i| self.uni(st, i)).sum();
    let b: f64 = self.g.bi.iter().map(|&(x, y, w)| w * self.bi(st, x, y)).sum();
    let t: f64 = self.g.tri.iter().map(|&(x, y, z, w)| w * self.tri(st, x, y, z)).sum();
    u + b + t
  }

  // 옮기는 항목들이 낀 부분의 비용
  fn part(&self, st: &State, mv: &Move, w: &mut Work) -> f64 {
    let mut c: f64 = mv.iter().map(|&(i, _)| self.uni(st, i)).sum();
    for &i in &w.bl { let (x, y, wt) = self.g.bi[i as usize]; c += wt * self.bi(st, x, y); }
    for &i in &w.tl { let (x, y, z, wt) = self.g.tri[i as usize]; c += wt * self.tri(st, x, y, z); }
    c
  }

  fn mark(&self, mv: &Move, w: &mut Work) {
    w.stamp = w.stamp.wrapping_add(1);
    if w.stamp == 0 { w.bmark.fill(0); w.tmark.fill(0); w.stamp = 1; }
    w.bl.clear();
    w.tl.clear();
    for &(item, _) in mv {
      if item >= NSYM { continue; }
      for &i in &self.g.bi_of[item] { if w.bmark[i as usize] != w.stamp { w.bmark[i as usize] = w.stamp; w.bl.push(i); } }
      for &i in &self.g.tri_of[item] { if w.tmark[i as usize] != w.stamp { w.tmark[i as usize] = w.stamp; w.tl.push(i); } }
    }
  }

  // 이동을 적용하고 비용 변화를 돌려줌. 되돌릴 때 쓸 이전 칸도 함께
  fn apply(&self, st: &mut State, mv: &Move, w: &mut Work) -> (f64, Move) {
    self.mark(mv, w);
    let before = self.part(st, mv, w);
    let old: Move = mv.iter().map(|&(i, _)| (i, st.slot_of[i])).collect();
    for &(i, s) in mv { self.place(st, i, s); }
    (self.part(st, mv, w) - before, old)
  }

  fn undo(&self, st: &mut State, old: &Move) {
    for &(i, s) in old { self.place(st, i, s); }
  }

  // 짝 p를 칸 (sa, sb)로 옮기는 이동 (그 칸에 있던 홀 항목은 짝이 비운 칸으로). 다른 짝과 겹치면 None
  fn pair_move(&self, st: &State, p: usize, sa: u8, sb: u8) -> Option<Move> {
    let pr = &self.pairs[p];
    let olds = [st.slot_of[pr.a], st.slot_of[pr.b]];
    if (sa, sb) == (olds[0], olds[1]) { return None; }
    let occ = [st.item_at[sa as usize] as usize, st.item_at[sb as usize] as usize];
    if occ.iter().any(|&o| o != pr.a && o != pr.b && self.pair_of[o].is_some()) { return None; }
    let vac: Vec<u8> = olds.iter().copied().filter(|&s| s != sa && s != sb).collect();
    let movers: Vec<usize> = occ.iter().copied().filter(|&o| o != pr.a && o != pr.b).collect();
    if movers.len() != vac.len() { return None; }
    let mut mv: Move = vec![(pr.a, sa), (pr.b, sb)];
    mv.extend(movers.into_iter().zip(vac));
    Some(mv)
  }

  // 두 짝의 칸을 맞바꾸는 이동 (방향은 각 짝이 받을 수 있는 쪽으로)
  fn pair_swap(&self, st: &State, p: usize, q: usize) -> Option<Move> {
    let (pp, qq) = (&self.pairs[p], &self.pairs[q]);
    let ps = (st.slot_of[pp.a], st.slot_of[pp.b]);
    let qs = (st.slot_of[qq.a], st.slot_of[qq.b]);
    let fit = |pr: &Pair, s: (u8, u8)| if pr.places.contains(&s) { Some(s) } else if pr.places.contains(&(s.1, s.0)) { Some((s.1, s.0)) } else { None };
    let (np, nq) = (fit(pp, qs)?, fit(qq, ps)?);
    Some(vec![(pp.a, np.0), (pp.b, np.1), (qq.a, nq.0), (qq.b, nq.1)])
  }

  fn random_state(&self, rng: &mut fastrand::Rng) -> State {
    let n = self.slots.len();
    loop {
      let mut slot_of = vec![u8::MAX; n];
      let mut used = vec![false; n];
      let mut ok = true;
      for pr in &self.pairs {
        let free: Vec<&(u8, u8)> = pr.places.iter().filter(|(a, b)| !used[*a as usize] && !used[*b as usize]).collect();
        if free.is_empty() { ok = false; break; }
        let &(a, b) = free[rng.usize(..free.len())];
        slot_of[pr.a] = a;
        slot_of[pr.b] = b;
        used[a as usize] = true;
        used[b as usize] = true;
      }
      if !ok { continue; }
      let mut rest: Vec<u8> = (0..n as u8).filter(|&s| !used[s as usize]).collect();
      rng.shuffle(&mut rest);
      for (&i, s) in self.singles.iter().zip(rest) { slot_of[i] = s; }
      return self.state_from(slot_of);
    }
  }

  fn random_move(&self, st: &State, rng: &mut fastrand::Rng) -> Option<Move> {
    let r = rng.f64();
    if r < 0.7 || self.pairs.is_empty() {
      let x = self.singles[rng.usize(..self.singles.len())];
      let y = self.singles[rng.usize(..self.singles.len())];
      if x == y || (x >= NSYM && y >= NSYM) { return None; }
      return Some(vec![(x, st.slot_of[y]), (y, st.slot_of[x])]);
    }
    let p = rng.usize(..self.pairs.len());
    if r < 0.95 {
      let pr = &self.pairs[p];
      let (sa, sb) = pr.places[rng.usize(..pr.places.len())];
      return self.pair_move(st, p, sa, sb);
    }
    let q = rng.usize(..self.pairs.len());
    if p == q { return None; }
    self.pair_swap(st, p, q)
  }

  // 짝 두 기호의 물리 위치 (칸 번호)
  fn pair_pos(&self, st: &State, p: &Pair) -> (usize, usize) {
    (self.slots[st.slot_of[p.a] as usize].pos as usize, self.slots[st.slot_of[p.b] as usize].pos as usize)
  }

  // 짝 맞춤 규칙을 어긴 쌍 수
  fn violations(&self, st: &State) -> usize {
    let mut places = [(0usize, 0usize); NSYM / 2];
    for (place, p) in places.iter_mut().zip(&self.pairs) { *place = self.pair_pos(st, p); }
    shape_violations(&places[..self.pairs.len()]).iter().sum()
  }

  // 이동이 짝을 옮기는지 (안 옮기면 맞춤 규칙을 어긴 정도가 그대로)
  fn moves_pair(&self, mv: &Move) -> bool {
    mv.iter().any(|&(i, _)| self.pair_of[i].is_some())
  }

  // 결정적 근처 탐색: 홀 항목 맞바꾸기, 짝 옮기기, 짝끼리 맞바꾸기를 더 좋아지지 않을 때까지.
  // 짝 맞춤 규칙을 더 어기게 되는 이동은 받지 않음 (규칙을 지키던 배치는 끝까지 지킴)
  fn polish(&self, mut st: State, w: &mut Work) -> (f64, State) {
    let mut viol = self.violations(&st);
    loop {
      let mut improved = false;
      let mut cands: Vec<Move> = Vec::new();
      for (i, &x) in self.singles.iter().enumerate() {
        for &y in &self.singles[i + 1..] {
          if x < NSYM || y < NSYM { cands.push(vec![(x, st.slot_of[y]), (y, st.slot_of[x])]); }
        }
      }
      for p in 0..self.pairs.len() {
        for &(sa, sb) in &self.pairs[p].places { if let Some(m) = self.pair_move(&st, p, sa, sb) { cands.push(m); } }
        for q in p + 1..self.pairs.len() { if let Some(m) = self.pair_swap(&st, p, q) { cands.push(m); } }
      }
      for mv in &cands {
        let (d, old) = self.apply(&mut st, mv, w);
        let v = self.violations(&st);
        if v <= viol && d + GROUP_PENALTY * (v as f64 - viol as f64) < -1e-9 { viol = v; improved = true; break; }
        self.undo(&mut st, &old);
      }
      if !improved { break; }
    }
    (self.total(&st), st)
  }

  fn anneal(&self, rng: &mut fastrand::Rng, steps: usize) -> (f64, State) {
    let mut w = self.work();
    let mut st = self.random_state(rng);
    let mut cur = self.total(&st);
    let mut viol = self.violations(&st);
    // 짝 맞춤 규칙을 지키는 배치 중 가장 좋은 것 (어기는 배치는 벌점을 안고 거쳐 가기만 함)
    let mut best: Option<(f64, State)> = (viol == 0).then(|| (cur, st.clone()));
    for it in 0..steps {
      let temp = T_START * (T_END / T_START).powf(it as f64 / steps as f64);
      let Some(mv) = self.random_move(&st, rng) else { continue };
      let (d, old) = self.apply(&mut st, &mv, &mut w);
      let v = if self.moves_pair(&mv) { self.violations(&st) } else { viol };
      let dv = d + GROUP_PENALTY * (v as f64 - viol as f64);
      if dv <= 0.0 || rng.f64() < (-dv / temp).exp() {
        cur += d;
        viol = v;
        if viol == 0 && best.as_ref().is_none_or(|(b, _)| cur < b - 1e-12) { best = Some((cur, st.clone())); }
      } else {
        self.undo(&mut st, &old);
      }
    }
    self.polish(best.map_or(st, |(_, s)| s), &mut w)
  }

  // 무작위 시작 담금질을 starts_per_batch개씩 돌리고 같은 최고점이 stop_after_hits번 나오면 멈춤 (배열 최적화와 같은 규칙).
  // seed가 있으면(이전 결과) 첫 시작으로 써서 결과가 그보다 나빠지지 않게 함
  // 짝 맞춤 규칙을 어긴 채 끝난 시작은 최고점으로 치지 않음 (모든 시작이 그러면 그중 가장 좋은 것을 돌려줌)
  pub fn search(&self, sa: &SaParams, steps: usize, seed: Option<State>) -> (f64, State, usize, usize) {
    let mut best: Option<(f64, State)> = None;
    let mut fallback: Option<(f64, State)> = None;
    let (mut hits, mut starts) = (0usize, 0usize);
    let t0 = std::time::Instant::now();
    loop {
      let n = sa.starts_per_batch.min(sa.max_starts - starts);
      let seeds: Vec<u64> = (0..n).map(|_| fastrand::u64(..)).collect();
      let first = starts == 0;
      let found: Vec<(f64, State)> = seeds.par_iter().enumerate().map(|(k, &s)| {
        let mut rng = fastrand::Rng::with_seed(s);
        match &seed {
          Some(st) if first && k == 0 => self.polish(st.clone(), &mut self.work()),
          _ => self.anneal(&mut rng, steps),
        }
      }).collect();
      for (c, st) in found {
        starts += 1;
        if self.violations(&st) > 0 {
          if fallback.as_ref().is_none_or(|(b, _)| c < *b) { fallback = Some((c, st)); }
          continue;
        }
        match &best {
          Some((b, _)) if c > b + 1e-9 => {}
          Some((b, _)) if c >= b - 1e-9 => hits += 1,
          _ => { best = Some((c, st)); hits = 1; }
        }
      }
      let b = best.as_ref().map_or(f64::NAN, |x| x.0);
      print!("\r\x1b[2K[Symbols] {starts} starts  best {b:.2} x{hits}\t{:.1}s", t0.elapsed().as_secs_f32());
      { use std::io::Write; let _ = std::io::stdout().flush(); }
      if crate::SHUTDOWN.load(std::sync::atomic::Ordering::SeqCst) || starts >= sa.max_starts
        || (starts >= sa.starts_per_batch && hits >= sa.stop_after_hits) { break; }
    }
    println!();
    let (c, st) = best.or(fallback).expect("at least one start");
    (c, st, hits, starts)
  }

  // 레이어 키 탭 수 (글자 100개당): 심볼 레이어 기호 누름에서 앞 키도 심볼 레이어 기호인 누름(누른 채로 이어 침)을 뺌
  pub fn layer_taps(&self, st: &State) -> f64 {
    let presses: f64 = (FIRST_SYM as usize..KEYS).filter(|&k| st.tap[k]).map(|k| self.g.uni[k]).sum();
    let held: f64 = self.g.bi.iter().filter(|&&(x, y, _)| Self::held(st, x, y)).map(|e| e.2).sum();
    presses - held
  }

  pub fn breakdown(&self, st: &State) -> Breakdown {
    let t = self.t;
    let mut b = Breakdown { taps: self.layer_taps(st) * self.tap_cost, ..Breakdown::default() };
    for k in FIRST_SYM as usize..KEYS {
      b.effort += self.g.uni[k] * (t.eff[st.pos[k] as usize] + if st.tap[k] { 0.0 } else { self.base_extra });
    }
    for &(x, y, w) in &self.g.bi {
      let i = st.pos[x as usize] as usize * N + st.pos[y as usize] as usize;
      if Self::tapped(st, x, y) { b.sfb += w * t.xsfs[i]; b.scissors += w * t.xssc[i]; }
      else { b.sfb += w * t.sfb[i]; b.scissors += w * t.sc[i]; b.rolling += w * t.roll[i]; }
    }
    for &(x, y, z, w) in &self.g.tri {
      if Self::tapped(st, x, y) || Self::tapped(st, y, z) { continue; }
      let j = (st.pos[x as usize] as usize * N + st.pos[y as usize] as usize) * N + st.pos[z as usize] as usize;
      b.sfb += w * t.sfs[j];
      b.scissors += w * t.ssc[j];
      b.rolling += w * t.roll3[j];
    }
    b
  }

  // 바이그램 하나의 분류 (표시용)
  fn kind(&self, st: &State, a: u8, b: u8) -> &'static str {
    let (pa, pb) = (st.pos[a as usize], st.pos[b as usize]);
    let i = pa as usize * N + pb as usize;
    if Self::tapped(st, a, b) {
      return if pa == pb { "tap:same-key" } else if self.t.xsfs[i] > 0.0 { "tap:sfs" } else if self.t.xssc[i] > 0.0 { "tap:skip-sc" } else { "tap" };
    }
    let (ca, cb) = (pa as usize % COLS, pb as usize % COLS);
    if pa == pb { "same-key" }
    else if (ca < COLS / 2) != (cb < COLS / 2) { "alt" }
    else if self.t.sfb[i] > 0.0 { "SFB" }
    else if self.t.sc[i] > 0.0 { if matches!(ca, 4 | 5) || matches!(cb, 4 | 5) { "LSB" } else { "scissor" } }
    else if self.t.roll[i] > 0.0 { if finger(pa as usize) > finger(pb as usize) { "inroll" } else { "outroll" } }
    else { "other" }
  }

  pub fn print_report(&self, st: &State) {
    let b = self.breakdown(st);
    println!("cost={:.2}  effort={:.2}  taps={:.2}  sfb={:.2}  scissors={:.2}  rolling={:.2}   (per 100 letters, n-grams with a symbol; {:.1} symbol presses, {:.1} layer taps)",
      b.total(), b.effort, b.taps, b.sfb, b.scissors, -b.rolling, self.g.symbol_presses(), self.layer_taps(st));
    let mut kinds: Vec<(&str, f64)> = Vec::new();
    let mut total = 0.0;
    for &(x, y, w) in &self.g.bi {
      let k = self.kind(st, x, y);
      total += w;
      match kinds.iter_mut().find(|e| e.0 == k) { Some(e) => e.1 += w, None => kinds.push((k, w)) }
    }
    kinds.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("bigrams with a symbol: {}", kinds.iter().map(|(k, w)| format!("{k} {:.1}%", 100.0 * w / total)).collect::<Vec<_>>().join("  "));
    let show = |ids: &[u8]| ids.iter().map(|&k| if k == SPACE { SPACE_TOKEN } else { keycorpus::key_char(k) }).collect::<String>();
    let mut worst: Vec<(f64, String, &str)> = Vec::new();
    for &(x, y, w) in &self.g.bi { let c = w * self.pair(st, x, y); if c > 0.0 { worst.push((c, show(&[x, y]), self.kind(st, x, y))); } }
    for &(x, y, z, w) in &self.g.tri { let c = w * self.tri(st, x, y, z); if c > 0.0 { worst.push((c, show(&[x, y, z]), "trigram")); } }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("worst: {}", worst.iter().take(12).map(|(c, s, k)| format!("[{s}] {k} {c:.2}")).collect::<Vec<_>>().join("  "));
  }

  // 표시·저장용 두 레이어 (각 3행 × 10칸 토큰)
  fn grids(&self, base: &Layout, st: &State) -> [[[char; COLS]; ROWS]; 2] {
    let mut g = [[[EMPTY_TOKEN; COLS]; ROWS]; 2];
    for p in 0..N {
      let id = base.slots[p];
      g[1][p / COLS][p % COLS] = if id == crate::SPACE_ID { SPACE_TOKEN } else if id < 26 { (b'a' + id) as char } else { EMPTY_TOKEN };
    }
    for item in 0..NSYM {
      let s = self.slots[st.slot_of[item] as usize];
      g[usize::from(!s.tap)][s.pos as usize / COLS][s.pos as usize % COLS] = keycorpus::key_char(FIRST_SYM + item as u8);
    }
    g
  }

  pub fn to_text(&self, base: &Layout, st: &State) -> String {
    let g = self.grids(base, st);
    let row = |r: &[char; COLS]| format!(" {}   {}", r[..5].iter().map(char::to_string).collect::<Vec<_>>().join(" "), r[5..].iter().map(char::to_string).collect::<Vec<_>>().join(" "));
    let mut out = String::from("symbol layer (one-shot)\n");
    for r in &g[0] { out.push_str(&row(r)); out.push('\n'); }
    out.push_str("base layer\n");
    for r in &g[1] { out.push_str(&row(r)); out.push('\n'); }
    out
  }

  // 파일에서 읽은 기호 배치 → 상태 (base는 parse_text가 돌려준 것과 같아야 함)
  pub fn state_of(&self, placed: &[(u8, bool); NSYM]) -> anyhow::Result<State> {
    let n = self.slots.len();
    let mut slot_of = vec![u8::MAX; n];
    let mut used = vec![false; n];
    for (item, &(pos, tap)) in placed.iter().enumerate() {
      let s = self.slots.iter().position(|sl| sl.pos == pos && sl.tap == tap)
        .ok_or_else(|| anyhow::anyhow!("symbol {:?} is on a base key that is not empty", keycorpus::key_char(FIRST_SYM + item as u8)))?;
      slot_of[item] = s as u8;
      used[s] = true;
    }
    let mut free = (0..n).filter(|&s| !used[s]);
    for x in slot_of.iter_mut().skip(NSYM) { *x = free.next().expect("one slot per item") as u8; }
    Ok(self.state_from(slot_of))
  }

  // 짝 규칙을 지키는지 (파일에서 읽은 배치는 지키지 않아도 채점은 함). 짝 맞춤 규칙을 어기면 그 모양의 짝을 함께 알림
  pub fn broken_pairs(&self, st: &State) -> Vec<String> {
    let name = |p: &Pair| format!("{}{}", keycorpus::key_char(FIRST_SYM + p.a as u8), keycorpus::key_char(FIRST_SYM + p.b as u8));
    let mut out: Vec<String> = self.pairs.iter().filter(|p| !p.places.contains(&(st.slot_of[p.a], st.slot_of[p.b]))).map(name).collect();
    let places: Vec<(usize, usize)> = self.pairs.iter().map(|p| self.pair_pos(st, p)).collect();
    for (g, &v) in shape_violations(&places).iter().enumerate() {
      if v == 0 { continue; }
      let names: Vec<String> = self.pairs.iter().zip(&places)
        .filter(|&(_, &(a, b))| pair_shape(a, b).is_some_and(|s| s.group() == g)).map(|(p, _)| name(p)).collect();
      out.push(format!("{} ({} pairs not lined up)", names.join(" "), ["one-hand row", "one-hand column", "mirrored"][g]));
    }
    out
  }
}

// 기호 배치 파일 읽기: 칸 10개짜리 줄만 행으로 봄 (나머지 줄은 제목 등으로 무시). 앞 3행은 심볼 레이어, 다음 3행은 베이스
// (글자, ␣ = 스페이스, · = 빈칸, 기호 = 베이스 빈칸에 둔 기호). 돌려주는 것: 베이스 배열과 기호마다 (위치, 심볼 레이어인지)
pub fn parse_text(text: &str) -> anyhow::Result<(Layout, [(u8, bool); NSYM])> {
  // 한 글자짜리 토큰만 칸으로 (여러 글자 토큰이 있는 줄은 제목으로 봄)
  let one_char = |t: &str| { let mut c = t.chars(); match (c.next(), c.next()) { (Some(ch), None) => Some(ch), _ => None } };
  let rows: Vec<Vec<char>> = text.lines()
    .map(|l| l.split_whitespace().map(one_char).collect::<Vec<Option<char>>>())
    .filter(|r| r.len() == COLS && r.iter().all(Option::is_some))
    .map(|r| r.into_iter().flatten().collect())
    .collect();
  anyhow::ensure!(rows.len() == 2 * ROWS, "expected {} rows of {COLS} keys (symbol layer, then base layer), found {}", 2 * ROWS, rows.len());
  let mut placed: [Option<(u8, bool)>; NSYM] = [None; NSYM];
  let mut base_text = String::new();
  for (r, row) in rows.iter().enumerate() {
    let sym_layer = r < ROWS;
    for (c, &ch) in row.iter().enumerate() {
      let pos = ((r % ROWS) * COLS + c) as u8;
      if let Some(i) = keycorpus::sym_index(ch) {
        anyhow::ensure!(placed[i].is_none(), "symbol {ch:?} appears more than once");
        placed[i] = Some((pos, sym_layer));
        if !sym_layer { base_text.push('.'); }
        continue;
      }
      if sym_layer {
        anyhow::ensure!(ch == EMPTY_TOKEN, "unexpected {ch:?} in the symbol layer (use {EMPTY_TOKEN} for an empty key)");
        continue;
      }
      base_text.push(match ch {
        SPACE_TOKEN => '_',
        EMPTY_TOKEN => '.',
        c if c.is_ascii_alphabetic() => c,
        _ => anyhow::bail!("unexpected {ch:?} in the base layer"),
      });
    }
    if !sym_layer { base_text.push('\n'); }
  }
  let missing: String = (0..NSYM).filter(|&i| placed[i].is_none()).map(|i| keycorpus::key_char(FIRST_SYM + i as u8)).collect();
  anyhow::ensure!(missing.is_empty(), "symbols missing from the file: {missing}");
  let base = Layout::parse(&base_text)?;
  anyhow::ensure!(base.key_count() == 27, "the base layer must hold a-z and ␣");
  Ok((base, placed.map(|p| p.expect("checked above"))))
}

// 심볼 레이어 최적화 (symbols 명령): base는 고른 베이스 배열(뒤집었으면 뒤집은 것), label은 결과 파일 이름에 붙임
pub fn optimize(result_path: &Path, config: &Config, base: &Layout, label: &str) -> anyhow::Result<()> {
  let g = code_ngrams(result_path, config)?;
  let t = Tables::new(config);
  let problem = Problem::new(&t, &g, base, &config.symbols)?;
  let out = result_path.join(format!("symbols-{label}.txt"));
  // 같은 베이스로 저장해 둔 결과가 있으면 첫 시작점으로 (이전보다 나빠지지 않게)
  let seed = std::fs::read_to_string(&out).ok().and_then(|text| parse_text(&text).ok())
    .filter(|(b, _)| b.slots == base.slots)
    .and_then(|(_, placed)| problem.state_of(&placed).ok())
    .filter(|st| problem.broken_pairs(st).is_empty());
  if seed.is_some() { println!("[Symbols] also starting from {}", out.display()); }
  let (_, st, hits, starts) = problem.search(&config.sa, config.symbols.steps, seed);
  println!("[Symbols] best found {hits} times in {starts} starts");
  let broken = problem.broken_pairs(&st);
  if !broken.is_empty() { println!("note: no start ended inside the pair rules; this layer breaks: {}", broken.join(" ")); }
  let text = problem.to_text(base, &st);
  print!("{text}");
  problem.print_report(&st);
  let tmp = out.with_extension("txt.tmp");
  std::fs::write(&tmp, &text)?;
  std::fs::rename(&tmp, &out)?;
  println!("[Symbols] saved {} (edit it and check with `symbols-score`)", out.display());
  Ok(())
}

// 기호 배치 파일들을 같은 기준으로 채점 (symbols-score 명령)
pub fn score_files(result_path: &Path, config: &Config, files: &[String]) -> anyhow::Result<()> {
  anyhow::ensure!(!files.is_empty(), "usage: <result_path> symbols-score <symbols_file>...");
  let g = code_ngrams(result_path, config)?;
  let t = Tables::new(config);
  for f in files {
    let (base, placed) = parse_text(&std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?)
      .with_context(|| format!("parsing {f}"))?;
    let problem = Problem::new(&t, &g, &base, &config.symbols)?;
    let st = problem.state_of(&placed)?;
    println!("[{f}]");
    print!("{}", problem.to_text(&base, &st));
    let broken = problem.broken_pairs(&st);
    if !broken.is_empty() { println!("note: pairs not placed by the pair rules: {}", broken.join(" ")); }
    problem.print_report(&st);
    println!();
  }
  Ok(())
}

// 심볼 레이어에 쓰는 코드 코퍼스의 n-gram (코드 코퍼스가 없거나 만들 때와 설정이 다르면 다시 만듦)
fn code_ngrams(result_path: &Path, config: &Config) -> anyhow::Result<Ngrams> {
  Ok(Ngrams::new(&crate::code::load_or_build(result_path, config)?.symbols, config.bigram_coverage, config.trigram_coverage))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn config() -> Config {
    let mut t: toml::Table = toml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml")).unwrap()).unwrap();
    t.insert("wikipedia_dump_url".into(), toml::Value::String("https://m.org/a.xml.bz2".into()));
    toml::Value::Table(t).try_into().unwrap()
  }

  // 뒤집은 grid 1등 (스페이스가 오른손 검지 홈)
  const BASE: &str = "plou.qbdfk\ncthe.x_snr\nwgja.zivmy\n";
  const CODE: &str = "fn new(&self, name: &str) -> Result<(), Error> {\nlet v: Vec<_> = items.iter().map(|x| x * 2).collect();\n\
    match self.kind { Some(k) => k, None => return Ok(()) }\n#[derive(Debug, Clone)]\nprintln!(\"{:?} {}\\n\", a, b);\n\
    if a != b && c == d || !e { x += y[i] - z / 2; }\nuse std::io::{self, Read}; // note\nlet s = 'c'; let t = \"it's\";\nx ^= y % 3; @ ~ $ `\n";

  fn ngrams() -> Ngrams {
    let mut code = keycorpus::KeyCorpus::new();
    keycorpus::count_text(crate::code::strip_comments(&CODE.repeat(3)).as_bytes(), &mut code);
    Ngrams::new(&code.into_stats(), 1.0, 1.0)
  }

  #[test]
  fn delta_matches_full_recompute() {
    let c = config();
    let (t, g) = (Tables::new(&c), ngrams());
    let base = Layout::parse(BASE).unwrap();
    let p = Problem::new(&t, &g, &base, &c.symbols).unwrap();
    let mut rng = fastrand::Rng::with_seed(7);
    let mut st = p.random_state(&mut rng);
    let mut w = p.work();
    let mut cur = p.total(&st);
    let mut applied = 0;
    for _ in 0..3000 {
      let Some(mv) = p.random_move(&st, &mut rng) else { continue };
      let (d, old) = p.apply(&mut st, &mv, &mut w);
      let full = p.total(&st);
      assert!((cur + d - full).abs() < 1e-7, "{cur} + {d} != {full}");
      if rng.bool() { cur = full; applied += 1; } else { p.undo(&mut st, &old); assert!((p.total(&st) - cur).abs() < 1e-7); }
      assert!((p.breakdown(&st).total() - p.total(&st)).abs() < 1e-7);
    }
    assert!(applied > 500);
  }

  #[test]
  fn tap_makes_pairs_cross_and_base_slots_cost_more() {
    let c = config();
    let t = Tables::new(&c);
    // 심볼 레이어 기호 앞의 쌍은 탭 너머 쌍: 같은 손가락 다른 키면 스킵그램, 다른 손가락은 0 (롤도 없음)
    let (e_pos, idx_top, mid_home) = (13usize, 3usize, 12usize);
    assert!(t.x[e_pos * N + idx_top] > 0.0 && t.x[e_pos * N + idx_top] < t.bg[e_pos * N + idx_top]);
    assert_eq!(t.x[e_pos * N + mid_home], 0.0);
    assert!(t.bg[mid_home * N + e_pos] < 0.0); // 안쪽 롤 보너스는 탭 없이 이어질 때만
    // 같은 키는 레이어가 달라도 비용 없음
    assert_eq!((t.x[e_pos * N + e_pos], t.bg[e_pos * N + e_pos]), (0.0, 0.0));
    let g = ngrams();
    let base = Layout::parse(BASE).unwrap();
    let p = Problem::new(&t, &g, &base, &c.symbols).unwrap();
    // 탭 비용은 노력값 표에서 가장 작은 값 (기본 설정은 중지·검지 홈 1.0)
    assert_eq!(p.tap_cost, 1.0);
    let mut c2 = config();
    c2.effort.home[3] = 0.8;
    let t2 = Tables::new(&c2);
    assert!((Problem::new(&t2, &g, &base, &c2.symbols).unwrap().tap_cost - 0.8).abs() < 1e-6);
    // 베이스 빈칸 3칸 + 심볼 레이어 30칸
    assert_eq!(p.slots.iter().filter(|s| !s.tap).map(|s| s.pos).collect::<Vec<_>>(), vec![4, 14, 24]);
    let mut st = p.random_state(&mut fastrand::Rng::with_seed(1));
    let paren = keycorpus::sym_index('(').unwrap();
    // '('를 심볼 레이어 14번 칸(왼손 가운데 열 홈)에, 그다음 베이스의 같은 자리 빈칸에
    let move_to = |st: &mut State, slot: u8| {
      let other = st.item_at[slot as usize] as usize;
      if other != paren { p.apply(st, &vec![(paren, slot), (other, st.slot_of[paren])], &mut p.work()); }
    };
    move_to(&mut st, 14);
    let on_layer = p.uni(&st, paren);
    move_to(&mut st, (N + 1) as u8);
    assert_eq!((st.pos[FIRST_SYM as usize + paren], st.tap[FIRST_SYM as usize + paren]), (14, false));
    let on_base = p.uni(&st, paren);
    assert!((on_base - on_layer - g.uni[(FIRST_SYM as usize) + paren] * (p.base_extra - p.tap_cost)).abs() < 1e-9);
  }

  #[test]
  fn consecutive_symbols_hold_the_layer_key() {
    let c = config();
    let t = Tables::new(&c);
    let base = Layout::parse(BASE).unwrap();
    // 끊김 없는 한 줄: 탭 수는 앞 키가 심볼 레이어 기호가 아닌 심볼 레이어 기호 누름 수
    let text = "a::b();c->d!=e[f]?;g";
    let mut code = keycorpus::KeyCorpus::new();
    keycorpus::count_text(text.as_bytes(), &mut code);
    let g = Ngrams::new(&code.into_stats(), 1.0, 1.0);
    // 같은 기호 연속(::)도 빅그램으로 들어옴
    let colon = FIRST_SYM + keycorpus::sym_index(':').unwrap() as u8;
    assert!(g.bi.iter().any(|&(x, y, _)| (x, y) == (colon, colon)));
    let p = Problem::new(&t, &g, &base, &c.symbols).unwrap();
    let ids: Vec<u8> = text.chars().map(|ch| keycorpus::char_key(ch).unwrap()).collect();
    let per = 100.0 / text.chars().filter(char::is_ascii_lowercase).count() as f64;
    let mut seen_mixed = false;
    for seed in 0..40 {
      let st = p.random_state(&mut fastrand::Rng::with_seed(seed));
      let taps = (0..ids.len()).filter(|&i| st.tap[ids[i] as usize] && (i == 0 || !st.tap[ids[i - 1] as usize])).count();
      assert!((p.layer_taps(&st) - taps as f64 * per).abs() < 1e-9, "seed {seed}");
      assert!((p.breakdown(&st).taps - p.layer_taps(&st) * p.tap_cost).abs() < 1e-9);
      // 이어 치면 보통 바이그램에서 탭 하나를 되돌려 받고, 엄지가 새로 끼면 탭 너머 쌍
      for &(x, y, _) in &g.bi {
        let i = st.pos[x as usize] as usize * N + st.pos[y as usize] as usize;
        let want = match (st.tap[x as usize], st.tap[y as usize]) { (true, true) => t.bg[i] - p.tap_cost, (false, true) => t.x[i], _ => t.bg[i] };
        assert!((p.bi(&st, x, y) - want).abs() < 1e-12);
      }
      // 기호 하나는 베이스, 이웃은 심볼 레이어인 경우도 거침
      seen_mixed |= ids.windows(2).any(|w| is_sym(w[0]) && is_sym(w[1]) && st.tap[w[0] as usize] != st.tap[w[1] as usize]);
      assert!((p.breakdown(&st).total() - p.total(&st)).abs() < 1e-9);
    }
    assert!(seen_mixed);
  }

  #[test]
  fn ordered_pairs_stay_in_one_row_and_loose_pairs_may_also_share_a_column() {
    let at = |r: usize, c: usize| (r * COLS + c) as u8;
    let rc = |p: u8| (p as usize / COLS, p as usize % COLS);
    let same_row = |a: u8, b: u8| { let ((ra, ca), (rb, cb)) = (rc(a), rc(b)); ra == rb && ca != cb && (ca < COLS / 2) == (cb < COLS / 2) };
    let same_col = |a: u8, b: u8| { let ((ra, ca), (rb, cb)) = (rc(a), rc(b)); ca == cb && ra != rb };
    let mirror = |a: u8, b: u8| { let ((ra, ca), (rb, cb)) = (rc(a), rc(b)); ra == rb && ca + cb == COLS - 1 };
    let distinct = |v: &[(u8, u8)]| { let mut s = v.to_vec(); s.sort_unstable(); s.dedup(); s.len() == v.len() };
    // 순서 있는 짝: 한 손 같은 줄 (5칸에서 2칸: 10가지 × 2손 × 3줄) + 좌우 대칭 (5가지 × 3줄). 첫 글자가 왼쪽·왼손
    let ordered = placements(true);
    assert!(distinct(&ordered) && ordered.len() == 60 + 15);
    for &(a, b) in &ordered { assert!((same_row(a, b) || mirror(a, b)) && rc(a).1 < rc(b).1, "{a} {b}"); }
    // 떨어진 자리도 됨 (같은 줄 새끼-가운데 열). 양손 가운데 열끼리는 좌우 대칭
    for p in [(at(0, 0), at(0, 4)), (at(2, 5), at(2, 9)), (at(1, 4), at(1, 5))] { assert!(ordered.contains(&p), "{p:?}"); }
    // 위아래(같은 열)와 거꾸로(오른쪽·오른손이 먼저)는 안 됨
    for p in [(at(0, 1), at(2, 1)), (at(1, 2), at(2, 2)), (at(0, 4), at(0, 0)), (at(1, 6), at(1, 3))] { assert!(!ordered.contains(&p), "{p:?}"); }
    // 순서 없는 짝: 위 자리 두 방향 + 한 손 같은 열 (3줄에서 2줄: 3가지 × 10열) 두 방향
    let loose = placements(false);
    assert!(distinct(&loose) && loose.len() == 2 * (60 + 30 + 15));
    for &(a, b) in &loose { assert!(same_row(a, b) || same_col(a, b) || mirror(a, b), "{a} {b}"); }
    assert!(ordered.iter().all(|&(a, b)| loose.contains(&(a, b)) && loose.contains(&(b, a))));
    for p in [(at(0, 1), at(2, 1)), (at(2, 1), at(0, 1))] { assert!(loose.contains(&p), "{p:?}"); }
    // 대각선이나 양손에 나뉜 대칭 아닌 자리는 어느 짝도 안 됨
    for p in [(at(0, 1), at(1, 2)), (at(1, 3), at(1, 7)), (at(0, 3), at(2, 6))] { assert!(!loose.contains(&p), "{p:?}"); }
  }

  // 예시 격자 (열 칸짜리 줄만, · = 다른 칸)에서 짝 두 기호의 칸 번호
  fn grid_places(grid: &str, pairs: &[&str]) -> Vec<(usize, usize)> {
    let rows: Vec<Vec<char>> = grid.lines().map(|l| l.split_whitespace().filter_map(|t| t.chars().next()).collect::<Vec<char>>())
      .filter(|r| r.len() == COLS).collect();
    let find = |ch: char| rows.iter().enumerate().find_map(|(r, row)| row.iter().position(|&c| c == ch).map(|c| r * COLS + c))
      .unwrap_or_else(|| panic!("{ch:?} is not in the grid"));
    pairs.iter().map(|p| { let cs: Vec<char> = p.chars().collect(); (find(cs[0]), find(cs[1])) }).collect()
  }

  // 예시 격자의 짝들이 모양마다 어긴 쌍 수 [가로, 세로, 대칭]
  fn broken(grid: &str, pairs: &[&str]) -> [usize; 3] { shape_violations(&grid_places(grid, pairs)) }

  #[test]
  fn pairs_line_up_by_shape() {
    let at = |r: usize, c: usize| r * COLS + c;
    // 모양: 한 손 가로 (오른손은 거울 열), 양손 대칭, 한 손 세로. 대각선은 짝 자리가 아님
    assert_eq!(pair_shape(at(1, 1), at(1, 2)), Some(Shape::Row { hand: 0, row: 1, cols: (1, 2) }));
    assert_eq!(pair_shape(at(2, 8), at(2, 7)), Some(Shape::Row { hand: 1, row: 2, cols: (1, 2) }));
    assert_eq!(pair_shape(at(0, 2), at(0, 7)), Some(Shape::Mirror { row: 0, col: 2 }));
    assert_eq!(pair_shape(at(2, 9), at(0, 9)), Some(Shape::Col { rows: (0, 2), col: 0 }));
    assert_eq!(pair_shape(at(0, 1), at(1, 2)), None);
    let none = [0, 0, 0];
    let row = |n: usize| [n, 0, 0];
    let mir = |n: usize| [0, 0, n];
    // 1. 한 손 가로 짝: 모두 같은 두 열 (줄과 손은 달라도 됨). 55.02 결과는 , . 와 \ / 가 괄호(약지·중지)와 열이 다름
    let rows = ["()", "{}", "[]", "\\/", ",."];
    assert_eq!(broken(r"
      · · · · ·   \ · · / ·
      · ( ) · ·   · . , · ·
      · { } · ·   · · [ ] ·", &rows), row(2));
    // 양쪽으로 한 칸씩 넓은 짝은 같은 줄에서 감쌀 때 됨 (\ / 가 , . 를, [ ] 가 ( ) 를)
    assert_eq!(broken(r"
      · · · · ·   · · · · ·
      · ( ) · ·   · \ , . /
      · { } · ·   · · [ ] ·", &rows), none);
    assert_eq!(broken("[ ( ) ] ·   · · · · ·", &["()", "[]"]), none);
    // 감쌀 짝 없이 다른 줄에 따로 있으면 안 됨
    assert_eq!(broken(r"
      · · · · ·   · \ · · /
      · ( ) · ·   · · , . ·
      · { } · ·   · · [ ] ·", &rows), row(1));
    // 열이 한 칸 어긋남. 같은 줄이어도 열이 다르면 안 됨
    assert_eq!(broken("
      · ( ) · ·   · . , · ·
      · { } · ·   · · [ ] ·", &["()", "{}", "[]", ",."]), row(1));
    assert_eq!(broken("· ( ) , .   · · · · ·", &["()", ",."]), row(1));
    // 2. 한 손 세로 짝: 모두 같은 두 줄 (열은 달라도 됨). 55.02 결과는 & * 만 1·3줄
    let cols = ["!?", "'\"", "&*"];
    assert_eq!(broken("
      ? · · ' ·   · · · · *
      ! · · \" ·   · · · · ·
      · · · · ·   · · · · &", &cols), [0, 1, 0]);
    assert_eq!(broken("
      ? · · ' ·   · · · · *
      ! · · \" ·   · · · · &
      · · · · ·   · · · · ·", &cols), none);
    // 3. 양손 대칭 짝: 모두 같은 열이거나 모두 같은 줄 (떨어져도 됨). 55.02 결과는 < > (중지·1줄)와 ; : (검지·3줄)가 다 다름
    assert_eq!(broken("
      · · < · ·   · · > · ·
      · · · · ·   · · · · ·
      · · · ; ·   · : · · ·", &["<>", ":;"]), mir(1));
    assert_eq!(broken("
      · · < · ·   · · > · ·
      · · · · ·   · · · · ·
      · · ; · ·   · · : · ·", &["<>", ":;"]), none);
    assert_eq!(broken("· · < ; ·   · : > · ·", &["<>", ":;"]), none);
    // 꺾인 모양은 바로 옆끼리 이어질 때만 (ㄱ자는 됨. 떨어져서 이어지거나 끊어지면 안 됨)
    let three = ["<>", ":;", "'\""];
    assert_eq!(broken("
      · · < ; ·   · : > · ·
      · · ' · ·   · · \" · ·", &three), none);
    assert_eq!(broken("
      < · · ; ·   · : · · >
      · · · ' ·   · \" · · ·", &three), mir(1));
    assert_eq!(broken("
      · · < ; ·   · : > · ·
      · ' · · ·   · · · \" ·", &three), mir(1));
    // 1·3줄로 떨어진 두 덩어리를 같은 열로 잇는 것도 꺾인 모양이라 안 됨 (덤 있음 56.04 결과의 대칭 짝)
    assert_eq!(broken(r"
      · [ < · ·   · · > ] ·
      · · · · ·   · · · · ·
      · { / ; ·   · : \ } ·", &["[]", "<>", "{}", "/\\", ":;"]), mir(2));
  }

  #[test]
  fn pairs_not_lined_up_are_reported() {
    let c = config();
    let (t, g) = (Tables::new(&c), ngrams());
    let base = Layout::parse(BASE).unwrap();
    let p = Problem::new(&t, &g, &base, &c.symbols).unwrap();
    // 55.02 결과의 심볼 레이어 (짝 하나하나는 규칙대로지만 모양끼리 맞지 않는 곳이 둘)
    let text = r#"
 ? - < ' @   \ = > / *
 ! ( ) " +   ` . , _ |
 # { } ; %   $ : [ ] &
 p l o u ^   q b d f k
 c t h e ~   x ␣ s n r
 w g j a ·   z i v m y
"#;
    let (_, placed) = parse_text(text).unwrap();
    let st = p.state_of(&placed).unwrap();
    assert_eq!(p.violations(&st), 2);
    // 저장된 결과가 이러면 다음 symbols의 출발점으로 쓰지 않음
    assert_eq!(p.broken_pairs(&st), vec![
      "() [] {} /\\ (one-hand row pairs not lined up)".to_string(),
      "<> :; (mirrored pairs not lined up)".to_string(),
    ]);
  }

  #[test]
  fn search_keeps_pairs_and_rarest_symbols_go_to_the_base() {
    fastrand::seed(11);
    let mut c = config();
    c.sa.starts_per_batch = 4;
    c.sa.stop_after_hits = 1;
    c.sa.max_starts = 4;
    let (t, g) = (Tables::new(&c), ngrams());
    // 모든 기호가 코퍼스에 있음 (빈도 0인 기호는 어디에 둬도 비용이 같아서 빈칸 표시와 자리를 바꿔도 같은 점수)
    assert!((0..NSYM).all(|i| g.uni[FIRST_SYM as usize + i] > 0.0));
    let base = Layout::parse(BASE).unwrap();
    let p = Problem::new(&t, &g, &base, &c.symbols).unwrap();
    // 짝 맞춤 규칙이 짝 열 개 모두에 걸려서 짧은 탐색도 이동이 꽤 필요함 (2만 번이면 시작 넷이 모두 규칙을 못 지키는 일이
    // 시드 20개 중 9개, 20만 번이면 시드 100개 중 0개)
    let (cost, st, _, starts) = p.search(&c.sa, 200_000, None);
    assert_eq!(starts, 4);
    assert!((cost - p.total(&st)).abs() < 1e-9);
    // 짝은 모두 규칙대로 (짝마다 놓을 수 있는 자리에, 모양이 같은 짝끼리 맞춰서)
    assert!(p.broken_pairs(&st).is_empty());
    // 베이스 빈칸에는 코퍼스에서 가장 드문 기호만. 심볼 레이어가 2칸 모자라므로 2개 이상이고, 심볼 레이어에 남는 칸이
    // 어떤 기호에도 나쁜 자리(예: 스페이스와 같은 손가락)면 그 칸을 비우고 3개를 둘 수도 있음
    let on_base: Vec<usize> = (0..NSYM).filter(|&i| !p.slots[st.slot_of[i] as usize].tap).collect();
    assert!((2..=3).contains(&on_base.len()), "{on_base:?}");
    let freq = |i: usize| g.uni[FIRST_SYM as usize + i];
    let mut sorted: Vec<f64> = (0..NSYM).map(freq).collect();
    sorted.sort_by(f64::total_cmp);
    assert!(on_base.iter().all(|&i| freq(i) <= sorted[NSYM / 4]), "{on_base:?}");
    // 근처 탐색으로 더 좋아지지 않음
    let (again, _) = p.polish(st.clone(), &mut p.work());
    assert!((again - cost).abs() < 1e-9);
  }

  #[test]
  fn text_roundtrip_and_parse_errors() {
    let c = config();
    let (t, g) = (Tables::new(&c), ngrams());
    let base = Layout::parse(BASE).unwrap();
    let p = Problem::new(&t, &g, &base, &c.symbols).unwrap();
    let st = p.random_state(&mut fastrand::Rng::with_seed(3));
    let text = p.to_text(&base, &st);
    let (b2, placed) = parse_text(&text).unwrap();
    assert_eq!(b2.slots, base.slots);
    let st2 = p.state_of(&placed).unwrap();
    assert!((p.total(&st2) - p.total(&st)).abs() < 1e-9);
    assert_eq!(p.to_text(&base, &st2), text);
    // 기호가 빠지거나 두 번 나오면 거부
    let dup = text.replacen(EMPTY_TOKEN, "(", 1);
    assert!(parse_text(&dup).is_err());
    let lines: Vec<&str> = text.lines().collect();
    assert!(parse_text(&lines[..4].join("\n")).is_err());
  }
}
