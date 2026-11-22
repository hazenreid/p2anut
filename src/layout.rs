#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use crate::score::{Score, ScoreNorm};
use crate::corpus::CorpusArrays;
use crate::precomp::Precomp;

pub fn fmt_num(n: i64) -> String {
  let s = n.unsigned_abs().to_string();
  let with_commas = s.as_bytes().rchunks(3)
    .rev()
    .map(|c| std::str::from_utf8(c).unwrap())
    .collect::<Vec<_>>()
    .join(",");
  if n < 0 { format!("-{with_commas}") } else { with_commas }
}

pub const EMPTY: u8 = 255;
pub const CHARS: usize = 27;  // a-z + space(26)
pub const ROWS: usize = 3;
pub const COLS: usize = 10;
// 손 사용 비율 위반 표시. 정상 총점(-(글자당 비용) × ABS_SCALE)은 이 값보다 항상 크게 잘림 (score::total_score)
pub const PENALTY_SCORE: i64 = i64::MIN / 2;


// 가운데 두 열(4·5열, 검지 안쪽)
pub fn is_center_slot(p: usize) -> bool { matches!(p % COLS, 4 | 5) }

pub fn home_row_keys(layout: &Layout) -> u32 {
  let mut mask = 0u32;
  let home = COLS;
  // 양손 비중심 홈행 키 (중앙 두 열 제외)
  for c in (0..COLS / 2 - 1).chain(COLS / 2 + 1..COLS) {
    let id = layout.slots[home + c];
    if id < CHARS as u8 { mask |= 1 << id; }
  }
  mask
}

// 두 홈행 마스크 사이에 서로 다른 글자 수 (빈칸으로 크기가 달라도 큰 쪽 기준)
pub fn home_diff(a: u32, b: u32) -> u32 {
  (a & !b).count_ones().max((b & !a).count_ones())
}

// h가 others 모두와 홈행 글자가 min_diff개 이상 다른지
pub fn is_home_distinct(h: u32, others: &[u32], min_diff: u32) -> bool {
  others.iter().all(|&o| home_diff(h, o) >= min_diff)
}


#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Layout {
  // mapping: slot_index -> letter_idx (0..27, 255 if empty)
  pub slots: [u8; ROWS * COLS],
  // mapping: letter_idx (0..27) -> slot_index (0..30)
  pub pos: [u8; CHARS],
  pub score: Score,
  pub total: i64,
  pub left_usage: u64,
  pub right_usage: u64,
}

impl Layout {
  pub fn new() -> Self {
    Self {
      slots: [EMPTY; ROWS * COLS],
      pos: [EMPTY; CHARS],
      score: Score::default(),
      total: 0,
      left_usage: 0,
      right_usage: 0,
    }
  }

  pub fn make_random(keys: &[char], excluded_homes: &[u32], min_diff: u32) -> Self {
    const SLOTS: usize = ROWS * COLS;
    let nk = keys.len();
    let mut items = [EMPTY; SLOTS];
    for (i, item) in items.iter_mut().take(nk).enumerate() { *item = i as u8; }
    loop {
      for i in (1..SLOTS).rev() { let j = fastrand::usize(..=i); items.swap(i, j); }
      let mut layout = Layout::new();
      layout.slots = items;
      for (p, &id) in items.iter().enumerate() {
        if id < CHARS as u8 { layout.pos[id as usize] = p as u8; }
      }
      let h = home_row_keys(&layout);
      if is_home_distinct(h, excluded_homes, min_diff) {
        return layout;
      }
    }
  }

  // center_mask 글자가 모두 가운데 두 열에 있고, 그 밖의 글자는 가운데에 없는지 (0 = 제약 없음)
  pub fn center_ok(&self, center_mask: u32) -> bool {
    center_mask == 0 || self.slots.iter().enumerate().all(|(p, &id)| {
      id >= CHARS as u8 || is_center_slot(p) == (center_mask >> id & 1 == 1)
    })
  }

  // make_random과 같지만 center_mask 글자는 가운데 칸에, 나머지는 바깥 칸에만 무작위 배치
  pub fn make_random_centered(keys: &[char], center_mask: u32, excluded_homes: &[u32], min_diff: u32) -> Self {
    if center_mask == 0 { return Self::make_random(keys, excluded_homes, min_diff); }
    let (inner, outer): (Vec<usize>, Vec<usize>) = (0..ROWS * COLS).partition(|&p| is_center_slot(p));
    loop {
      let mut layout = Layout::new();
      let mut ins: Vec<u8> = vec![EMPTY; inner.len()];
      let mut outs: Vec<u8> = vec![EMPTY; outer.len()];
      let (mut ni, mut no) = (0, 0);
      for i in 0..keys.len() as u8 {
        if center_mask >> i & 1 == 1 { ins[ni] = i; ni += 1; } else { outs[no] = i; no += 1; }
      }
      fastrand::shuffle(&mut ins);
      fastrand::shuffle(&mut outs);
      for (&p, &id) in inner.iter().zip(&ins).chain(outer.iter().zip(&outs)) {
        layout.slots[p] = id;
        if id < CHARS as u8 { layout.pos[id as usize] = p as u8; }
      }
      if is_home_distinct(home_row_keys(&layout), excluded_homes, min_diff) { return layout; }
    }
  }

  // 배열 텍스트 읽기: 행마다 10칸, 글자는 a-z, 빈칸은 '.' 또는 '_'(예전 출력). 공백은 무시 (print() 출력도 그대로 읽힘).
  // 빈 줄은 건너뛰고, 정확히 3행이어야 하며 각 글자는 한 번씩만
  pub fn parse(text: &str) -> anyhow::Result<Self> {
    let rows: Vec<Vec<char>> = text.lines()
      .map(|l| l.chars().filter(|c| !c.is_whitespace()).collect::<Vec<char>>())
      .filter(|r| !r.is_empty())
      .collect();
    anyhow::ensure!(rows.len() == ROWS, "layout must have {ROWS} rows, found {}", rows.len());
    let mut l = Layout::new();
    for (r, row) in rows.iter().enumerate() {
      anyhow::ensure!(row.len() == COLS, "row {} must have {COLS} keys, found {}: {}", r + 1, row.len(), row.iter().collect::<String>());
      for (c, &ch) in row.iter().enumerate() {
        let ch = ch.to_ascii_lowercase();
        if ch == '.' || ch == '_' { continue; }
        anyhow::ensure!(ch.is_ascii_lowercase(), "unexpected character {ch:?} in row {}", r + 1);
        let id = ch as u8 - b'a';
        anyhow::ensure!(l.pos[id as usize] == EMPTY, "letter {ch} appears more than once");
        let p = r * COLS + c;
        l.slots[p] = id;
        l.pos[id as usize] = p as u8;
      }
    }
    Ok(l)
  }

  // 사람이 읽기 쉬운 비율 지표 (corpus 기준, 0~100%). 분모는 각각 전체 글자·바이그램·트라이그램 빈도.
  // windows: 공백(id 26, 엄지)까지 포함한 모든 트라이그램 창 (CorpusStats::windows). 커뮤니티 분석기식 창 기준 통계에 씀
  pub fn print_stats(&self, corpus: &CorpusArrays, precomp: &Precomp, windows: &[(u8, u8, u8, u64)]) {
    let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 * 100.0 / b as f64 };
    let mut rows = [0u64; ROWS];
    let mut letters = 0u64;
    for (p, &id) in self.slots.iter().enumerate() {
      if id < CHARS as u8 { rows[p / COLS] += corpus.freq[id as usize]; letters += corpus.freq[id as usize]; }
    }
    let (mut bg, mut sfb, mut roll, mut sciss) = (0u64, 0u64, 0u64, 0u64);
    for &(a, b, n) in &corpus.nz_bigrams {
      let (p1, p2) = (self.pos[a as usize], self.pos[b as usize]);
      bg += n;
      if p1 == EMPTY || p2 == EMPTY { continue; }
      let e = precomp.bigram(p1 as usize, p2 as usize);
      if e.sfb_base > 0.0 { sfb += n; }
      if e.rolling_base > 0.0 { roll += n; }
      if e.scissors_base > 0.0 { sciss += n; }
    }
    let (mut tg, mut sfs, mut redirect, mut roll3) = (0u64, 0u64, 0u64, 0u64);
    for &(a, b, c, n) in &corpus.nz_trigrams {
      let (p1, p2, p3) = (self.pos[a as usize], self.pos[b as usize], self.pos[c as usize]);
      tg += n;
      if p1 == EMPTY || p2 == EMPTY || p3 == EMPTY { continue; }
      let e = precomp.trigram(p1 as usize, p2 as usize, p3 as usize);
      if e.sfs_base > 0.0 && p1 != p3 { sfs += n; } // 같은 키 스킵그램(ere)은 배열로 피할 수 없어 빼고 셈
      if e.roll_base < 0.0 { redirect += n; }
      if e.roll_base > 0.0 { roll3 += n; } // 세 글자 롤
    }
    println!("rows: top={:.1}%  home={:.1}%  bottom={:.1}%", pct(rows[0], letters), pct(rows[1], letters), pct(rows[2], letters));
    println!("bigrams: sfb={:.2}%  roll={:.1}%  scissors={:.2}%   trigrams: skipgram(other key)={:.2}%  one-hand-roll={:.1}%  redirect={:.2}%",
      pct(sfb, bg), pct(roll, bg), pct(sciss, bg), pct(sfs, tg), pct(roll3, tg), pct(redirect, tg));
    self.print_window_stats(precomp, windows);
  }

  // 창 기준 통계: 세 키(공백은 엄지)를 하나씩 밀며 나눈 분류의 비율. 스킵그램은 다른 분류와 겹쳐서 따로 셈
  fn print_window_stats(&self, precomp: &Precomp, windows: &[(u8, u8, u8, u64)]) {
    use crate::precomp::{finger, Window};
    let at = |id: u8| -> Option<Option<usize>> {
      if id as usize == crate::SPACE_ID as usize { return Some(None); } // 엄지
      let p = self.pos[id as usize];
      (p != EMPTY).then_some(Some(p as usize))
    };
    let kinds = [Window::Alternate, Window::Roll2In, Window::Roll2Out, Window::Roll3In, Window::Roll3Out,
      Window::Redirect, Window::BadRedirect, Window::SameHandOther];
    let mut counts = [0u64; 8];
    let (mut total, mut skip, mut skip_space) = (0u64, 0u64, 0u64);
    for &(a, b, c, n) in windows {
      let (Some(p1), Some(p2), Some(p3)) = (at(a), at(b), at(c)) else { continue };
      total += n;
      let w = precomp.window(p1, p2, p3);
      counts[kinds.iter().position(|&k| k == w).unwrap_or(0)] += n;
      // 1·3번이 같은 손가락의 다른 키 (가운데는 다른 손가락이나 엄지)
      if let (Some(x), Some(z)) = (p1, p3) && x != z && finger(x) == finger(z) && p2.is_none_or(|y| finger(y) != finger(x)) {
        skip += n;
        if p2.is_none() { skip_space += n; }
      }
    }
    let pct = |a: u64| if total == 0 { 0.0 } else { a as f64 * 100.0 / total as f64 };
    println!("windows (space = thumb): alternate={:.1}%  roll2 in={:.1}% out={:.1}%  roll3 in={:.1}% out={:.1}%  redirect={:.2}%  bad-redirect={:.2}%  same-hand-other={:.1}%  skipgram={:.2}% (across space {:.2}%)",
      pct(counts[0]), pct(counts[1]), pct(counts[2]), pct(counts[3]), pct(counts[4]), pct(counts[5]), pct(counts[6]), pct(counts[7]),
      pct(skip), pct(skip_space));
  }

  pub fn key_count(&self) -> usize {
    self.slots.iter().filter(|&&id| id < CHARS as u8).count()
  }

  pub fn calc_scores(&mut self, corpus: &CorpusArrays, precomp: &Precomp) {
    let mid = COLS / 2;
    let mut left_usage = 0u64;
    let mut right_usage = 0u64;

    for p in 0..(ROWS * COLS) {
      let id = self.slots[p];
      if id < CHARS as u8 {
        let freq = corpus.freq[id as usize];
        if (p % COLS) < mid { left_usage += freq; } else { right_usage += freq; }
      }
    }

    // 페널티 배열도 score/usage는 실제 값으로 채워야 delta 이동(swap2/3)이 올바르게 동작함
    let penalized = precomp.violates_hand_usage(left_usage, right_usage);

    self.left_usage = left_usage;
    self.right_usage = right_usage;

    let mut effort = 0.0f64;
    for p in 0..(ROWS * COLS) {
      let id = self.slots[p];
      if id < CHARS as u8 {
        effort += precomp.effort(p, id as usize);
      }
    }

    let mut sfb = 0.0f64;
    let mut rolling = 0.0f64;
    let mut scissors = 0.0f64;

    for &(a, b, count) in &corpus.nz_bigrams {
      let p1 = self.pos[a as usize];
      let p2 = self.pos[b as usize];
      if p1 == EMPTY || p2 == EMPTY { continue; }
      let e = precomp.bigram(p1 as usize, p2 as usize);
      sfb      += count as f64 * f64::from(e.sfb_base);
      rolling  += count as f64 * f64::from(e.rolling_base);
      scissors += count as f64 * f64::from(e.scissors_base);
    }

    for &(a, b, c, count) in &corpus.nz_trigrams {
      let p1 = self.pos[a as usize];
      let p2 = self.pos[b as usize];
      let p3 = self.pos[c as usize];
      if p1 == EMPTY || p2 == EMPTY || p3 == EMPTY { continue; }
      let e = precomp.trigram(p1 as usize, p2 as usize, p3 as usize);
      sfb      += count as f64 * f64::from(e.sfs_base);
      rolling  += count as f64 * f64::from(e.roll_base);
      scissors += count as f64 * f64::from(e.skip_scissors_base);
    }

    // 공백 건너 쌍 (a _ c): 공백 너머 스킵그램·스킵 시저스
    for &(a, c, count) in &corpus.nz_cross {
      let p1 = self.pos[a as usize];
      let p3 = self.pos[c as usize];
      if p1 == EMPTY || p3 == EMPTY { continue; }
      let e = precomp.cross(p1 as usize, p3 as usize);
      sfb      += count as f64 * f64::from(e.sfs_base);
      scissors += count as f64 * f64::from(e.skip_scissors_base);
    }

    self.score = Score {
      effort,
      sfb,
      rolling,
      scissors,
    };
    // 이전 PENALTY 값이 남아 calc_total_score가 건너뛰지 않도록 항상 초기화
    self.total = if penalized { PENALTY_SCORE } else { 0 };
  }

  pub fn print(&self) {
    for r in 0..ROWS {
      let mut s = String::new();
      for c in 0..COLS {
        let char_idx = self.slots[r * COLS + c];
        let ch = if char_idx < 26 { (b'a' + char_idx) as char } else { '.' };
        if c == 5 { s.push_str("  "); }
        s.push(' ');
        s.push(ch);
      }
      println!("{s}");
    }
    let total_usage = (self.left_usage + self.right_usage) as f32;
    let (left_pct, right_pct) = if total_usage > 0.0 {
      (self.left_usage as f32 / total_usage * 100.0, self.right_usage as f32 / total_usage * 100.0)
    } else {
      (0.0, 0.0)
    };
    println!("effort={}  sfb={}  rolling={}  scissors={}  total={}",
      fmt_num(self.score.effort as i64), fmt_num(self.score.sfb as i64),
      fmt_num(self.score.rolling as i64), fmt_num(self.score.scissors as i64),
      fmt_num(self.total));
    println!("left={left_pct:.1}%  right={right_pct:.1}%");
  }

  // 글자 100개당 항목별 비용과 무작위 배열 대비 변화율
  pub fn print_goodness(&self, norm: &ScoreNorm) {
    if self.total <= PENALTY_SCORE {
      println!("DISQUALIFIED: one hand types less than hand_hard_limit (scores below are for reference only)");
    }
    let c = norm.cost_breakdown(&self.score);
    let hand = norm.hand_cost(self.left_usage, self.right_usage);
    let total = c.effort + c.sfb + c.rolling + c.scissors + hand;
    let hand_part = if hand > 0.0 { format!("  hand-balance={hand:.1}") } else { String::new() };
    println!("cost per 100 letters (lower is better): total={total:.1}  effort={:.1}  sfb={:.1}  scissors={:.1}  rolling={:.1}{hand_part}", c.effort, c.sfb, c.scissors, c.rolling);
    let v = norm.vs_random(&self.score);
    println!("vs random median: effort {:+.1}%  sfb {:+.1}%  scissors {:+.1}%  rolling {:+.1}%", v.effort, v.sfb, v.scissors, v.rolling);
  }

  pub fn calc_total_score(&mut self, norm: &ScoreNorm) {
    if self.total <= PENALTY_SCORE {
      return; // Hand usage penalty already applied
    }
    self.total = norm.total_score(
      self.score.effort,
      self.score.sfb,
      self.score.rolling,
      self.score.scissors,
      self.left_usage,
      self.right_usage,
    );
  }

  #[cfg(any(test, debug_assertions))]
  pub fn assert_invariants(&self) {
    for (p, &id) in self.slots.iter().enumerate() {
      if id < CHARS as u8 {
        assert_eq!(self.pos[id as usize], p as u8, "Pos mismatch for char {id}");
      }
    }
    for (id, &p) in self.pos.iter().enumerate() {
      if p != EMPTY {
        assert_eq!(self.slots[p as usize], id as u8, "Slot mismatch for pos {p}");
      }
    }
  }
}

// 쓰는 도중에 강제 종료돼도 기존 결과가 깨지지 않게 임시 파일에 쓰고 바꿔치기
pub fn save_results(layouts: &[Layout], path: &str) -> anyhow::Result<()> {
  let bytes = bincode::serialize(layouts)?;
  let tmp = format!("{path}.tmp");
  std::fs::write(&tmp, bytes)?;
  std::fs::rename(&tmp, path)?;
  Ok(())
}

pub fn load_results(path: &str) -> anyhow::Result<Vec<Layout>> {
  let bytes = std::fs::read(path)?;
  Ok(bincode::deserialize(&bytes)?)
}

// 총점 기준: [costs]의 대표 사건 비용을 환산한 배율과 손 비율 감점. 무작위 배열 표본(center_mask 제약을 지킴)의
// 항목별 중앙값은 "무작위 배열 대비" 표시에만 씀
pub fn init_score_norm(arrays: &CorpusArrays, precomp: &Precomp, costs: &crate::config::Costs, hand: crate::score::HandRule, center_mask: u32) -> ScoreNorm {
  const N: usize = 10001;
  let mut effort_vals   = Vec::with_capacity(N);
  let mut sfb_vals      = Vec::with_capacity(N);
  let mut rolling_vals  = Vec::with_capacity(N);
  let mut scissors_vals = Vec::with_capacity(N);

  while effort_vals.len() < N {
    let mut l = Layout::make_random_centered(&arrays.keys, center_mask, &[], 0);
    l.calc_scores(arrays, precomp);
    if l.total <= PENALTY_SCORE { continue; }
    effort_vals.push(l.score.effort as f32);
    sfb_vals.push(l.score.sfb as f32);
    rolling_vals.push(l.score.rolling as f32);
    scissors_vals.push(l.score.scissors as f32);
  }

  ScoreNorm {
    median: ScoreNorm::medians(&effort_vals, &sfb_vals, &rolling_vals, &scissors_vals),
    abs: abs_weights(arrays, precomp, costs),
    hand,
  }
}

// 대표 사건 (왼손): 검지 홈(1,3)·검지 윗줄(0,3)·중지 홈(1,2)·중지 아랫줄(2,2). 롤·리다이렉트의 대표 사건은 Precomp에 있음
const REF_INDEX_HOME: usize = COLS + 3;
const REF_INDEX_TOP: usize = 3;
#[cfg(test)]
const REF_MIDDLE_HOME: usize = COLS + 2;
const REF_MIDDLE_BOTTOM: usize = 2 * COLS + 2;

// [costs]의 대표 사건 비용을 원래 점수 1단위당 비용으로 환산
pub fn abs_weights(arrays: &CorpusArrays, precomp: &Precomp, c: &crate::config::Costs) -> crate::score::AbsWeights {
  let per = |cost: f32, base: f32| if base > 0.0 { f64::from(cost) / f64::from(base) } else { 0.0 };
  let letters: u64 = arrays.freq.iter().sum();
  crate::score::AbsWeights {
    effort: f64::from(c.effort),
    sfb: per(c.sfb, precomp.bigram(REF_INDEX_HOME, REF_INDEX_TOP).sfb_base),
    scissors: per(c.full_scissors, precomp.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).scissors_base),
    // 롤·리다이렉트는 Precomp가 이미 비용 단위로 계산함 (대표 안쪽 롤 = inroll)
    rolling: 1.0,
    per_letter: if letters > 0 { 1.0 / letters as f64 } else { 0.0 },
  }
}

fn layout_key(l: &Layout) -> impl Ord {
  let ci = |v: f64| v.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
  (
    l.total,
    ci(l.score.rolling * 1e6),
    ci(-l.score.sfb * 1e6),
    ci(-l.score.scissors * 1e6),
    ci(-l.score.effort * 1e6),
  )
}

pub fn sort_layouts(layouts: &mut [Layout]) {
  layouts.sort_by(|a, b| layout_key(b).cmp(&layout_key(a)));
}

// 정렬 후 홈행(양손 비중심 키)이 서로, 그리고 excluded_homes와 min_diff글자 이상 다른 레이아웃만 남김
pub fn sort_unique_layouts(mut layouts: Vec<Layout>, size: usize, excluded_homes: &[u32], min_diff: u32) -> Vec<Layout> {
  sort_layouts(&mut layouts);
  if layouts.is_empty() || size == 0 { return vec![]; }

  // P-22: In-place filtering to reduce allocations
  let mut seen_masks: Vec<u32> = Vec::with_capacity(size + excluded_homes.len());
  seen_masks.extend_from_slice(excluded_homes);

  layouts.retain(|l| {
    if seen_masks.len() >= size + excluded_homes.len() { return false; }
    let h = home_row_keys(l);
    // min_diff=0이어도 완전히 같은 홈행은 중복으로 처리
    if !is_home_distinct(h, &seen_masks, min_diff.max(1)) { return false; }
    seen_masks.push(h);
    true
  });

  layouts.truncate(size);
  layouts
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_util;

  fn identity_layout() -> Layout {
    let mut l = Layout::new();
    for p in 0..CHARS { l.slots[p] = p as u8; l.pos[p] = p as u8; }
    l
  }

  #[test]
  fn home_row_keys_covers_both_hands_without_center() {
    let l = identity_layout();
    // 홈행 = slot 10..20, 중앙 열(14, 15) 제외
    let expect: u32 = [10, 11, 12, 13, 16, 17, 18, 19].iter().map(|&b| 1u32 << b).sum();
    assert_eq!(home_row_keys(&l), expect);
    let mut r = l; r.slots.swap(17, 7);
    assert_ne!(home_row_keys(&r), expect);
  }

  #[test]
  fn home_diff_counts_differing_letters() {
    assert_eq!(home_diff(0b1111, 0b1111), 0);
    assert_eq!(home_diff(0b1111, 0b0111), 1);
    assert_eq!(home_diff(0b0011, 0b1100), 2);
    assert!(is_home_distinct(0b0011, &[0b1100, 0b0111_0000], 2));
    assert!(!is_home_distinct(0b0011, &[0b0110], 2));
  }

  #[test]
  fn cross_space_pairs_add_to_sfb_and_scissors_only() {
    let arrays = test_util::stats().to_arrays();
    assert!(!arrays.nz_cross.is_empty());
    let mut no_cross = arrays.clone();
    no_cross.nz_cross.clear();
    let (precomp, _) = test_util::precomp_norm_for(&arrays);
    for _ in 0..20 {
      let base = Layout::make_random(&arrays.keys, &[], 0);
      let (mut with, mut without) = (base, base);
      with.calc_scores(&arrays, &precomp);
      without.calc_scores(&no_cross, &precomp);
      let (mut s, mut k) = (0.0, 0.0);
      for &(a, c, n) in &arrays.nz_cross {
        let e = precomp.cross(base.pos[a as usize] as usize, base.pos[c as usize] as usize);
        s += n as f64 * f64::from(e.sfs_base);
        k += n as f64 * f64::from(e.skip_scissors_base);
      }
      assert!((with.score.sfb - without.score.sfb - s).abs() < 1e-6);
      assert!((with.score.scissors - without.score.scissors - k).abs() < 1e-6);
      assert_eq!((with.score.effort, with.score.rolling), (without.score.effort, without.score.rolling));
    }
  }

  #[test]
  fn sort_unique_respects_min_diff() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let layouts: Vec<Layout> = (0..50).map(|_| {
      let mut l = Layout::make_random(&arrays.keys, &[], 0);
      l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm); l
    }).collect();
    let kept = sort_unique_layouts(layouts, 10, &[], 3);
    for i in 0..kept.len() { for j in i + 1..kept.len() {
      assert!(home_diff(home_row_keys(&kept[i]), home_row_keys(&kept[j])) >= 3);
    }}
    assert!(kept.windows(2).all(|w| w[0].total >= w[1].total));
  }

  #[test]
  fn parse_reads_files_and_print_output() {
    let l = Layout::parse(".blyx.foup\nmanijzreht\ncgkdq.s.vw\n").unwrap();
    assert_eq!(l.key_count(), 26);
    l.assert_invariants();
    assert_eq!(l.slots[1], b'b' - b'a');
    assert_eq!(l.slots[COLS], b'm' - b'a');
    assert_eq!(l.slots[5], EMPTY);
    // print() 형식(공백, 가운데 간격, '_')도 읽힘
    let printed = Layout::parse(" . b l y x   _ f o u p\n m a n i j   z r e h t\n\n c g k d q   . s . v w").unwrap();
    assert_eq!(printed.slots, l.slots);
    assert!(Layout::parse("abc").is_err());
    assert!(Layout::parse(".blyx.foup\nmanijzreht\ncgkdq.s.va\n").err().unwrap().to_string().contains("more than once"));
    assert!(Layout::parse(".blyx.foup\nmanijzreht\ncgkdq.s.v\n").is_err());
  }

  #[test]
  fn absolute_costs_match_reference_events() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let (precomp, _) = test_util::precomp_norm_for(&arrays);
    let c = crate::config::Costs::default();
    let a = abs_weights(&arrays, &precomp, &c);
    // 대표 사건 하나의 원래 점수 × 환산 배율 = 설정한 비용
    let sfb = f64::from(precomp.bigram(REF_INDEX_HOME, REF_INDEX_TOP).sfb_base) * a.sfb;
    let sci = f64::from(precomp.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).scissors_base) * a.scissors;
    let roll = f64::from(precomp.bigram(REF_MIDDLE_HOME, REF_INDEX_HOME).rolling_base) * a.rolling;
    assert!((sfb - 2.5).abs() < 1e-6 && (sci - 8.0).abs() < 1e-6 && (roll - 2.0).abs() < 1e-6, "{sfb} {sci} {roll}");
    // 대표 사건이 실제로 그 종류인지
    assert!(precomp.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).rolling_base == 0.0);
    assert!(precomp.bigram(REF_MIDDLE_HOME, REF_INDEX_HOME).scissors_base == 0.0);
    assert!((a.per_letter * arrays.freq.iter().sum::<u64>() as f64 - 1.0).abs() < 1e-12);
  }

  #[test]
  fn changing_effort_keeps_reference_event_costs() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let costs = crate::config::Costs::default();
    let mut effort = crate::config::EffortGrid::default();
    let base = Precomp::new(&arrays.freq, &test_util::weights(), &costs, &effort, 0.45);
    // 검지 윗줄과 중지 홈을 더 힘들게
    effort.top[3] = 2.0;
    effort.home[2] = 1.4;
    let pc = Precomp::new(&arrays.freq, &test_util::weights(), &costs, &effort, 0.45);
    let a = abs_weights(&arrays, &pc, &costs);
    // 대표 사건 비용은 그대로
    let sfb = f64::from(pc.bigram(REF_INDEX_HOME, REF_INDEX_TOP).sfb_base) * a.sfb;
    let sci = f64::from(pc.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).scissors_base) * a.scissors;
    let roll = f64::from(pc.bigram(REF_MIDDLE_HOME, REF_INDEX_HOME).rolling_base) * a.rolling;
    assert!((sfb - f64::from(costs.sfb)).abs() < 1e-5 && (sci - f64::from(costs.full_scissors)).abs() < 1e-5 && (roll - f64::from(costs.inroll)).abs() < 1e-5, "{sfb} {sci} {roll}");
    // 노력값 비용은 바뀜 (왼손·오른손 모두)
    let e = b'e' - b'a';
    assert!((pc.effort(3, e as usize) / base.effort(3, e as usize) - 2.0 / 1.3).abs() < 1e-6);
    assert!((pc.effort(COLS + 7, e as usize) / base.effort(COLS + 7, e as usize) - 1.4).abs() < 1e-6);
    // 대표 사건이 아닌 사건은 노력값 합에 따라 상대적으로 바뀜 (검지 윗줄 → 중지 홈 하프 시저스)
    assert!(pc.bigram(REF_INDEX_TOP, REF_MIDDLE_HOME).scissors_base > base.bigram(REF_INDEX_TOP, REF_MIDDLE_HOME).scissors_base);
  }

  #[test]
  fn absolute_mode_ranks_by_lower_cost() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let (precomp, _) = test_util::precomp_norm_for(&arrays);
    let norm = init_score_norm(&arrays, &precomp, &crate::config::Costs::default(), crate::score::HandRule::default(), 0);
    let mut ls: Vec<Layout> = (0..30).map(|_| {
      let mut l = Layout::make_random(&arrays.keys, &[], 0);
      l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm); l
    }).filter(|l| l.total > PENALTY_SCORE).collect();
    sort_layouts(&mut ls);
    let cost = |l: &Layout| { let c = norm.cost_breakdown(&l.score); c.effort + c.sfb + c.rolling + c.scissors };
    assert!(ls.len() > 5);
    assert!(ls.windows(2).all(|w| cost(&w[0]) <= cost(&w[1]) + 1e-6));
    // 총점 = -(글자당 비용) × ABS_SCALE
    let l = &ls[0];
    assert!(((-l.total as f64) / crate::score::ABS_SCALE * 100.0 - cost(l)).abs() < 1e-4);
  }

  #[test]
  fn soft_hand_limit_penalizes_instead_of_disqualifying() {
    let arrays = test_util::stats().to_arrays();
    let (precomp, norm) = test_util::precomp_norm_soft_hand(&arrays);
    let (mut soft_seen, mut hard_seen) = (false, false);
    for _ in 0..3000 {
      let mut l = Layout::make_random(&arrays.keys, &[], 0);
      l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm);
      let share = l.left_usage.min(l.right_usage) as f64 / (l.left_usage + l.right_usage) as f64;
      let hand = norm.hand_cost(l.left_usage, l.right_usage);
      if share < 0.40 {
        assert!(l.total <= PENALTY_SCORE, "share {share}");
        hard_seen = true;
      } else {
        assert!(l.total > PENALTY_SCORE, "share {share}");
        // 45% 아래는 감점이 붙고, 총점에 그대로 들어감
        assert_eq!(hand > 0.0, share < 0.45, "share {share}");
        let s = &l.score;
        assert_eq!(l.total, norm.total_score(s.effort, s.sfb, s.rolling, s.scissors, l.left_usage, l.right_usage));
        if share < 0.45 {
          soft_seen = true;
          let without = norm.total_score(s.effort, s.sfb, s.rolling, s.scissors, 1, 1);
          assert!(l.total < without);
        }
      }
    }
    assert!(soft_seen && hard_seen, "soft {soft_seen} hard {hard_seen}");
  }

  #[test]
  fn save_results_roundtrip_leaves_no_temp_file() {
    let dir = std::env::temp_dir().join(format!("p2anut-results-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("result.bin");
    let path = path.to_str().unwrap();
    let ls = vec![Layout::make_random(&test_util::stats().to_arrays().keys, &[], 0); 2];
    save_results(&ls, path).unwrap();
    save_results(&ls[..1], path).unwrap();
    assert_eq!(load_results(path).unwrap().len(), 1);
    assert!(!std::path::Path::new(&format!("{path}.tmp")).exists());
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn penalized_layout_keeps_real_scores_and_recovers() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let (mut checked, mut sides) = (0, [false; 2]);
    for _ in 0..1000 {
      let mut l = Layout::make_random(&arrays.keys, &[], 0);
      l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm);
      let share = l.left_usage as f32 / (l.left_usage + l.right_usage) as f32;
      // 양손 대칭 제약: 어느 쪽이든 45% 미만이면 페널티
      assert_eq!(l.total <= PENALTY_SCORE, !(0.45..=0.55).contains(&share), "left share {share}");
      if l.total > PENALTY_SCORE {
        // 이전 PENALTY 값이 남아 있어도 다시 계산하면 풀려야 함
        let mut m = l;
        m.total = PENALTY_SCORE;
        m.calc_scores(&arrays, &precomp); m.calc_total_score(&norm);
        assert_eq!(m.total, l.total);
        continue;
      }
      checked += 1;
      sides[usize::from(share > 0.5)] = true;
      assert!(l.left_usage + l.right_usage > 0 && l.score.effort > 0.0);
    }
    assert!(sides[0] && sides[1], "both left-heavy and right-heavy layouts should be penalized");
    assert!(checked > 0);
  }
}
