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
pub const PENALTY_SCORE: i64 = -1_000_000_000_000;


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
        let ch = if char_idx < 26 {
          (b'a' + char_idx) as char
        } else if char_idx == 26 {
          '_'
        } else {
          '.'
        };
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

  pub fn calc_total_score(&mut self, norm: &ScoreNorm) {
    if self.total <= PENALTY_SCORE {
      return; // Hand usage penalty already applied
    }
    self.total = norm.total_score(
      self.score.effort,
      self.score.sfb,
      self.score.rolling,
      self.score.scissors,
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

pub fn save_results(layouts: &[Layout], path: &str) -> anyhow::Result<()> {
  let bytes = bincode::serialize(layouts)?;
  std::fs::write(path, bytes)?;
  Ok(())
}

pub fn load_results(path: &str) -> anyhow::Result<Vec<Layout>> {
  let bytes = std::fs::read(path)?;
  Ok(bincode::deserialize(&bytes)?)
}

pub fn init_score_norm(arrays: &CorpusArrays, precomp: &Precomp, rates: &crate::config::ScoreRates) -> ScoreNorm {
  const N: usize = 10001;
  let mut effort_vals   = Vec::with_capacity(N);
  let mut sfb_vals      = Vec::with_capacity(N);
  let mut rolling_vals  = Vec::with_capacity(N);
  let mut scissors_vals = Vec::with_capacity(N);

  while effort_vals.len() < N {
    let mut l = Layout::make_random(&arrays.keys, &[], 0);
    l.calc_scores(arrays, precomp);
    if l.total <= PENALTY_SCORE { continue; }
    effort_vals.push(l.score.effort as f32);
    sfb_vals.push(l.score.sfb as f32);
    rolling_vals.push(l.score.rolling as f32);
    scissors_vals.push(l.score.scissors as f32);
  }

  let mut norm = ScoreNorm::from_samples(&effort_vals, &sfb_vals, &rolling_vals, &scissors_vals);
  norm.set_rates(rates);
  norm
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
  fn sort_unique_respects_min_diff() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays(true);
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
  fn penalized_layout_keeps_real_scores_and_recovers() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays(true);
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
