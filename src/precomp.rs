#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use crate::config::Weights;
use crate::layout::{ROWS, COLS};

const N: usize = ROWS * COLS; // 30

const EFFORT_GRID: [[f32; COLS]; ROWS] = [
  [2.4, 1.3, 1.1, 1.3, 3.5, 3.5, 1.3, 1.1, 1.3, 2.4],  // top
  [1.6, 1.1, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.1, 1.6],  // home
  [2.2, 1.5, 1.3, 1.1, 4.0, 4.0, 1.1, 1.3, 1.5, 2.2],  // bottom
];

// 손가락 길이 순위 (클수록 김): 중지 > 약지 > 검지 > 새끼. FINGER_GRID 번호(1~8)로 인덱스
const FINGER_LEN: [u8; 9] = [0, 2, 4, 3, 1, 2, 4, 3, 1];

// 두 손가락 중 긴 손가락이 더 아래 행인지 (시저스의 나쁜 방향)
fn long_finger_lower(f1: u8, r1: usize, f2: u8, r2: usize) -> bool {
  let (long_row, short_row) = if FINGER_LEN[f1 as usize] > FINGER_LEN[f2 as usize] { (r1, r2) } else { (r2, r1) };
  long_row > short_row
}

// 인접 손가락이 한 줄 차이로 연달아 칠 때, 긴 손가락이 더 아래 행이면 나쁜 방향 하프 시저스
fn is_bad_half_scissor(f1: u8, r1: usize, f2: u8, r2: usize) -> bool {
  f1.abs_diff(f2) == 1 && r1.abs_diff(r2) == 1 && long_finger_lower(f1, r1, f2, r2)
}

const FINGER_GRID: [[u8; COLS]; ROWS] = [
  [4, 3, 2, 1, 1, 5, 5, 6, 7, 8],
  [4, 3, 2, 1, 1, 5, 5, 6, 7, 8],
  [4, 3, 2, 1, 1, 5, 5, 6, 7, 8],
];

#[derive(Clone, Copy, Default)]
#[allow(clippy::struct_field_names)]
pub struct BigramEntry {
  pub sfb_base: f32,
  pub rolling_base: f32,
  pub scissors_base: f32,
}

#[derive(Clone, Copy, Default)]
pub struct TrigramEntry {
  pub sfs_base: f32,
  pub roll_base: f32,
  // 스킵 시저스: 1·3번 키가 같은 손 인접 손가락, 두 줄 차이 (scissors 항목에 더함)
  pub skip_scissors_base: f32,
}

pub struct Precomp {
  pub bigrams: Box<[BigramEntry]>,
  pub trigrams: Box<[TrigramEntry]>,
  pub alphabet_effort: Box<[f64]>, // Flattened: [pos_idx * CHARS + char_idx]
  pub min_hand_usage: f32,
}

impl Precomp {
  pub fn new(freqs: &[u64], weights: &Weights, min_hand_usage: f32) -> Self {
    let mut bigrams = vec![BigramEntry::default(); N * N].into_boxed_slice();
    let mut trigrams = vec![TrigramEntry::default(); N * N * N].into_boxed_slice();
    
    let mid = COLS / 2;
    let mut alphabet_effort = vec![0.0f64; crate::layout::CHARS * N].into_boxed_slice();
    for char_idx in 0..crate::layout::CHARS {
      let f = freqs[char_idx] as f64;
      for pos_idx in 0..N {
        let r = pos_idx / COLS; let c = pos_idx % COLS;
        alphabet_effort[pos_idx * crate::layout::CHARS + char_idx] = f * f64::from(EFFORT_GRID[r][c]);
      }
    }

    let pos_info: Vec<(usize, u8, f32, bool, bool)> = (0..N).map(|p| {
      let r = p / COLS; let c = p % COLS;
      let finger = FINGER_GRID[r][c];
      let is_center = c == mid-1 || c == mid;
      let is_left = c < mid;
      (r, finger, EFFORT_GRID[r][c], is_center, is_left)
    }).collect();

    for p1 in 0..N {
      let (r1, f1, e1, ic1, il1) = pos_info[p1];
      for p2 in 0..N {
        if p1 == p2 { continue; }
        let (r2, f2, e2, ic2, il2) = pos_info[p2];
        if il1 != il2 { continue; }
        let row_dist = r1.abs_diff(r2);
        let is_center = ic1 || ic2;
        let effort_sum = e1 + e2;
        let entry = &mut bigrams[p1 * N + p2];
        if f1 == f2 {
          let row_mult = 1.0 + (weights.row * row_dist as f32);
          entry.sfb_base = effort_sum * (if is_center { weights.center } else { 1.0 }) * row_mult;
        } else {
          let gap_dist = f1.abs_diff(f2);
          if is_center {
            let row_mult = 1.0 + (weights.row * row_dist as f32);
            entry.scissors_base = effort_sum * weights.center * row_mult;
          } else if gap_dist <= 1 && row_dist == 2 {
            // 풀 시저스: 긴 손가락이 아래면 1.0, 위(편한 방향)면 full_scissors_good 배
            let dir = if long_finger_lower(f1, r1, f2, r2) { 1.0 } else { weights.full_scissors_good };
            entry.scissors_base = dir * effort_sum * (1.0 + weights.row * row_dist as f32);
          } else if is_bad_half_scissor(f1, r1, f2, r2) {
            // 나쁜 방향 하프 시저스: 풀 시저스처럼 롤이 아닌 시저스로 분류
            entry.scissors_base = weights.half_scissors * effort_sum * (1.0 + weights.row * row_dist as f32);
          } else {
            let gap_mult = if gap_dist > 1 { weights.gap.powf(f32::from(gap_dist - 1)) } else { 1.0 };
            let row_mult = weights.roll_row().powf(row_dist as f32);
            entry.rolling_base = (if f1 > f2 { 1.0f32 } else { weights.outroll }) * gap_mult * row_mult / effort_sum;
          }
        }
      }
    }

    for p1 in 0..N {
      let (r1, f1, e1, ic1, il1) = pos_info[p1];
      for p2 in 0..N {
        if p1 == p2 { continue; }
        let (r2, f2, e2, ic2, il2) = pos_info[p2];
        for p3 in 0..N {
          if p2 == p3 { continue; }
          let (r3, f3, e3, ic3, il3) = pos_info[p3];
          let entry = &mut trigrams[p1 * N * N + p2 * N + p3];
          // 스킵 풀 시저스: 가운데 키와 상관없이 1·3번만 봄. 센터는 바이그램에서 이미 감점하므로 제외.
          // 방향은 풀 시저스와 같음 (긴 손가락이 아래면 1.0, 위면 full_scissors_good)
          if il1 == il3 && f1.abs_diff(f3) == 1 && r1.abs_diff(r3) == 2 && !ic1 && !ic3 {
            let dir = if long_finger_lower(f1, r1, f3, r3) { 1.0 } else { weights.full_scissors_good };
            entry.skip_scissors_base = weights.skip_scissors * dir * (e1 + e3) * (1.0 + weights.row * 2.0);
          }
          if f1 == f3 && f1 != f2 {
            let row_dist = r1.abs_diff(r3);
            let row_mult = 1.0 + (weights.row * row_dist as f32);
            entry.sfs_base = (e1 + e3)
              * (if ic1 || ic3 { weights.center } else { 1.0 })
              * row_mult;
          } else if il1 == il2 && il2 == il3 {
            let is_center = ic1 || ic2 || ic3;
            let row_delta1 = r1.abs_diff(r2); let row_delta2 = r2.abs_diff(r3);
            // 풀 시저스(행 2칸)·나쁜 방향 하프 시저스 전이가 있으면 롤/리다이렉트로 치지 않음
            if f1 != f2 && f2 != f3 && f1 != f3 && !is_center && row_delta1 != 2 && row_delta2 != 2
              && !is_bad_half_scissor(f1, r1, f2, r2) && !is_bad_half_scissor(f2, r2, f3, r3) {
              let w: f32 = if f1 > f2 && f2 > f3 { 1.0 } else if f1 < f2 && f2 < f3 { weights.outroll } else { weights.redirect };
              let effort_sum = e1 + e2 + e3;
              let gap_dist1 = f1.abs_diff(f2);
              let gap_dist2 = f2.abs_diff(f3);
              let gap_mult1 = if gap_dist1 > 1 { weights.gap.powf(f32::from(gap_dist1 - 1)) } else { 1.0 };
              let gap_mult2 = if gap_dist2 > 1 { weights.gap.powf(f32::from(gap_dist2 - 1)) } else { 1.0 };
              let total_row_dist = (row_delta1 + row_delta2) as f32;
              let row_mult = weights.roll_row().powf(total_row_dist);
              entry.roll_base = w * gap_mult1 * gap_mult2 * row_mult / effort_sum;
            }
          }
        }
      }
    }
    Self { bigrams, trigrams, alphabet_effort, min_hand_usage }
  }

  // 왼손·오른손 중 한쪽이라도 min_hand_usage 미만이면 위반
  #[inline] pub fn violates_hand_usage(&self, left: u64, right: u64) -> bool {
    let total = left + right;
    if total == 0 { return false; }
    let l = left as f32 / total as f32;
    l < self.min_hand_usage || 1.0 - l < self.min_hand_usage
  }
  #[inline] pub fn bigram(&self, p1: usize, p2: usize) -> &BigramEntry { &self.bigrams[p1 * N + p2] }
  #[inline] pub fn trigram(&self, p1: usize, p2: usize, p3: usize) -> &TrigramEntry { &self.trigrams[p1 * N * N + p2 * N + p3] }
  #[inline] pub fn effort(&self, pos: usize, char_idx: usize) -> f64 { self.alphabet_effort[pos * crate::layout::CHARS + char_idx] }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bad_half_scissor_direction() {
    // 왼손: 검지(1) 윗줄 + 중지(2) 홈 → 긴 손가락(중지)이 아래 → 나쁨
    assert!(is_bad_half_scissor(1, 0, 2, 1));
    assert!(is_bad_half_scissor(2, 1, 1, 0));
    // 중지 윗줄 + 검지 홈 → 편한 방향
    assert!(!is_bad_half_scissor(2, 0, 1, 1));
    // 약지(3) 아래 + 새끼(4) 홈 → 나쁨, 약지 윗줄 + 새끼 홈 → 편함
    assert!(is_bad_half_scissor(3, 2, 4, 1));
    assert!(!is_bad_half_scissor(3, 0, 4, 1));
    // 두 줄 차이, 같은 줄, 인접하지 않은 손가락은 하프 시저스가 아님
    assert!(!is_bad_half_scissor(1, 0, 2, 2));
    assert!(!is_bad_half_scissor(1, 1, 2, 1));
    assert!(!is_bad_half_scissor(1, 0, 3, 1));
    // 오른손 검지(5) 윗줄 + 중지(6) 홈 → 나쁨
    assert!(is_bad_half_scissor(5, 0, 6, 1));
  }

  #[test]
  fn half_scissors_weight_applies_to_bad_direction_only() {
    let mut w = crate::test_util::weights();
    w.half_scissors = 0.0;
    let off = Precomp::new(&[1; crate::layout::CHARS], &w, 0.45);
    w.half_scissors = 1.0;
    let on = Precomp::new(&[1; crate::layout::CHARS], &w, 0.45);
    // 왼손 검지 윗줄(0,3) → 중지 홈(1,2): 나쁜 방향, 롤 보상은 유지
    let (bad1, bad2) = (3, COLS + 2);
    assert_eq!(off.bigram(bad1, bad2).scissors_base, 0.0);
    assert!(on.bigram(bad1, bad2).scissors_base > 0.0);
    // 나쁜 방향은 롤로 분류되지 않음
    assert_eq!(on.bigram(bad1, bad2).rolling_base, 0.0);
    // 트라이그램: 새끼 홈(1,0) → 검지 윗줄(0,3) → 중지 홈(1,2)은 둘째 전이가 나쁜 하프 → 롤/리다이렉트 아님
    assert_eq!(on.trigram(COLS, 3, COLS + 2).roll_base, 0.0);
    // 같은 손가락 순서라도 편한 방향(새끼 홈 → 검지 홈 → 중지 윗줄)이면 리다이렉트로 계산됨
    assert!(on.trigram(COLS, COLS + 3, 2).roll_base != 0.0);
    // 풀 시저스 방향: 중지 윗줄(0,2)·검지 아랫줄(2,3)은 편한 방향, 중지 아랫줄(2,2)·검지 윗줄(0,3)은 나쁜 방향
    let good = on.bigram(2, 2 * COLS + 3).scissors_base;
    let bad = on.bigram(2 * COLS + 2, 3).scissors_base;
    assert!(good > 0.0 && bad > 0.0);
    assert!((good / bad - w.full_scissors_good * (1.1 + 1.1) / (1.3 + 1.3)).abs() < 1e-5);
    // 스킵 풀 시저스: 중지 아랫줄(2,2) → 아무 키 → 검지 윗줄(0,3)은 나쁜 방향, 반대는 편한 방향
    let sk_bad = on.trigram(2 * COLS + 2, COLS + 7, 3).skip_scissors_base;
    let sk_good = on.trigram(2, COLS + 7, 2 * COLS + 3).skip_scissors_base;
    assert!((sk_bad - w.skip_scissors * bad).abs() < 1e-5);
    assert!((sk_good - w.skip_scissors * good).abs() < 1e-5);
    // 인접하지 않거나 한 줄 차이면 스킵 시저스 아님
    assert_eq!(on.trigram(2 * COLS + 1, COLS + 7, 3).skip_scissors_base, 0.0);
    assert_eq!(on.trigram(COLS + 2, COLS + 7, 3).skip_scissors_base, 0.0);
    // 중지 윗줄(0,2) → 검지 홈(1,3): 편한 방향, 롤 보상 유지
    assert_eq!(on.bigram(2, COLS + 3).scissors_base, 0.0);
    assert!(on.bigram(2, COLS + 3).rolling_base > 0.0);
  }
}
