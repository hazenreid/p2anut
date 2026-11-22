#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use crate::config::{Costs, EffortGrid, Weights};
use crate::layout::{ROWS, COLS};

const N: usize = ROWS * COLS; // 30

// 키 노력값은 설정의 [effort] (config::EffortGrid)에서 받음

// 대표 사건 위치 (왼손 홈 행): [costs]의 inroll(중지 → 검지), redirect(약지 → 검지 → 중지)가 가리키는 사건
const REF_RING_HOME: usize = COLS + 1;
const REF_MIDDLE_HOME: usize = COLS + 2;
const REF_INDEX_HOME: usize = COLS + 3;

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
  // 세 글자 롤의 추가 보너스(양수), 또는 리다이렉트(음수: 안의 두 쌍의 롤 보너스를 되돌리고 리다이렉트 비용을 뺀 값).
  // 바이그램 rolling_base와 같은 단위 (비용 단위, 검지 홈 키 한 번 = 1)
  pub roll_base: f32,
  // 스킵 시저스: 1·3번 키가 같은 손 인접 손가락, 두 줄 차이 (scissors 항목에 더함)
  pub skip_scissors_base: f32,
}

// 공백 건너 쌍 "a _ c" (엄지가 가운데): 1·3번 키만 보는 공백 너머 스킵그램·스킵 시저스. 트라이그램과 같은 식
#[derive(Clone, Copy, Default)]
pub struct CrossEntry {
  pub sfs_base: f32,
  pub skip_scissors_base: f32,
}

// 트라이그램 창 하나의 분류 (결과 통계 표시용). 엄지(공백)는 어느 손도 아니라 교차를 만듦
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Window {
  Alternate,
  Roll2In,
  Roll2Out,
  Roll3In,
  Roll3Out,
  Redirect,
  BadRedirect,
  // 같은 손 연속에 SFB·시저스·가운데 열이 끼어 롤로 치지 않는 창
  SameHandOther,
}

// 손가락 번호 (왼손 검지 1 … 새끼 4, 오른손 검지 5 … 새끼 8). 가운데 열은 검지
pub fn finger(p: usize) -> u8 { FINGER_GRID[p / COLS][p % COLS] }
fn is_left(p: usize) -> bool { p % COLS < COLS / 2 }

pub struct Precomp {
  pub bigrams: Box<[BigramEntry]>,
  pub trigrams: Box<[TrigramEntry]>,
  pub cross: Box<[CrossEntry]>,
  pub alphabet_effort: Box<[f64]>, // Flattened: [pos_idx * CHARS + char_idx]
  pub min_hand_usage: f32,
}

impl Precomp {
  // 롤 값(바이그램 rolling_base, 트라이그램 roll_base)은 비용 단위로 계산: 대표 안쪽 롤 = costs.inroll,
  // 대표 리다이렉트 세 글자 전체 = -costs.redirect. 그래서 리다이렉트 크기가 inroll과 독립적임
  pub fn new(freqs: &[u64], weights: &Weights, costs: &Costs, effort: &EffortGrid, min_hand_usage: f32) -> Self {
    let effort_grid = effort.grid();
    let mut bigrams = vec![BigramEntry::default(); N * N].into_boxed_slice();
    let mut trigrams = vec![TrigramEntry::default(); N * N * N].into_boxed_slice();
    
    let mid = COLS / 2;
    let mut alphabet_effort = vec![0.0f64; crate::layout::CHARS * N].into_boxed_slice();
    for char_idx in 0..crate::layout::CHARS {
      let f = freqs[char_idx] as f64;
      for pos_idx in 0..N {
        let r = pos_idx / COLS; let c = pos_idx % COLS;
        alphabet_effort[pos_idx * crate::layout::CHARS + char_idx] = f * f64::from(effort_grid[r][c]);
      }
    }

    let pos_info: Vec<(usize, u8, f32, bool, bool)> = (0..N).map(|p| {
      let r = p / COLS; let c = p % COLS;
      let finger = FINGER_GRID[r][c];
      let is_center = c == mid-1 || c == mid;
      let is_left = c < mid;
      (r, finger, effort_grid[r][c], is_center, is_left)
    }).collect();

    // 손가락 간격 배율 (붙은 손가락 = 1, 하나 건너뛸 때마다 gap배)
    let gap_mult = |fa: u8, fb: u8| { let g = fa.abs_diff(fb); if g > 1 { weights.gap.powf(f32::from(g - 1)) } else { 1.0 } };
    // 롤 모양 (방향 × 간격 × 행 이동 / 노력값 합). 대표 안쪽 롤이 costs.inroll이 되도록 배율을 맞춤
    let raw_roll = |p1: usize, p2: usize| {
      let (r1, f1, e1, _, _) = pos_info[p1]; let (r2, f2, e2, _, _) = pos_info[p2];
      (if f1 > f2 { 1.0f32 } else { weights.outroll }) * gap_mult(f1, f2) * weights.roll_row().powf(r1.abs_diff(r2) as f32) / (e1 + e2)
    };
    let ref_inroll = raw_roll(REF_MIDDLE_HOME, REF_INDEX_HOME);
    let roll_scale = if ref_inroll > 0.0 { costs.inroll / ref_inroll } else { 0.0 };
    // 리다이렉트 모양 (노력값 합 / (간격 × 행 이동 배율)): 대표 리다이렉트가 1이 되도록 나눠서 씀
    let redirect_shape = |p1: usize, p2: usize, p3: usize| {
      let (r1, f1, e1, _, _) = pos_info[p1]; let (r2, f2, e2, _, _) = pos_info[p2]; let (r3, f3, e3, _, _) = pos_info[p3];
      let m = gap_mult(f1, f2) * gap_mult(f2, f3) * weights.roll_row().powf((r1.abs_diff(r2) + r2.abs_diff(r3)) as f32);
      (e1 + e2 + e3) / m
    };
    let ref_redirect = redirect_shape(REF_RING_HOME, REF_INDEX_HOME, REF_MIDDLE_HOME);
    // 1·3번 키만 보는 값 (가운데가 다른 손가락이거나 엄지일 때 공통)
    // 스킵 풀 시저스: 같은 손 인접 손가락, 두 줄 차이. 센터는 바이그램에서 이미 감점하므로 제외.
    // 방향은 풀 시저스와 같음 (긴 손가락이 아래면 1.0, 위면 full_scissors_good)
    let skip_scissors_of = |p1: usize, p3: usize| {
      let (r1, f1, e1, ic1, il1) = pos_info[p1]; let (r3, f3, e3, ic3, il3) = pos_info[p3];
      if il1 == il3 && f1.abs_diff(f3) == 1 && r1.abs_diff(r3) == 2 && !ic1 && !ic3 {
        let dir = if long_finger_lower(f1, r1, f3, r3) { 1.0 } else { weights.full_scissors_good };
        weights.skip_scissors * dir * (e1 + e3) * (1.0 + weights.row * 2.0)
      } else { 0.0 }
    };
    // 같은 손가락 스킵그램 (같은 키면 sfs_same_key)
    let sfs_of = |p1: usize, p3: usize| {
      let (r1, f1, e1, ic1, _) = pos_info[p1]; let (r3, f3, e3, ic3, _) = pos_info[p3];
      if f1 != f3 { return 0.0; }
      let w = if p1 == p3 { weights.sfs_same_key() } else { weights.sfs };
      w * (e1 + e3) * (if ic1 || ic3 { weights.center } else { 1.0 }) * (1.0 + weights.row * r1.abs_diff(r3) as f32)
    };
    let mut cross = vec![CrossEntry::default(); N * N].into_boxed_slice();
    for p1 in 0..N { for p3 in 0..N {
      cross[p1 * N + p3] = CrossEntry { sfs_base: sfs_of(p1, p3), skip_scissors_base: skip_scissors_of(p1, p3) };
    }}

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
            entry.rolling_base = roll_scale * raw_roll(p1, p2);
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
          entry.skip_scissors_base = skip_scissors_of(p1, p3);
          if f1 == f3 && f1 != f2 {
            entry.sfs_base = sfs_of(p1, p3);
          } else if il1 == il2 && il2 == il3 {
            let is_center = ic1 || ic2 || ic3;
            let row_delta1 = r1.abs_diff(r2); let row_delta2 = r2.abs_diff(r3);
            // 풀 시저스(행 2칸)·나쁜 방향 하프 시저스 전이가 있으면 롤/리다이렉트로 치지 않음
            if f1 != f2 && f2 != f3 && f1 != f3 && !is_center && row_delta1 != 2 && row_delta2 != 2
              && !is_bad_half_scissor(f1, r1, f2, r2) && !is_bad_half_scissor(f2, r2, f3, r3) {
              let monotonic = (f1 > f2 && f2 > f3) || (f1 < f2 && f2 < f3);
              if monotonic {
                // 세 글자 롤: 두 쌍의 롤 보너스에 더해 주는 추가 보너스 (안쪽 1, 바깥쪽 outroll배)
                let dir = if f1 > f2 { 1.0 } else { weights.outroll };
                let m = gap_mult(f1, f2) * gap_mult(f2, f3) * weights.roll_row().powf((row_delta1 + row_delta2) as f32);
                entry.roll_base = roll_scale * dir * m / (e1 + e2 + e3);
              } else {
                // 리다이렉트: 하나의 사건으로 침. 안의 두 쌍(둘 다 롤로 분류됨)의 보너스를 되돌리고 리다이렉트 비용을 뺌
                let bad = ![f1, f2, f3].iter().any(|&f| f == 1 || f == 5);
                let cost = costs.redirect * (if bad { weights.bad_redirect } else { 1.0 }) * redirect_shape(p1, p2, p3) / ref_redirect;
                let pair_bonus = bigrams[p1 * N + p2].rolling_base + bigrams[p2 * N + p3].rolling_base;
                entry.roll_base = -pair_bonus - cost;
              }
            }
          }
        }
      }
    }
    Self { bigrams, trigrams, cross, alphabet_effort, min_hand_usage }
  }

  // 왼손·오른손 중 한쪽이라도 min_hand_usage(여기서는 실격 기준 hand_hard_limit을 받음) 미만이면 실격
  #[inline] pub fn violates_hand_usage(&self, left: u64, right: u64) -> bool {
    let total = left + right;
    if total == 0 { return false; }
    let l = left as f32 / total as f32;
    l < self.min_hand_usage || 1.0 - l < self.min_hand_usage
  }
  #[inline] pub fn bigram(&self, p1: usize, p2: usize) -> &BigramEntry { &self.bigrams[p1 * N + p2] }
  #[inline] pub fn trigram(&self, p1: usize, p2: usize, p3: usize) -> &TrigramEntry { &self.trigrams[p1 * N * N + p2 * N + p3] }
  #[inline] pub fn effort(&self, pos: usize, char_idx: usize) -> f64 { self.alphabet_effort[pos * crate::layout::CHARS + char_idx] }
  #[inline] pub fn cross(&self, p1: usize, p3: usize) -> &CrossEntry { &self.cross[p1 * N + p3] }

  // 창 하나의 분류. None = 엄지(공백). 같은 손 두 키가 붙어 있고 나머지 하나가 다른 손이나 엄지면 2키 롤,
  // 같은 손 세 키면 세 글자 롤·리다이렉트 (점수 계산과 같은 기준: 롤 값이 있는 쌍·창만 롤)
  pub fn window(&self, p1: Option<usize>, p2: Option<usize>, p3: Option<usize>) -> Window {
    let same = |a: Option<usize>, b: Option<usize>| matches!((a, b), (Some(a), Some(b)) if is_left(a) == is_left(b));
    let pair = |a: usize, b: usize| {
      if self.bigram(a, b).rolling_base <= 0.0 { Window::SameHandOther }
      else if finger(a) > finger(b) { Window::Roll2In } else { Window::Roll2Out }
    };
    match (p1, p2, p3) {
      (Some(a), Some(b), Some(c)) if same(p1, p2) && same(p2, p3) => {
        let t = self.trigram(a, b, c).roll_base;
        if t > 0.0 { if finger(a) > finger(b) { Window::Roll3In } else { Window::Roll3Out } }
        else if t < 0.0 {
          if [a, b, c].iter().any(|&p| matches!(finger(p), 1 | 5)) { Window::Redirect } else { Window::BadRedirect }
        } else { Window::SameHandOther }
      }
      (Some(a), Some(b), _) if same(p1, p2) => pair(a, b),
      (_, Some(b), Some(c)) if same(p2, p3) => pair(b, c),
      _ => Window::Alternate,
    }
  }
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
  fn sfs_weight_scales_skipgrams_only() {
    let mut w = crate::test_util::weights();
    w.sfs = 1.0;
    let full = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
    w.sfs = 0.5;
    let half = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
    // 왼손 중지 윗줄(0,2) → 다른 손가락 → 중지 아랫줄(2,2): 스킵그램
    let (a, b, c) = (2, COLS + 3, 2 * COLS + 2);
    assert!(full.trigram(a, b, c).sfs_base > 0.0);
    assert!((half.trigram(a, b, c).sfs_base - 0.5 * full.trigram(a, b, c).sfs_base).abs() < 1e-6);
    // 같은 조합의 SFB는 그대로
    assert_eq!(half.bigram(a, c).sfb_base, full.bigram(a, c).sfb_base);
    assert!(half.bigram(a, c).sfb_base > 0.0);
    // 1이면 스킵그램과 SFB가 같은 무게 (예전 동작)
    assert!((full.trigram(a, b, c).sfs_base - full.bigram(a, c).sfb_base).abs() < 1e-6);
  }

  #[test]
  fn same_key_skipgram_has_its_own_weight() {
    let mut w = crate::test_util::weights();
    w.sfs = 0.5;
    let old = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
    w.sfs_same_key = Some(0.0);
    let new = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
    // 같은 키(검지 홈 → 다른 손가락 → 같은 검지 홈): 없으면 sfs와 같고, 0이면 감점 없음
    let (k, mid) = (COLS + 3, COLS + 1);
    assert!(old.trigram(k, mid, k).sfs_base > 0.0);
    assert_eq!(new.trigram(k, mid, k).sfs_base, 0.0);
    // 다른 키 스킵그램(검지 홈 → 다른 손가락 → 검지 윗줄)은 그대로
    assert_eq!(new.trigram(k, mid, 3).sfs_base, old.trigram(k, mid, 3).sfs_base);
    assert!(new.trigram(k, mid, 3).sfs_base > 0.0);
  }

  // 세 글자 전체의 롤 값 (두 쌍의 바이그램 + 트라이그램). 비용 단위, 양수 = 이득
  fn whole3(pc: &Precomp, p1: usize, p2: usize, p3: usize) -> f32 {
    pc.bigram(p1, p2).rolling_base + pc.bigram(p2, p3).rolling_base + pc.trigram(p1, p2, p3).roll_base
  }

  #[test]
  fn redirect_is_one_event_with_its_own_cost() {
    let mut w = crate::test_util::weights();
    w.bad_redirect = 2.0;
    let c = Costs::default();
    let pc = Precomp::new(&[1; crate::layout::CHARS], &w, &c, &EffortGrid::default(), 0.45);
    // 대표 리다이렉트(약지 → 검지 → 중지, 홈 행): 세 글자 전체가 정확히 -costs.redirect (안의 롤 보너스는 되돌림)
    let home = whole3(&pc, REF_RING_HOME, REF_INDEX_HOME, REF_MIDDLE_HOME);
    assert!((home + c.redirect).abs() < 1e-5, "{home}");
    // 어려운 키·넓은 간격일수록 비쌈
    let top = whole3(&pc, 1, 3, 2);
    let wide = whole3(&pc, COLS, COLS + 3, COLS + 1); // 새끼 → 검지 → 약지
    assert!(top < home && wide < home, "top {top}, wide {wide}");
    // 검지 없는 리다이렉트(약지 → 새끼 → 중지)는 bad_redirect배: 노력값 (1.1+1.6+1.0)/3.1, 간격 배율은 대표와 같음
    let bad = whole3(&pc, COLS + 1, COLS, COLS + 2);
    assert!((bad + 2.0 * c.redirect * 3.7 / 3.1).abs() < 1e-4, "{bad}");
    // inroll을 바꿔도 리다이렉트 크기는 그대로
    let c2 = Costs { inroll: 4.0, ..Costs::default() };
    let pc2 = Precomp::new(&[1; crate::layout::CHARS], &w, &c2, &EffortGrid::default(), 0.45);
    assert!((whole3(&pc2, REF_RING_HOME, REF_INDEX_HOME, REF_MIDDLE_HOME) + c.redirect).abs() < 1e-5);
  }

  #[test]
  fn rolls_are_in_cost_units() {
    let w = crate::test_util::weights();
    let c = Costs::default();
    let pc = Precomp::new(&[1; crate::layout::CHARS], &w, &c, &EffortGrid::default(), 0.45);
    // 대표 안쪽 롤(중지 홈 → 검지 홈) = costs.inroll, 바깥쪽은 outroll배
    assert!((pc.bigram(REF_MIDDLE_HOME, REF_INDEX_HOME).rolling_base - c.inroll).abs() < 1e-6);
    assert!((pc.bigram(REF_INDEX_HOME, REF_MIDDLE_HOME).rolling_base - c.inroll * w.outroll).abs() < 1e-6);
    // 세 글자 안쪽 롤(약지 → 중지 → 검지)의 추가 보너스: 대표 롤과 같은 배율 × 1/(1.1+1.0+1.0)
    let scale = c.inroll / (1.0 / 2.0);
    let extra = pc.trigram(REF_RING_HOME, REF_MIDDLE_HOME, REF_INDEX_HOME).roll_base;
    assert!((extra - scale / 3.1).abs() < 1e-5, "{extra}");
    // 세 글자 롤 전체는 두 쌍의 롤 + 추가 보너스 (리다이렉트처럼 되돌리지 않음)
    let total = whole3(&pc, REF_RING_HOME, REF_MIDDLE_HOME, REF_INDEX_HOME);
    assert!(total > pc.bigram(REF_RING_HOME, REF_MIDDLE_HOME).rolling_base + pc.bigram(REF_MIDDLE_HOME, REF_INDEX_HOME).rolling_base);
  }

  #[test]
  fn cross_space_values_match_the_trigram_formulas() {
    let mut w = crate::test_util::weights();
    w.sfs = 0.5;
    w.sfs_same_key = Some(0.0);
    let pc = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
    let mut sfs = 0;
    let mut skip = 0;
    for p1 in 0..N { for p3 in 0..N {
      let x = pc.cross(p1, p3);
      if p1 == p3 { assert_eq!(x.sfs_base, 0.0); continue; } // 같은 키(e_e)는 sfs_same_key = 0
      // 가운데를 다른 손 키로 둔 트라이그램과 같은 값
      let other = if is_left(p1) { COLS + 7 } else { COLS + 2 };
      let t = pc.trigram(p1, other, p3);
      assert_eq!((x.sfs_base, x.skip_scissors_base), (t.sfs_base, t.skip_scissors_base), "{p1} _ {p3}");
      if x.sfs_base > 0.0 { sfs += 1; }
      if x.skip_scissors_base > 0.0 { skip += 1; }
    }}
    assert!(sfs > 0 && skip > 0);
    // 같은 키 배율을 주면 같은 키도 감점
    w.sfs_same_key = Some(0.5);
    let pc = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
    assert!(pc.cross(COLS + 2, COLS + 2).sfs_base > 0.0);
  }

  #[test]
  fn windows_are_classified_like_the_score() {
    let pc = Precomp::new(&[1; crate::layout::CHARS], &crate::test_util::weights(), &Costs::default(), &EffortGrid::default(), 0.45);
    let (pinky, ring, middle, index) = (Some(COLS), Some(REF_RING_HOME), Some(REF_MIDDLE_HOME), Some(REF_INDEX_HOME));
    let right = Some(COLS + 7);
    // 엄지나 다른 손이 끼면 교차 / 2키 롤
    assert_eq!(pc.window(middle, None, index), Window::Alternate);
    assert_eq!(pc.window(middle, right, index), Window::Alternate);
    assert_eq!(pc.window(None, middle, None), Window::Alternate);
    assert_eq!(pc.window(middle, index, None), Window::Roll2In);
    assert_eq!(pc.window(right, index, middle), Window::Roll2Out);
    // 같은 손 세 키
    assert_eq!(pc.window(ring, middle, index), Window::Roll3In);
    assert_eq!(pc.window(index, middle, ring), Window::Roll3Out);
    assert_eq!(pc.window(ring, index, middle), Window::Redirect);
    assert_eq!(pc.window(ring, pinky, middle), Window::BadRedirect);
    // SFB(검지 홈 → 검지 윗줄)가 낀 창은 롤이 아님
    assert_eq!(pc.window(index, Some(3), None), Window::SameHandOther);
    assert_eq!(pc.window(middle, index, Some(3)), Window::SameHandOther);
  }

  #[test]
  fn half_scissors_weight_applies_to_bad_direction_only() {
    let mut w = crate::test_util::weights();
    w.half_scissors = 0.0;
    let off = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
    w.half_scissors = 1.0;
    let on = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
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
