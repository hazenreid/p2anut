#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use crate::config::{Costs, EffortGrid, Weights};
use crate::layout::{ROWS, COLS};

const N: usize = ROWS * COLS; // 30

// 키 노력값은 설정의 [effort] (config::EffortGrid)에서 받음

// 대표 사건 위치 (왼손 홈 행): [costs]의 inroll(중지 → 검지), redirect(약지 → 검지 → 중지)가 가리키는 사건
const REF_RING_HOME: usize = COLS + 1;
const REF_MIDDLE_HOME: usize = COLS + 2;
const REF_INDEX_HOME: usize = COLS + 3;
// [costs]의 lateral(중지 홈 → 가운데 홈)과 full_scissors(중지 아랫줄 → 검지 윗줄)가 가리키는 사건
const REF_CENTER_HOME: usize = COLS + 4;
const REF_MIDDLE_BOTTOM: usize = 2 * COLS + 2;
const REF_INDEX_TOP: usize = 3;

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
  // 한손: 엄지층 글자가 이어지면 엄지를 누른 채로 쳐서 탭 하나를 돌려줌 (노력값 단위, 음수. 두 손이면 0)
  pub tap_base: f32,
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

// 엄지가 사이에 끼는 두 누름 "a T c" (심볼 레이어의 원샷 탭): 1·3번 키만 보는 스킵그램·스킵 시저스. 트라이그램과 같은 식
#[derive(Clone, Copy, Default)]
pub struct CrossEntry {
  pub sfs_base: f32,
  pub skip_scissors_base: f32,
}

// 트라이그램 창 하나의 분류 (결과 통계 표시용)
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
  // 한손: 엄지 탭이 낀 창 (롤로 치지 않음)
  Tap,
}

// 손가락 번호 (왼손 검지 1 … 새끼 4, 오른손 검지 5 … 새끼 8). 가운데 열은 검지
pub fn finger(p: usize) -> u8 { FINGER_GRID[p / COLS][p % COLS] }
#[cfg(test)]
fn is_left(p: usize) -> bool { p % COLS < COLS / 2 }

pub struct Precomp {
  pub bigrams: Box<[BigramEntry]>,
  pub trigrams: Box<[TrigramEntry]>,
  pub cross: Box<[CrossEntry]>,
  pub alphabet_effort: Box<[f64]>, // Flattened: [pos_idx * CHARS + char_idx]
  pub min_hand_usage: f32,
  // 한손 배열 (왼손 3×5 + 엄지 원샷 레이어 키). 왼쪽 5열은 기본층, 오른쪽 5열은 엄지층을 좌우로 뒤집어 둔 것
  // (9열 = 새끼 … 5열 = 가운데 열. 노력값 표와 가운데 열 규칙을 그대로 쓰려고 두 손의 오른손 자리와 맞춤)
  pub one_hand: bool,
  // 엄지 탭 한 번의 노력값 (노력값 표에서 가장 작은 값, 두 손이면 0). 엄지층 글자를 누를 때마다 더하고, 앞 글자도
  // 엄지층이면(누른 채로 이어 침) 바이그램의 tap_base로 돌려줌. 심볼 레이어의 탭과 같은 값·같은 규칙
  pub tap: f32,
  // 칸마다 실제로 누르는 손가락, 키(엄지층은 같은 자리 기본층 칸 번호), 엄지층 여부, 왼손 여부
  fingers: [u8; N],
  keys: [u8; N],
  layers: [bool; N],
  lefts: [bool; N],
}

impl Precomp {
  // 두 손 배열의 표
  pub fn new(freqs: &[u64], weights: &Weights, costs: &Costs, effort: &EffortGrid, min_hand_usage: f32) -> Self {
    Self::build(freqs, weights, costs, effort, min_hand_usage, false, 1.0)
  }

  // 한손 배열의 표 (손 비율 규칙 없음). layer_effort: 엄지층 키의 노력값 배율 (설정의 one_hand_layer_effort)
  pub fn one_hand(freqs: &[u64], weights: &Weights, costs: &Costs, effort: &EffortGrid, layer_effort: f32) -> Self {
    Self::build(freqs, weights, costs, effort, 0.0, true, layer_effort)
  }

  // 롤 값(바이그램 rolling_base, 트라이그램 roll_base)은 비용 단위로 계산: 대표 안쪽 롤 = costs.inroll,
  // 대표 리다이렉트 세 글자 전체 = -costs.redirect. 그래서 리다이렉트 크기가 inroll과 독립적임.
  // 한손이면 엄지층 칸도 같은 자리 기본층의 손가락·행·노력값으로 셈. 원샷 레이어라 기본층 다음 엄지층 글자는 엄지 탭이
  // 사이에 끼고(탭 너머 쌍: 같은 손가락이면 스킵그램, 인접 손가락 두 줄 차이면 스킵 시저스만, 롤 없음), 엄지층 글자끼리는
  // 엄지를 누른 채로, 엄지층 다음 기본층 글자는 원샷이 풀려 바로 이어 치므로 보통 바이그램(롤 포함).
  // 같은 키의 두 층 글자를 잇따라 치는 것은 같은 키 연타라 SFB가 아님.
  // 엄지층 키의 노력값은 같은 자리 기본층 노력값의 layer_effort배로, 누름 노력값과 SFB·시저스·롤·리다이렉트 크기에 모두 씀
  // (대표 사건은 모두 기본층이라 [costs]의 기준은 그대로). 엄지 탭은 노력값 표의 가장 작은 값 그대로
  fn build(freqs: &[u64], weights: &Weights, costs: &Costs, effort: &EffortGrid, min_hand_usage: f32, one_hand: bool, layer_effort: f32) -> Self {
    let effort_grid = effort.grid();
    let mut bigrams = vec![BigramEntry::default(); N * N].into_boxed_slice();
    let mut trigrams = vec![TrigramEntry::default(); N * N * N].into_boxed_slice();

    let mid = COLS / 2;
    // 실제로 누르는 키의 열: 한손이면 엄지층(오른쪽 5열)을 뒤집어 왼손 자리로
    let phys_col = |c: usize| if one_hand && c >= mid { COLS - 1 - c } else { c };
    let layers: [bool; N] = std::array::from_fn(|p| one_hand && p % COLS >= mid);
    let keys: [u8; N] = std::array::from_fn(|p| (p / COLS * COLS + phys_col(p % COLS)) as u8);
    let tap = if one_hand { effort_grid.iter().flatten().copied().fold(f32::INFINITY, f32::min) } else { 0.0 };
    let tapped = |a: usize, b: usize| layers[b] && !layers[a];
    let key_effort = |p: usize| effort_grid[p / COLS][phys_col(p % COLS)] * if layers[p] { layer_effort } else { 1.0 };
    let mut alphabet_effort = vec![0.0f64; crate::layout::CHARS * N].into_boxed_slice();
    for char_idx in 0..crate::layout::CHARS {
      let f = freqs[char_idx] as f64;
      for pos_idx in 0..N {
        alphabet_effort[pos_idx * crate::layout::CHARS + char_idx] = f * f64::from(key_effort(pos_idx) + if layers[pos_idx] { tap } else { 0.0 });
      }
    }

    let pos_info: Vec<(usize, u8, f32, bool, bool)> = (0..N).map(|p| {
      let r = p / COLS; let c = phys_col(p % COLS);
      let finger = FINGER_GRID[r][c];
      let is_center = c == mid-1 || c == mid;
      let is_left = c < mid;
      (r, finger, key_effort(p), is_center, is_left)
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
    // 가운데 열로 뻗기(LSB): 가운데 열 키와 같은 손 다른 손가락 키. scissors 항목에 시저스 단위로 들어감.
    // lateral이 있으면 대표 LSB 한 번이 비용 lateral이 되도록 맞춤 (다른 LSB는 노력값 합 × 행 배율에 비례, center 배율은 안 씀).
    // 없으면 예전 동작: 노력값 합 × center × 행 배율을 그대로 시저스 단위로
    let lsb_shape = |p1: usize, p2: usize| {
      let (r1, _, e1, _, _) = pos_info[p1]; let (r2, _, e2, _, _) = pos_info[p2];
      (e1 + e2) * (1.0 + weights.row * r1.abs_diff(r2) as f32)
    };
    let lsb_scale = match costs.lateral {
      None => weights.center,
      Some(lateral) => {
        // 시저스 단위: 대표 풀 시저스(나쁜 방향) 한 번 = costs.full_scissors
        let (rb, _, eb, _, _) = pos_info[REF_MIDDLE_BOTTOM]; let (rt, _, et, _, _) = pos_info[REF_INDEX_TOP];
        let full_ref = (eb + et) * (1.0 + weights.row * rb.abs_diff(rt) as f32);
        let per_unit = if full_ref > 0.0 { costs.full_scissors / full_ref } else { 0.0 };
        let ref_shape = lsb_shape(REF_MIDDLE_HOME, REF_CENTER_HOME);
        if per_unit > 0.0 && ref_shape > 0.0 { lateral / per_unit / ref_shape } else { 0.0 }
      }
    };
    // 1·3번 키만 보는 값 (가운데가 다른 손가락이거나 엄지 탭일 때 공통)
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
      let w = if keys[p1] == keys[p3] { weights.sfs_same_key() } else { weights.sfs };
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
        // 한손: 엄지층 글자끼리는 엄지를 누른 채로 이어 쳐서 탭을 돌려줌
        if layers[p1] && layers[p2] { entry.tap_base = -tap; }
        // 한손: 기본층 다음 엄지층 글자는 엄지 탭이 사이에 낌 (탭 너머 쌍. 같은 키면 같은 키 스킵그램 배율)
        if tapped(p1, p2) {
          entry.sfb_base = sfs_of(p1, p2);
          entry.scissors_base = skip_scissors_of(p1, p2);
          continue;
        }
        // 한손: 같은 키의 엄지층 글자 다음 기본층 글자는 같은 키 연타
        if keys[p1] == keys[p2] { continue; }
        if f1 == f2 {
          let row_mult = 1.0 + (weights.row * row_dist as f32);
          entry.sfb_base = effort_sum * (if is_center { weights.center } else { 1.0 }) * row_mult;
        } else {
          let gap_dist = f1.abs_diff(f2);
          if is_center {
            entry.scissors_base = lsb_scale * lsb_shape(p1, p2);
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
          // 한손: 엄지 탭이 사이에 끼거나 같은 키를 잇따라 치는 세 글자는 셈에서 뺌 (탭 너머 쌍과 같은 키 연타는
          // 바이그램에서 셈. 심볼 레이어와 같은 규칙). 두 손이면 해당 없음
          if tapped(p1, p2) || tapped(p2, p3) || keys[p1] == keys[p2] || keys[p2] == keys[p3] { continue; }
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
    let fingers = std::array::from_fn(|p| pos_info[p].1);
    let lefts = std::array::from_fn(|p| pos_info[p].4);
    Self { bigrams, trigrams, cross, alphabet_effort, min_hand_usage, one_hand, tap, fingers, keys, layers, lefts }
  }

  // 칸 p를 누르는 손가락 (한손이면 엄지층 칸도 왼손 손가락 번호 1~4)
  #[inline] pub fn finger_of(&self, p: usize) -> u8 { self.fingers[p] }
  #[inline] pub fn same_hand(&self, p1: usize, p2: usize) -> bool { self.lefts[p1] == self.lefts[p2] }
  // 같은 물리 키인지 (한손이면 같은 자리의 기본층·엄지층 칸)
  #[inline] pub fn same_key(&self, p1: usize, p2: usize) -> bool { self.keys[p1] == self.keys[p2] }
  // 한손의 엄지층 칸인지
  #[inline] pub fn on_layer(&self, p: usize) -> bool { self.layers[p] }
  // p1 다음 p2를 칠 때 엄지가 레이어 키를 새로 누르는지 (기본층 다음 엄지층 글자)
  #[inline] pub fn tapped(&self, p1: usize, p2: usize) -> bool { self.layers[p2] && !self.layers[p1] }

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

  // 창 하나의 분류. 같은 손 두 키가 붙어 있고 나머지 하나가 다른 손이면 2키 롤,
  // 같은 손 세 키면 세 글자 롤·리다이렉트 (점수 계산과 같은 기준: 롤 값이 있는 쌍·창만 롤)
  // 한손이면 엄지 탭이 낀 창은 따로 (Tap)
  pub fn window(&self, a: usize, b: usize, c: usize) -> Window {
    if self.tapped(a, b) || self.tapped(b, c) { return Window::Tap; }
    let f = |p: usize| self.fingers[p];
    let pair = |a: usize, b: usize| {
      if self.bigram(a, b).rolling_base <= 0.0 { Window::SameHandOther }
      else if f(a) > f(b) { Window::Roll2In } else { Window::Roll2Out }
    };
    let (ab, bc) = (self.same_hand(a, b), self.same_hand(b, c));
    if ab && bc {
      let t = self.trigram(a, b, c).roll_base;
      if t > 0.0 { if f(a) > f(b) { Window::Roll3In } else { Window::Roll3Out } }
      else if t < 0.0 {
        if [a, b, c].iter().any(|&p| matches!(f(p), 1 | 5)) { Window::Redirect } else { Window::BadRedirect }
      } else { Window::SameHandOther }
    } else if ab { pair(a, b) } else if bc { pair(b, c) } else { Window::Alternate }
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
      if p1 == p3 { assert_eq!(x.sfs_base, 0.0); continue; } // 같은 키는 sfs_same_key = 0
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
    let (pinky, ring, middle, index) = (COLS, REF_RING_HOME, REF_MIDDLE_HOME, REF_INDEX_HOME);
    let (right, right2) = (COLS + 7, COLS + 8);
    // 다른 손이 끼면 교차 / 2키 롤
    assert_eq!(pc.window(middle, right, index), Window::Alternate);
    assert_eq!(pc.window(right, middle, right2), Window::Alternate);
    assert_eq!(pc.window(middle, index, right), Window::Roll2In);
    assert_eq!(pc.window(right, index, middle), Window::Roll2Out);
    // 같은 손 세 키
    assert_eq!(pc.window(ring, middle, index), Window::Roll3In);
    assert_eq!(pc.window(index, middle, ring), Window::Roll3Out);
    assert_eq!(pc.window(ring, index, middle), Window::Redirect);
    assert_eq!(pc.window(ring, pinky, middle), Window::BadRedirect);
    // SFB(검지 홈 → 검지 윗줄)가 낀 창은 롤이 아님
    assert_eq!(pc.window(index, 3, right), Window::SameHandOther);
    assert_eq!(pc.window(middle, index, 3), Window::SameHandOther);
  }

  #[test]
  fn lateral_sets_the_cost_of_one_center_stretch() {
    let w = crate::test_util::weights();
    let costs = Costs { lateral: Some(9.6), ..Costs::default() };
    let pc = Precomp::new(&[1; crate::layout::CHARS], &w, &costs, &EffortGrid::default(), 0.45);
    // 시저스 단위 → 비용: 대표 풀 시저스 = costs.full_scissors
    let per_unit = f64::from(costs.full_scissors) / f64::from(pc.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).scissors_base);
    let cost = |p1: usize, p2: usize| f64::from(pc.bigram(p1, p2).scissors_base) * per_unit;
    // 대표 LSB(중지 홈 → 가운데 홈) 한 번 = lateral
    assert!((cost(REF_MIDDLE_HOME, REF_CENTER_HOME) - 9.6).abs() < 1e-4);
    assert!((cost(REF_CENTER_HOME, REF_MIDDLE_HOME) - 9.6).abs() < 1e-4);
    // 다른 LSB는 노력값 합 × 행 배율에 비례 (새끼 홈 1.6 → 가운데 홈 3.0, 중지 홈 1.0 → 가운데 윗줄 3.5)
    assert!((cost(COLS, REF_CENTER_HOME) - 9.6 * 4.6 / 4.0).abs() < 1e-4);
    assert!((cost(REF_MIDDLE_HOME, 4) - 9.6 * 4.5 * (1.0 + f64::from(w.row)) / 4.0).abs() < 1e-3);
    // 가운데 SFB(검지 홈 → 가운데 홈)는 SFB 식 그대로, center 배율은 LSB에 영향 없음
    let none = Precomp::new(&[1; crate::layout::CHARS], &w, &Costs::default(), &EffortGrid::default(), 0.45);
    assert_eq!(pc.bigram(REF_INDEX_HOME, REF_CENTER_HOME).sfb_base, none.bigram(REF_INDEX_HOME, REF_CENTER_HOME).sfb_base);
    let w2 = Weights { center: 5.0, ..w.clone() };
    let pc2 = Precomp::new(&[1; crate::layout::CHARS], &w2, &costs, &EffortGrid::default(), 0.45);
    assert_eq!(pc2.bigram(REF_MIDDLE_HOME, REF_CENTER_HOME).scissors_base, pc.bigram(REF_MIDDLE_HOME, REF_CENTER_HOME).scissors_base);
    // 없으면 예전 동작: 노력값 합 × center × 행 배율 (시저스 단위)
    let old = none.bigram(REF_MIDDLE_HOME, REF_CENTER_HOME).scissors_base;
    assert!((old - 4.0 * w.center).abs() < 1e-5, "{old}");
    // 다른 손 키는 LSB가 아님
    assert_eq!(pc.bigram(COLS + 7, REF_CENTER_HOME).scissors_base, 0.0);
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

  // 한손 표: 엄지층 칸 (같은 자리 기본층 칸 p의 거울 자리. 9열 = 새끼 … 5열 = 가운데 열)
  fn layer(p: usize) -> usize { p / COLS * COLS + (COLS - 1 - p % COLS) }

  fn one_hand_weights() -> Weights {
    // 설정 파일처럼 스킵그램 0.5, 같은 키 스킵그램 0
    Weights { sfs: 0.5, sfs_same_key: Some(0.0), ..crate::test_util::weights() }
  }

  #[test]
  fn one_hand_counts_taps_the_way_we_agreed() {
    let (w, c) = (one_hand_weights(), Costs::default());
    let pc = Precomp::one_hand(&[1; crate::layout::CHARS], &w, &c, &EffortGrid::default(), 1.0);
    let two = Precomp::new(&[1; crate::layout::CHARS], &w, &c, &EffortGrid::default(), 0.45);
    let (middle, index, index_top) = (REF_MIDDLE_HOME, REF_INDEX_HOME, REF_INDEX_TOP);
    // 엄지층 칸은 같은 자리 기본층의 손가락·노력값에 탭(노력값 표의 가장 작은 값)을 더함
    assert!(pc.one_hand && (pc.tap - 1.0).abs() < 1e-6);
    assert_eq!(pc.finger_of(layer(index)), pc.finger_of(index));
    assert!(pc.on_layer(layer(index)) && !pc.on_layer(index) && pc.same_key(index, layer(index)));
    assert!((pc.effort(layer(index), 0) - pc.effort(index, 0) - 1.0).abs() < 1e-9);
    // 기본 중지 홈 → 기본 검지 홈: 안쪽 롤, 두 손 표와 같음
    assert_eq!(pc.bigram(middle, index).rolling_base, two.bigram(middle, index).rolling_base);
    // 기본 중지 홈 → 엄지층 검지 홈: 탭이 끼어 롤 없음 (탭 비용은 엄지층 글자의 노력값에)
    let e = pc.bigram(middle, layer(index));
    assert_eq!((e.sfb_base, e.rolling_base, e.scissors_base, e.tap_base), (0.0, 0.0, 0.0, 0.0));
    assert!(pc.tapped(middle, layer(index)));
    // 기본 검지 홈 → 엄지층 검지 윗줄: SFB가 아니라 스킵그램 (sfs 배)
    let x = pc.bigram(index, layer(index_top));
    assert!((x.sfb_base - w.sfs * pc.bigram(index, index_top).sfb_base).abs() < 1e-6 && x.rolling_base == 0.0);
    // 엄지층 중지 홈 → 엄지층 검지 홈: 누른 채로 이어 침 → 안쪽 롤, 탭 하나를 돌려줌
    let h = pc.bigram(layer(middle), layer(index));
    assert!((h.rolling_base - c.inroll).abs() < 1e-6 && (h.tap_base + 1.0).abs() < 1e-6);
    // 엄지층 → 기본층: 원샷이 풀려 바로 이어 침 → 안쪽 롤(바깥쪽은 outroll배)과 SFB를 그대로 셈, 탭은 돌려주지 않음
    assert!((pc.bigram(layer(middle), index).rolling_base - c.inroll).abs() < 1e-6);
    assert!((pc.bigram(layer(index), middle).rolling_base - c.inroll * w.outroll).abs() < 1e-6);
    assert_eq!(pc.bigram(layer(index), index_top).sfb_base, pc.bigram(index, index_top).sfb_base);
    assert_eq!(pc.bigram(layer(middle), index).tap_base, 0.0);
    // 같은 키의 두 층 글자: 기본 → 엄지층은 탭만, 엄지층 → 기본은 같은 키 연타라 아무것도 없음
    for (a, b) in [(index, layer(index)), (layer(index), index)] {
      let e = pc.bigram(a, b);
      assert_eq!((e.sfb_base, e.rolling_base, e.scissors_base, e.tap_base), (0.0, 0.0, 0.0, 0.0), "{a} {b}");
    }
    // 한 손이라 손 비율 규칙 없음
    assert!(!pc.violates_hand_usage(1, 0) && !pc.violates_hand_usage(0, 1));
  }

  #[test]
  fn one_hand_trigrams_skip_taps_and_repeated_keys() {
    let (w, c) = (one_hand_weights(), Costs::default());
    let pc = Precomp::one_hand(&[1; crate::layout::CHARS], &w, &c, &EffortGrid::default(), 1.0);
    let (ring, middle, index) = (REF_RING_HOME, REF_MIDDLE_HOME, REF_INDEX_HOME);
    let zero = |t: &TrigramEntry| t.sfs_base == 0.0 && t.roll_base == 0.0 && t.skip_scissors_base == 0.0;
    // 세 글자 롤: 기본층끼리, 엄지층끼리(누른 채로), 엄지층 → 기본층 → 기본층(원샷이 풀림)
    for (a, b, d) in [(ring, middle, index), (layer(ring), layer(middle), layer(index)), (layer(ring), middle, index)] {
      assert!(pc.trigram(a, b, d).roll_base > 0.0, "{a} {b} {d}");
      assert_eq!(pc.window(a, b, d), Window::Roll3In);
    }
    // 엄지 탭이 사이에 끼면 롤·스킵그램 없음
    for (a, b, d) in [(ring, layer(middle), layer(index)), (layer(ring), middle, layer(index))] {
      assert!(zero(pc.trigram(a, b, d)), "{a} {b} {d}");
      assert_eq!(pc.window(a, b, d), Window::Tap);
    }
    // 스킵그램: 검지 홈 → 중지 → 검지 윗줄 (층이 달라도 같은 손가락)
    assert!(pc.trigram(layer(index), middle, REF_INDEX_TOP).sfs_base > 0.0);
    // 같은 키를 잇따라 치는 세 글자는 바이그램에서만 셈
    assert!(zero(pc.trigram(layer(middle), middle, index)));
    // 리다이렉트도 층과 상관없이 같은 손 세 키
    assert!(pc.trigram(layer(ring), index, middle).roll_base < 0.0);
    assert_eq!(pc.window(layer(ring), index, middle), Window::Redirect);
  }

  #[test]
  fn one_hand_layer_effort_scales_thumb_keys_everywhere() {
    let (w, c) = (one_hand_weights(), Costs::default());
    let x1 = Precomp::one_hand(&[1; crate::layout::CHARS], &w, &c, &EffortGrid::default(), 1.0);
    let x3 = Precomp::one_hand(&[1; crate::layout::CHARS], &w, &c, &EffortGrid::default(), 3.0);
    let (middle, index, index_top) = (REF_MIDDLE_HOME, REF_INDEX_HOME, REF_INDEX_TOP);
    // 누름: 엄지층 키 노력값 3배 + 탭 (탭은 그대로 노력값 표의 가장 작은 값)
    assert!((x3.effort(layer(index), 0) - (3.0 * x1.effort(index, 0) + 1.0)).abs() < 1e-6);
    assert_eq!(x3.tap, x1.tap);
    // 기본층 키는 그대로
    assert_eq!(x3.effort(index, 0), x1.effort(index, 0));
    assert_eq!(x3.bigram(middle, index).rolling_base, x1.bigram(middle, index).rolling_base);
    // 엄지층끼리 SFB는 노력값 합에 비례해 3배, 롤은 노력값 합에 반비례해 1/3
    let sfb = |pc: &Precomp| pc.bigram(layer(index), layer(index_top)).sfb_base;
    assert!((sfb(&x3) - 3.0 * sfb(&x1)).abs() < 1e-5);
    let roll = |pc: &Precomp, a: usize, b: usize| pc.bigram(a, b).rolling_base;
    assert!((roll(&x3, layer(middle), layer(index)) - roll(&x1, layer(middle), layer(index)) / 3.0).abs() < 1e-5);
    // 층을 건너면 두 키의 노력값 합으로: 엄지층 중지 홈(3.0) → 기본 검지 홈(1.0)의 롤은 합 2 → 4라 절반
    assert!((roll(&x3, layer(middle), index) - roll(&x1, layer(middle), index) / 2.0).abs() < 1e-5);
    // 탭 너머 스킵그램도 노력값 합으로: 기본 검지 홈(1.0) → 엄지층 검지 윗줄(1.3 → 3.9)
    let x = |pc: &Precomp| pc.bigram(index, layer(index_top)).sfb_base;
    assert!((x(&x3) / x(&x1) - (1.0 + 3.9) / (1.0 + 1.3)).abs() < 1e-5);
  }

  #[test]
  fn two_hand_tables_have_no_taps() {
    let pc = Precomp::new(&[1; crate::layout::CHARS], &crate::test_util::weights(), &Costs::default(), &EffortGrid::default(), 0.45);
    assert!(!pc.one_hand && pc.tap == 0.0);
    assert!(pc.bigrams.iter().all(|e| e.tap_base == 0.0));
    assert!((0..N).all(|p| !pc.on_layer(p) && pc.finger_of(p) == finger(p) && pc.same_key(p, p)));
    assert!(!pc.same_key(COLS + 3, COLS + 6) && !pc.same_hand(COLS + 3, COLS + 6));
  }
}
