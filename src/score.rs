#![allow(clippy::cast_possible_truncation)]

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Score {
  pub effort:   f64,
  pub sfb:      f64,
  pub rolling:  f64,
  pub scissors: f64,
}

// 총점 = -(글자당 비용) × ABS_SCALE (정수 비교 해상도용)
pub const ABS_SCALE: f64 = 1e9;

// 원래 점수 1단위당 비용 (검지 홈 행 키 한 번 = 1). rolling은 보너스라 비용에서 뺌
#[derive(Clone, Copy, Debug, Default)]
pub struct AbsWeights {
  pub effort: f64,
  pub sfb: f64,
  pub scissors: f64,
  pub rolling: f64,
  pub per_letter: f64, // 1 / 전체 글자 빈도
}

// 손 비율 감점: 한쪽 손이 soft_min보다 적게 치면 글자 100개당 penalty × (모자란 %p)²
#[derive(Clone, Copy, Debug, Default)]
pub struct HandRule {
  pub soft_min: f64,
  pub penalty: f64,
}

impl HandRule {
  pub fn from_config(c: &crate::config::Config) -> Self {
    Self { soft_min: f64::from(c.min_hand_usage), penalty: f64::from(c.hand_penalty) }
  }
}

// 총점 계산 기준: [costs]를 환산한 비용 배율, 손 비율 감점, 표시용 무작위 배열 중앙값
#[derive(Default)]
pub struct ScoreNorm {
  pub median: Score,
  pub abs: AbsWeights,
  pub hand: HandRule,
}

impl ScoreNorm {
  // 항목별 무작위 배열 표본의 중앙값 (무작위 배열 대비 변화율 표시용)
  pub fn medians(effort: &[f32], sfb: &[f32], rolling: &[f32], scissors: &[f32]) -> Score {
    fn median(v: &[f32]) -> f64 {
      let mut s = v.to_vec();
      s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
      let n = s.len();
      if n == 0 { return 0.0; }
      if n.is_multiple_of(2) { (f64::from(s[n / 2 - 1]) + f64::from(s[n / 2])) / 2.0 } else { f64::from(s[n / 2]) }
    }
    Score { effort: median(effort), sfb: median(sfb), rolling: median(rolling), scissors: median(scissors) }
  }

  // 항목별 비용 (글자 100개당, 검지 홈 행 키 한 번 = 1, 낮을수록 좋음). rolling은 보너스라 음수
  pub fn cost_breakdown(&self, s: &Score) -> Score {
    let a = self.abs;
    let k = 100.0 * a.per_letter;
    Score { effort: s.effort * a.effort * k, sfb: s.sfb * a.sfb * k, rolling: -s.rolling * a.rolling * k, scissors: s.scissors * a.scissors * k }
  }

  // 무작위 배열 중앙값 대비 변화율 (%, 클수록 좋음)
  pub fn vs_random(&self, s: &Score) -> Score {
    let r = |v: f64, m: f64, better_low: bool| if m == 0.0 { 0.0 } else { (if better_low { m - v } else { v - m }) / m.abs() * 100.0 };
    Score {
      effort: r(s.effort, self.median.effort, true),
      sfb: r(s.sfb, self.median.sfb, true),
      rolling: r(s.rolling, self.median.rolling, false),
      scissors: r(s.scissors, self.median.scissors, true),
    }
  }

  // 손 비율 감점 (글자 100개당). 덜 치는 손이 soft_min 이상이면 0
  pub fn hand_cost(&self, left: u64, right: u64) -> f64 {
    let total = left + right;
    if total == 0 || self.hand.penalty == 0.0 { return 0.0; }
    let share = left.min(right) as f64 / total as f64;
    let deficit_pp = (self.hand.soft_min - share) * 100.0;
    if deficit_pp > 0.0 { self.hand.penalty * deficit_pp * deficit_pp } else { 0.0 }
  }

  pub fn total_score(&self, eff: f64, sfb: f64, roll: f64, sciss: f64, left: u64, right: u64) -> i64 {
    let a = self.abs;
    let cost = (eff * a.effort + sfb * a.sfb + sciss * a.scissors - roll * a.rolling) * a.per_letter
      + self.hand_cost(left, right) / 100.0;
    // 비용을 아주 크게 잡아도 손 비율 위반 표시(PENALTY_SCORE)와 겹치지 않게 그보다 위에서 자름
    ((-cost * ABS_SCALE) as i64).max(crate::layout::PENALTY_SCORE + 1)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn total_is_negative_cost_per_letter() {
    let n = ScoreNorm { abs: AbsWeights { effort: 1.0, sfb: 0.5, scissors: 2.0, rolling: 4.0, per_letter: 0.01 }, ..Default::default() };
    // 비용 = (100×1 + 10×0.5 + 3×2 - 2×4) / 100 글자 = 1.03 / 글자
    assert_eq!(n.total_score(100.0, 10.0, 2.0, 3.0, 1, 1), (-1.03 * ABS_SCALE) as i64);
    // 비용이 낮을수록 총점이 큼
    assert!(n.total_score(100.0, 5.0, 2.0, 3.0, 1, 1) > n.total_score(100.0, 10.0, 2.0, 3.0, 1, 1));
    assert!(n.total_score(100.0, 10.0, 3.0, 3.0, 1, 1) > n.total_score(100.0, 10.0, 2.0, 3.0, 1, 1));
    let b = n.cost_breakdown(&Score { effort: 100.0, sfb: 10.0, rolling: 2.0, scissors: 3.0 });
    assert!((b.effort - 100.0).abs() < 1e-9 && (b.sfb - 5.0).abs() < 1e-9 && (b.rolling + 8.0).abs() < 1e-9 && (b.scissors - 6.0).abs() < 1e-9);
  }

  #[test]
  fn huge_costs_never_reach_the_penalty_marker() {
    let n = ScoreNorm { abs: AbsWeights { effort: 1.0, sfb: 0.0, scissors: 0.0, rolling: 0.0, per_letter: 1.0 }, ..Default::default() };
    // 글자당 비용 1e12 → -1e21은 i64 범위를 넘지만 위반 표시보다 위에서 잘림
    let t = n.total_score(1e12, 0.0, 0.0, 0.0, 1, 1);
    assert!(t > crate::layout::PENALTY_SCORE);
    // 범위 안의 비용은 그대로: 비용이 낮을수록 총점이 큼
    assert_eq!(n.total_score(1.0, 0.0, 0.0, 0.0, 1, 1), -1_000_000_000);
    assert!(n.total_score(1.0, 0.0, 0.0, 0.0, 1, 1) > t);
  }

  #[test]
  fn hand_penalty_is_quadratic_outside_the_free_range() {
    let n = ScoreNorm { hand: HandRule { soft_min: 0.45, penalty: 1.0 }, ..Default::default() };
    // 45~55% 안은 감점 없음
    assert_eq!(n.hand_cost(45, 55), 0.0);
    assert_eq!(n.hand_cost(50, 50), 0.0);
    assert_eq!(n.hand_cost(55, 45), 0.0);
    // 44% → 1, 43% → 4, 42% → 9 (어느 손이 모자라든 같음)
    assert!((n.hand_cost(44, 56) - 1.0).abs() < 1e-9);
    assert!((n.hand_cost(57, 43) - 4.0).abs() < 1e-9);
    assert!((n.hand_cost(42, 58) - 9.0).abs() < 1e-9);
    // 총점에도 그대로 반영 (글자 100개당 감점 → 글자당 /100)
    assert_eq!(n.total_score(0.0, 0.0, 0.0, 0.0, 44, 56), -(0.01 * ABS_SCALE) as i64);
    // 세기 0이면 감점 없음
    let off = ScoreNorm { hand: HandRule { soft_min: 0.45, penalty: 0.0 }, ..Default::default() };
    assert_eq!(off.hand_cost(40, 60), 0.0);
  }

  #[test]
  fn medians_and_vs_random() {
    let m = ScoreNorm::medians(&[1.0, 3.0, 2.0], &[4.0, 2.0], &[5.0], &[]);
    assert_eq!((m.effort, m.sfb, m.rolling, m.scissors), (2.0, 3.0, 5.0, 0.0));
    let n = ScoreNorm { median: Score { effort: 100.0, sfb: 10.0, rolling: 4.0, scissors: 0.0 }, ..Default::default() };
    let v = n.vs_random(&Score { effort: 90.0, sfb: 5.0, rolling: 6.0, scissors: 1.0 });
    assert!((v.effort - 10.0).abs() < 1e-9 && (v.sfb - 50.0).abs() < 1e-9 && (v.rolling - 50.0).abs() < 1e-9 && v.scissors == 0.0);
  }
}
