#![allow(clippy::cast_possible_truncation)]
use crate::config::ScoreRates;

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Score {
  pub effort:   f64,
  pub sfb:      f64,
  pub rolling:  f64,
  pub scissors: f64,
}

// 항목별 정규화 점수를 자르는 한계 (한 항목이 총점을 독차지하지 않도록)
pub const Z_CLAMP: f64 = 20.0;

// absolute 방식에서 총점 = -(글자당 비용) × ABS_SCALE (정수 비교 해상도용)
pub const ABS_SCALE: f64 = 1e9;

// absolute 방식의 원래 점수 1단위당 비용 (검지 홈 행 키 한 번 = 1). rolling은 보너스라 비용에서 뺌
#[derive(Clone, Copy, Debug)]
pub struct AbsWeights {
  pub effort: f64,
  pub sfb: f64,
  pub scissors: f64,
  pub rolling: f64,
  pub per_letter: f64, // 1 / 전체 글자 빈도
}

#[derive(Default)]
pub struct ScoreNorm {
  pub median:  Score,
  pub iqr:     Score,
  // rates를 합이 1이 되도록 정규화한 항목별 가중치 (iqr 방식)
  pub weights: Score,
  // 있으면 absolute 방식으로 총점 계산
  pub abs: Option<AbsWeights>,
}

impl ScoreNorm {
  pub fn from_samples(
    effort: &[f32], sfb: &[f32], rolling: &[f32], scissors: &[f32],
  ) -> Self {
    fn median_iqr(v: &[f32]) -> (f64, f64) {
      let mut s = v.to_vec();
      s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
      let n = s.len();
      if n == 0 { return (0.0, 0.01); }
      let med = if n.is_multiple_of(2) { f64::from(s[n/2 - 1] + s[n/2]) / 2.0 } else { f64::from(s[n/2]) };
      let iqr = (f64::from(s[3 * n / 4]) - f64::from(s[n / 4])).max(med.abs() * 0.05).max(0.01);
      (med, iqr)
    }

    let (em, ei) = median_iqr(effort);
    let (sm, si) = median_iqr(sfb);
    let (rm, ri) = median_iqr(rolling);
    let (cm, ci) = median_iqr(scissors);

    ScoreNorm {
      median:  Score { effort: em, sfb: sm, rolling: rm, scissors: cm },
      iqr:     Score { effort: ei, sfb: si, rolling: ri, scissors: ci },
      weights: Score::default(),
      abs: None,
    }
  }

  pub fn set_rates(&mut self, rates: &ScoreRates) {
    let total_rate = f64::from(rates.effort + rates.sfb + rates.rolling + rates.scissors);
    if total_rate > 0.0 {
      self.weights = Score {
        effort:   f64::from(rates.effort) / total_rate,
        sfb:      f64::from(rates.sfb) / total_rate,
        rolling:  f64::from(rates.rolling) / total_rate,
        scissors: f64::from(rates.scissors) / total_rate,
      };
    }
  }

  // 항목별 정규화 점수: 무작위 배열의 중앙값에서 IQR 몇 배만큼 좋은지 (클수록 좋음, 자르기 전 값)
  pub fn goodness(&self, s: &Score) -> Score {
    Score {
      effort:   -(s.effort   - self.median.effort)   / self.iqr.effort,
      sfb:      -(s.sfb      - self.median.sfb)      / self.iqr.sfb,
      rolling:   (s.rolling  - self.median.rolling)  / self.iqr.rolling,
      scissors: -(s.scissors - self.median.scissors) / self.iqr.scissors,
    }
  }

  // absolute 방식의 항목별 비용 (글자 100개당, 검지 홈 행 키 한 번 = 1, 낮을수록 좋음). rolling은 보너스라 음수
  pub fn cost_breakdown(&self, s: &Score) -> Option<Score> {
    let a = self.abs?;
    let k = 100.0 * a.per_letter;
    Some(Score { effort: s.effort * a.effort * k, sfb: s.sfb * a.sfb * k, rolling: -s.rolling * a.rolling * k, scissors: s.scissors * a.scissors * k })
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

  pub fn total_score(&self, eff: f64, sfb: f64, roll: f64, sciss: f64) -> i64 {
    if let Some(a) = self.abs {
      let cost = (eff * a.effort + sfb * a.sfb + sciss * a.scissors - roll * a.rolling) * a.per_letter;
      return (-cost * ABS_SCALE) as i64;
    }
    let n = |v: f64, m: f64, d: f64| ((v - m) / d).clamp(-Z_CLAMP, Z_CLAMP);
    let ne = n(eff,   self.median.effort,   self.iqr.effort);
    let ns = n(sfb,   self.median.sfb,      self.iqr.sfb);
    let nr = n(roll,  self.median.rolling,  self.iqr.rolling);
    let nc = n(sciss, self.median.scissors, self.iqr.scissors);
    (((-ne) * self.weights.effort + (-ns) * self.weights.sfb + nr * self.weights.rolling + (-nc) * self.weights.scissors) * 1e9) as i64
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn absolute_total_is_negative_cost_per_letter() {
    let n = ScoreNorm { abs: Some(AbsWeights { effort: 1.0, sfb: 0.5, scissors: 2.0, rolling: 4.0, per_letter: 0.01 }), ..Default::default() };
    // 비용 = (100×1 + 10×0.5 + 3×2 - 2×4) / 100 글자 = 1.03 / 글자
    assert_eq!(n.total_score(100.0, 10.0, 2.0, 3.0), (-1.03 * ABS_SCALE) as i64);
    // 비용이 낮을수록 총점이 큼
    assert!(n.total_score(100.0, 5.0, 2.0, 3.0) > n.total_score(100.0, 10.0, 2.0, 3.0));
    assert!(n.total_score(100.0, 10.0, 3.0, 3.0) > n.total_score(100.0, 10.0, 2.0, 3.0));
    let b = n.cost_breakdown(&Score { effort: 100.0, sfb: 10.0, rolling: 2.0, scissors: 3.0 }).unwrap();
    assert!((b.effort - 100.0).abs() < 1e-9 && (b.sfb - 5.0).abs() < 1e-9 && (b.rolling + 8.0).abs() < 1e-9 && (b.scissors - 6.0).abs() < 1e-9);
    assert!(ScoreNorm::default().cost_breakdown(&b).is_none());
  }
}
