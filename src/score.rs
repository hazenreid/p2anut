#![allow(clippy::cast_possible_truncation)]
use crate::config::ScoreRates;

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Score {
  pub effort:   f64,
  pub sfb:      f64,
  pub rolling:  f64,
  pub scissors: f64,
}

#[derive(Default)]
pub struct ScoreNorm {
  pub median:  Score,
  pub iqr:     Score,
  // rates를 합이 1이 되도록 정규화한 항목별 가중치
  pub weights: Score,
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

  pub fn total_score(&self, eff: f64, sfb: f64, roll: f64, sciss: f64) -> i64 {
    let n = |v: f64, m: f64, d: f64| ((v - m) / d).clamp(-20.0, 20.0);
    let ne = n(eff,   self.median.effort,   self.iqr.effort);
    let ns = n(sfb,   self.median.sfb,      self.iqr.sfb);
    let nr = n(roll,  self.median.rolling,  self.iqr.rolling);
    let nc = n(sciss, self.median.scissors, self.iqr.scissors);
    (((-ne) * self.weights.effort + (-ns) * self.weights.sfb + nr * self.weights.rolling + (-nc) * self.weights.scissors) * 1e9) as i64
  }
}
