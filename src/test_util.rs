// 테스트 전용: 고정 영문 텍스트로 만든 작은 코퍼스와 기본 설정값
use crate::config::{SaParams, Weights};
use crate::corpus::{CorpusArrays, CorpusStats, FastCorpus};
use crate::precomp::Precomp;
use crate::score::ScoreNorm;

pub const TEXT: &str = "the quick brown fox jumps over the lazy dog while seven wizards quietly judge \
boxing matches in the old town square people often think about what happens when nobody is \
watching and they write long letters to friends who live far away from the coast every morning \
the baker opens his shop before sunrise and the smell of fresh bread fills the narrow streets \
students gather near the library to discuss their projects and share notes about history science \
and mathematics while the river flows gently under the stone bridge toward the distant mountains";

pub fn stats() -> CorpusStats {
  let text = TEXT.repeat(20);
  let mut f = FastCorpus::new();
  crate::wiki::analyze_bytes(text.as_bytes(), &mut f);
  f.into_stats()
}

pub fn weights() -> Weights {
  Weights { gap: 0.78, row: 0.97, center: 2.5, outroll: 0.70, bad_redirect: 1.0, roll_row: None, half_scissors: 1.0, full_scissors_good: 0.5, skip_scissors: 0.5, sfs: 1.0, sfs_same_key: None }
}

pub fn sa() -> SaParams {
  SaParams { max_no_improve: 300, accept_start: 0.3, cooling: 0.98, starts_per_batch: 8, stop_after_hits: 2, max_starts: 64 }
}

pub fn precomp_norm(stats: &CorpusStats) -> (Precomp, ScoreNorm) {
  precomp_norm_for(&stats.to_arrays())
}

// 부드러운 손 비율 감점: 45% 아래는 감점, 40% 아래만 실격
pub fn precomp_norm_soft_hand(arrays: &CorpusArrays) -> (Precomp, ScoreNorm) {
  let precomp = Precomp::new(&arrays.freq, &weights(), &crate::config::Costs::default(), &crate::config::EffortGrid::default(), 0.40);
  let hand = crate::score::HandRule { soft_min: 0.45, penalty: 1.0 };
  let norm = crate::layout::init_score_norm(arrays, &precomp, &crate::config::Costs::default(), hand);
  (precomp, norm)
}

pub fn precomp_norm_for(arrays: &CorpusArrays) -> (Precomp, ScoreNorm) {
  let precomp = Precomp::new(&arrays.freq, &weights(), &crate::config::Costs::default(), &crate::config::EffortGrid::default(), 0.45);
  let norm = crate::layout::init_score_norm(arrays, &precomp, &crate::config::Costs::default(), crate::score::HandRule::default());
  (precomp, norm)
}

// 한손 표의 엄지층 노력값 배율 (설정 기본값과 같음)
pub const ONE_HAND_LAYER_EFFORT: f32 = 3.0;

// 한손 표 (왼손 3×5 + 엄지층, 손 비율 규칙 없음)
pub fn precomp_norm_one_hand(arrays: &CorpusArrays) -> (Precomp, ScoreNorm) {
  let precomp = Precomp::one_hand(&arrays.freq, &weights(), &crate::config::Costs::default(), &crate::config::EffortGrid::default(), ONE_HAND_LAYER_EFFORT);
  let norm = crate::layout::init_score_norm(arrays, &precomp, &crate::config::Costs::default(), crate::score::HandRule::default());
  (precomp, norm)
}
