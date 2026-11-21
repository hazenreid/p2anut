// 테스트 전용: 고정 영문 텍스트로 만든 작은 코퍼스와 기본 설정값
use crate::config::{GaParams, SaParams, ScoreRates, Weights};
use crate::corpus::{CorpusArrays, CorpusStats, FastCorpus};
use crate::precomp::Precomp;
use crate::score::ScoreNorm;

const TEXT: &str = "the quick brown fox jumps over the lazy dog while seven wizards quietly judge \
boxing matches in the old town square people often think about what happens when nobody is \
watching and they write long letters to friends who live far away from the coast every morning \
the baker opens his shop before sunrise and the smell of fresh bread fills the narrow streets \
students gather near the library to discuss their projects and share notes about history science \
and mathematics while the river flows gently under the stone bridge toward the distant mountains";

pub fn stats() -> CorpusStats {
  let text = TEXT.repeat(20);
  let mut f = FastCorpus::new();
  crate::analyze_bytes(text.as_bytes(), &mut f);
  f.into_stats()
}

pub fn weights() -> Weights {
  Weights { gap: 0.78, row: 0.97, center: 2.5, outroll: 0.70, redirect: 0.30, roll_row: None, half_scissors: 1.0, full_scissors_good: 0.5, skip_scissors: 0.5 }
}

pub fn rates() -> ScoreRates {
  ScoreRates { effort: 0.20, sfb: 0.25, rolling: 0.30, scissors: 0.25 }
}

pub fn sa() -> SaParams {
  SaParams { max_no_improve: 300, accept_start: 0.3, cooling: 0.98 }
}

pub fn ga() -> GaParams {
  GaParams { population: 8, tournament: 3, max_stall: None }
}

pub fn precomp_norm(stats: &CorpusStats) -> (Precomp, ScoreNorm) {
  precomp_norm_for(&stats.to_arrays(true))
}

pub fn precomp_norm_for(arrays: &CorpusArrays) -> (Precomp, ScoreNorm) {
  let precomp = Precomp::new(&arrays.freq, &weights(), 0.45);
  let norm = crate::layout::init_score_norm(arrays, &precomp, &rates(), 0);
  (precomp, norm)
}
