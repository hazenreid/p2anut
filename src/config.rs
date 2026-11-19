use std::path::Path;
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Weights {
  pub gap: f32,
  pub row: f32,
  pub center: f32,
  pub outroll: f32,
  pub redirect: f32,
  // 롤(바이그램·트라이그램)에서 행이 바뀔 때마다 곱하는 값. 없으면 row와 같음 (row는 SFB/scissors의 행 페널티에도 쓰임)
  #[serde(default)]
  pub roll_row: Option<f32>,
  // 나쁜 방향 하프 시저스(인접 손가락, 한 줄 차이, 긴 손가락이 더 아래 행) 페널티 배율.
  // 이 바이그램은 풀 시저스처럼 롤이 아닌 시저스로 분류되고, 같은 식(effort 합 × (1 + row × 행차이))에 곱함
  #[serde(default)]
  pub half_scissors: f32,
  // 편한 방향 풀 시저스(긴 손가락이 위, 예: 중지 윗줄 + 검지 아랫줄) 페널티 배율. 나쁜 방향은 1.0.
  // 1(기본) = 방향 구분 없음 (예전 동작)
  #[serde(default = "default_one")]
  pub full_scissors_good: f32,
  // 스킵 풀 시저스(1·3번 키가 인접 손가락 두 줄 차이) 페널티 배율. 풀 시저스 식에 곱함. 0(기본) = 끔
  #[serde(default)]
  pub skip_scissors: f32,
}

fn default_one() -> f32 { 1.0 }

impl Weights {
  pub fn roll_row(&self) -> f32 { self.roll_row.unwrap_or(self.row) }
}

#[derive(Deserialize, Clone)]
pub struct ScoreRates {
  pub effort: f32,
  pub sfb: f32,
  pub rolling: f32,
  pub scissors: f32,
}

#[derive(Deserialize, Clone)]
pub struct SaParams {
  pub max_no_improve: usize,
  // 초기 온도: 평균적인 악화 이동을 이 확률로 수락하도록 샘플링해서 결정 (0 = 담금질 끔, 개선만 수락)
  #[serde(default)]
  pub accept_start: f64,
  // Phase 2 스텝마다 온도에 곱하는 냉각 계수
  #[serde(default = "default_cooling")]
  pub cooling: f64,
}

fn default_cooling() -> f64 { 0.998 }
fn default_min_home_diff() -> u32 { 1 }

#[derive(Deserialize, Clone)]
pub struct GaParams {
  pub population: usize,
  pub tournament: usize,
  // 최고 점수가 이 세대 수 동안 안 오르면 해당 엘리트 탐색 종료. 없으면 population 값 (예전 동작)
  #[serde(default)]
  pub max_stall: Option<usize>,
}

impl GaParams {
  pub fn max_stall(&self) -> usize { self.max_stall.unwrap_or(self.population) }
}

#[derive(Deserialize)]
pub struct Config {
  pub corpus_file: String,
  pub result_file: String,
  pub elites: usize,
  // 각 손이 맡아야 할 최소 입력 비율. 예전 이름 min_left_usage(왼손 하한만)도 받지만 이제 양손에 적용됨
  #[serde(alias = "min_left_usage")]
  pub min_hand_usage: f32,
  // 엘리트끼리 홈행(양손 비중심 8칸) 글자가 최소 몇 개 달라야 하는지 (1 = 조합만 다르면 됨)
  #[serde(default = "default_min_home_diff")]
  pub min_home_diff: u32,
  pub wikipedia_limit: usize,  // 0 = 전체 덤프
  pub wikipedia_dump_url: String,
  // 난수 시드. 없으면 실행마다 새로 뽑아 출력하므로, 그 값을 넣으면 같은 실행을 재현할 수 있음
  #[serde(default)]
  pub seed: Option<u64>,
  // 설정 시 덤프를 이 파일(result_path 기준)로 먼저 받아 두고 파일에서 분석. 끊기면 이어받기.
  // 이미 있으면 다운로드 없이 바로 사용. 없으면(기본) 예전처럼 스트리밍
  #[serde(default)]
  pub wikipedia_dump_file: Option<String>,
  // 코퍼스 빈도 최하위 n글자를 가운데 두 열(검지 안쪽)에 강제 배치. 다른 글자는 절대 센터에 안 옴.
  // 0(기본) = 제약 없음. 2~6 (센터는 6칸, 나머지 24칸에 26-n글자가 들어가야 하므로 2 이상)
  #[serde(default)]
  pub center_rare: usize,
  pub weights: Weights,
  pub rates: ScoreRates,
  pub sa: SaParams,
  pub ga: GaParams,
}

impl Config {
  pub fn validate(&self) -> anyhow::Result<()> {
    if self.elites == 0 { anyhow::bail!("elites must be > 0"); }
    if !(1..=8).contains(&self.min_home_diff) { anyhow::bail!("min_home_diff must be in 1..=8"); }
    if self.wikipedia_dump_url.is_empty() && self.wikipedia_dump_file.is_none() {
      anyhow::bail!("wikipedia_dump_url must not be empty (unless wikipedia_dump_file is set)");
    }
    if self.wikipedia_dump_file.as_deref() == Some("") { anyhow::bail!("wikipedia_dump_file must not be empty"); }
    if self.center_rare != 0 && !(2..=6).contains(&self.center_rare) {
      anyhow::bail!("center_rare must be 0 (off) or 2..=6");
    }
    let w = &self.weights;
    if w.gap < 0.0 || w.row < 0.0 || w.center < 0.0 || w.outroll < 0.0 || w.roll_row() < 0.0 || w.half_scissors < 0.0 || w.full_scissors_good < 0.0 || w.skip_scissors < 0.0 {
      anyhow::bail!("weights.gap, row, roll_row, center, outroll, half_scissors, full_scissors_good, skip_scissors must be non-negative (redirect may be negative)");
    }
    if w.redirect < -5.0 || w.redirect > 5.0 {
      anyhow::bail!("weights.redirect must be in [-5.0, 5.0]");
    }
    let r = &self.rates;
    if r.effort < 0.0 || r.sfb < 0.0 || r.rolling < 0.0 || r.scissors < 0.0 {
      anyhow::bail!("rates must be non-negative");
    }
    if r.effort + r.sfb + r.rolling + r.scissors <= 0.0 {
      anyhow::bail!("at least one rate must be > 0");
    }
    let sa = &self.sa;
    if !(0.0..=0.5).contains(&self.min_hand_usage) {
      anyhow::bail!("min_hand_usage must be between 0.0 and 0.5 (applies to each hand)");
    }
    if sa.max_no_improve == 0 { anyhow::bail!("sa.max_no_improve must be > 0"); }
    if !(0.0..1.0).contains(&sa.accept_start) { anyhow::bail!("sa.accept_start must be in [0.0, 1.0)"); }
    if !(sa.cooling > 0.0 && sa.cooling <= 1.0) { anyhow::bail!("sa.cooling must be in (0.0, 1.0]"); }
    let ga = &self.ga;
    if ga.population == 0 { anyhow::bail!("ga.population must be > 0"); }
    if ga.tournament == 0 { anyhow::bail!("ga.tournament must be > 0"); }
    if ga.tournament > ga.population { anyhow::bail!("ga.tournament must be <= ga.population"); }
    if ga.max_stall == Some(0) { anyhow::bail!("ga.max_stall must be > 0"); }
    Ok(())
  }
}

pub fn load_config(path: &Path) -> anyhow::Result<Config> {
  let content = std::fs::read_to_string(path)?;
  let config: Config = toml::from_str(&content)?;
  config.validate()?;
  Ok(config)
}
