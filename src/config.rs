use std::path::Path;
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Weights {
  pub gap: f32,
  pub row: f32,
  pub center: f32,
  pub outroll: f32,
  // 예전의 redirect 배율은 [costs]의 redirect로 옮김 (설정 파일에 남아 있어도 무시됨)
  // 검지 없는 리다이렉트(나쁜 리다이렉트) 배율. [costs]의 redirect에 곱함. 1(기본) = 구분 없음
  #[serde(default = "default_one")]
  pub bad_redirect: f32,
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
  // 같은 손가락 스킵그램(1·3번 키가 같은 손가락, 가운데는 다른 손가락) 페널티 배율. SFB와 같은 식에 곱해 sfb 항목에 더함.
  // 1(기본) = SFB와 같은 무게 (예전 동작)
  #[serde(default = "default_one")]
  pub sfs: f32,
  // 같은 키 스킵그램(1·3번이 같은 글자, 예: ere, iti) 배율. 같은 글자는 어느 배열에서나 같은 키라 배열로 피할 수 없고,
  // 감점이 그 키의 노력값만 한 번 더 세는 셈이 됨. SFB도 같은 키 연속(ll)은 세지 않음. 없으면 sfs와 같음 (예전 동작)
  #[serde(default)]
  pub sfs_same_key: Option<f32>,
}

fn default_one() -> f32 { 1.0 }

impl Weights {
  pub fn roll_row(&self) -> f32 { self.roll_row.unwrap_or(self.row) }
  pub fn sfs_same_key(&self) -> f32 { self.sfs_same_key.unwrap_or(self.sfs) }
}

// 총점의 기준값: 모든 항목을 "검지 홈 행 키 한 번 = 1" 단위의 비용으로 바꿔 그대로 더함.
// 대표 사건 한 번이 검지 홈 행 키 몇 번을 누르는 만큼인지.
// 나머지 사건은 [weights]의 세부 배율(행·간격·방향·sfs·bad_redirect 등)로 이 값에서 자동으로 정해짐
#[derive(Deserialize, Clone, Debug)]
pub struct Costs {
  // 키 노력값 배율 (1 = 노력값 표 그대로)
  #[serde(default = "default_one")]
  pub effort: f32,
  // SFB 한 번 (왼손 검지 홈 → 검지 윗줄)
  #[serde(default = "default_cost_sfb")]
  pub sfb: f32,
  // 나쁜 방향 풀 시저스 한 번 (중지 아랫줄 → 검지 윗줄)
  #[serde(default = "default_cost_full_scissors")]
  pub full_scissors: f32,
  // 안쪽 롤 한 번으로 아끼는 양 (중지 홈 → 검지 홈)
  #[serde(default = "default_cost_inroll")]
  pub inroll: f32,
  // 리다이렉트 한 번 (홈 행 약지 → 검지 → 중지). 같은 손 세 글자가 방향을 꺾으면 안의 두 쌍은 롤로 치지 않고
  // 이 비용만 매김. 노력값이 큰 키·손가락 간격·행 이동이 있을수록 커지고, 검지가 없으면 bad_redirect 배
  #[serde(default = "default_cost_redirect")]
  pub redirect: f32,
}

fn default_cost_sfb() -> f32 { 2.5 }
fn default_cost_full_scissors() -> f32 { 8.0 }
fn default_cost_inroll() -> f32 { 2.0 }
fn default_cost_redirect() -> f32 { 1.0 }

impl Default for Costs {
  fn default() -> Self {
    Self {
      effort: 1.0, sfb: default_cost_sfb(), full_scissors: default_cost_full_scissors(),
      inroll: default_cost_inroll(), redirect: default_cost_redirect(),
    }
  }
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
fn default_bigram_coverage() -> f64 { 0.99 }
fn default_trigram_coverage() -> f64 { 0.97 }
fn default_sample() -> f64 { 1.0 }
fn default_true() -> bool { true }
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
  // 각 손이 맡아야 할 최소 입력 비율. 이 위면 감점 없음, 아래면 hand_penalty로 감점 (hand_hard_limit 아래는 실격).
  // 예전 이름 min_left_usage(왼손 하한만)도 받지만 이제 양손에 적용됨
  #[serde(alias = "min_left_usage")]
  pub min_hand_usage: f32,
  // 이 비율보다 적게 치는 손이 있으면 실격. 없으면 min_hand_usage와 같음 (예전 동작: 기준 아래는 바로 실격)
  #[serde(default)]
  pub hand_hard_limit: Option<f32>,
  // min_hand_usage보다 모자란 손에 매기는 감점: 글자 100개당 hand_penalty × (모자란 %p)²
  #[serde(default = "default_one")]
  pub hand_penalty: f32,
  // 엘리트끼리 홈행(양손 비중심 8칸) 글자가 최소 몇 개 달라야 하는지 (1 = 조합만 다르면 됨)
  #[serde(default = "default_min_home_diff")]
  pub min_home_diff: u32,
  // 덤프 앞쪽 N개 문서만 분석 (0 = 전체). 앞쪽은 오래된 문서라 편향되므로 빠른 확인용. 고른 표본은 wikipedia_sample
  #[serde(default)]
  pub wikipedia_limit: usize,
  // 문서 표본 비율 (0 초과 1 이하). 제목 해시로 덤프 전체에서 고르게 뽑음. 1(기본) = 전부
  #[serde(default = "default_sample")]
  pub wikipedia_sample: f64,
  // 덤프 주소. 문자열 하나 또는 목록(공식·미러). 목록이면 받기 전에 속도를 재서 빠른 곳부터 씀
  pub wikipedia_dump_url: UrlList,
  // true면 덤프를 result_path 아래에 먼저 받아 두고 파일에서 분석. 파일 이름은 URL의 마지막 부분
  // (날짜판 URL을 바꾸면 파일 이름도 바뀌어서 예전 덤프를 새 것으로 착각하지 않음)
  #[serde(default)]
  pub wikipedia_download: bool,
  // 다 받은 덤프를 Wikimedia가 공개하는 SHA-1(<wiki>-<날짜>-sha1sums.txt)과 비교. 기본 true
  #[serde(default = "default_true")]
  pub wikipedia_verify: bool,
  // 난수 시드. 없으면 실행마다 새로 뽑아 출력하므로, 그 값을 넣으면 같은 실행을 재현할 수 있음
  #[serde(default)]
  pub seed: Option<u64>,
  // 받아 둘 덤프 파일 이름(result_path 기준)을 직접 정할 때만. 설정하면 wikipedia_download와 상관없이 파일로 받음.
  // 끊기면 이어받기, 이미 있으면 다운로드 없이 사용. 둘 다 없으면 스트리밍(병렬·이어서 분석 불가)
  #[serde(default)]
  pub wikipedia_dump_file: Option<String>,
  // 최적화에 쓸 빅그램·트라이그램의 누적 빈도 비율 (1.0 = 전부). 코퍼스 파일에는 전체를 저장하고 불러올 때 적용
  #[serde(default = "default_bigram_coverage")]
  pub bigram_coverage: f64,
  #[serde(default = "default_trigram_coverage")]
  pub trigram_coverage: f64,
  // 코퍼스 빈도 최하위 n글자를 가운데 두 열(검지 안쪽)에 강제 배치. 다른 글자는 절대 센터에 안 옴.
  // 0(기본) = 제약 없음. 2~6 (센터는 6칸, 나머지 24칸에 26-n글자가 들어가야 하므로 2 이상)
  #[serde(default)]
  pub center_rare: usize,
  pub weights: Weights,
  // 예전 설정의 score_normalization, [rates]는 더 이상 쓰지 않으며 남아 있어도 무시됨
  #[serde(default)]
  pub costs: Costs,
  pub sa: SaParams,
  pub ga: GaParams,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum UrlList {
  One(String),
  Many(Vec<String>),
}

// URL의 마지막 경로 부분 (쿼리·조각 제외)
fn url_file_name(url: &str) -> Option<&str> {
  let path = url.split(['?', '#']).next()?;
  path.rsplit('/').next().filter(|s| !s.is_empty())
}

impl Config {
  pub fn hand_hard_limit(&self) -> f32 { self.hand_hard_limit.unwrap_or(self.min_hand_usage) }

  // 비어 있지 않은 덤프 주소들 (적은 순서대로)
  pub fn dump_urls(&self) -> Vec<String> {
    let all = match &self.wikipedia_dump_url {
      UrlList::One(u) => vec![u.clone()],
      UrlList::Many(v) => v.clone(),
    };
    all.into_iter().map(|u| u.trim().to_string()).filter(|u| !u.is_empty()).collect()
  }

  // 덤프를 받아 둘 로컬 경로. None이면 스트리밍
  pub fn dump_path(&self, result_path: &Path) -> Option<std::path::PathBuf> {
    match &self.wikipedia_dump_file {
      Some(file) => Some(result_path.join(file)),
      None if self.wikipedia_download => self.dump_urls().first().and_then(|u| url_file_name(u)).map(|f| result_path.join(f)),
      None => None,
    }
  }

  pub fn validate(&self) -> anyhow::Result<()> {
    if self.elites == 0 { anyhow::bail!("elites must be > 0"); }
    if !(1..=8).contains(&self.min_home_diff) { anyhow::bail!("min_home_diff must be in 1..=8"); }
    let urls = self.dump_urls();
    if urls.is_empty() && self.wikipedia_dump_file.is_none() {
      anyhow::bail!("wikipedia_dump_url must not be empty (unless wikipedia_dump_file is set)");
    }
    // 여러 주소는 같은 파일이어야 함 (중간에 주소를 바꿔 이어받으므로)
    if let Some(first) = urls.first() {
      let name = url_file_name(first);
      if let Some(other) = urls.iter().find(|u| url_file_name(u) != name) {
        anyhow::bail!("all wikipedia_dump_url entries must end with the same file name ({first} vs {other})");
      }
    }
    if self.wikipedia_dump_file.as_deref() == Some("") { anyhow::bail!("wikipedia_dump_file must not be empty"); }
    if self.wikipedia_download && self.wikipedia_dump_file.is_none() && urls.first().and_then(|u| url_file_name(u)).is_none() {
      anyhow::bail!("wikipedia_download needs a wikipedia_dump_url that ends with a file name");
    }
    if !(self.wikipedia_sample > 0.0 && self.wikipedia_sample <= 1.0) {
      anyhow::bail!("wikipedia_sample must be in (0.0, 1.0]");
    }
    for (name, v) in [("bigram_coverage", self.bigram_coverage), ("trigram_coverage", self.trigram_coverage)] {
      if !(v > 0.0 && v <= 1.0) { anyhow::bail!("{name} must be in (0.0, 1.0]"); }
    }
    if self.center_rare != 0 && !(2..=6).contains(&self.center_rare) {
      anyhow::bail!("center_rare must be 0 (off) or 2..=6");
    }
    let w = &self.weights;
    if w.gap < 0.0 || w.row < 0.0 || w.center < 0.0 || w.outroll < 0.0 || w.roll_row() < 0.0 || w.half_scissors < 0.0 || w.full_scissors_good < 0.0 || w.skip_scissors < 0.0 || w.sfs < 0.0 || w.sfs_same_key() < 0.0 || w.bad_redirect < 0.0 {
      anyhow::bail!("weights.gap, row, roll_row, center, outroll, half_scissors, full_scissors_good, skip_scissors, sfs, sfs_same_key, bad_redirect must be non-negative");
    }
    let c = &self.costs;
    if c.effort < 0.0 || c.sfb < 0.0 || c.full_scissors < 0.0 || c.inroll < 0.0 || c.redirect < 0.0 {
      anyhow::bail!("costs.effort, sfb, full_scissors, inroll, redirect must be non-negative");
    }
    if c.effort + c.sfb + c.full_scissors + c.inroll <= 0.0 {
      anyhow::bail!("at least one cost must be > 0");
    }
    let sa = &self.sa;
    if !(0.0..=0.5).contains(&self.min_hand_usage) {
      anyhow::bail!("min_hand_usage must be between 0.0 and 0.5 (applies to each hand)");
    }
    if !(0.0..=self.min_hand_usage).contains(&self.hand_hard_limit()) {
      anyhow::bail!("hand_hard_limit must be between 0.0 and min_hand_usage");
    }
    if self.hand_penalty < 0.0 { anyhow::bail!("hand_penalty must be non-negative"); }
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

#[cfg(test)]
mod tests {
  use super::*;

  fn config(extra: &str) -> Config {
    let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml")).unwrap();
    // 기본 config.toml에서 덤프·커버리지 항목을 빼고 테스트할 값만 넣음
    let mut table: toml::Table = toml::from_str(&base).unwrap();
    table.retain(|k, _| !k.starts_with("wikipedia_") && !k.ends_with("_coverage"));
    table.extend(toml::from_str::<toml::Table>(extra).unwrap());
    toml::Value::Table(table).try_into().unwrap()
  }

  #[test]
  fn dump_path_comes_from_url_or_explicit_file() {
    let root = Path::new("/r");
    let c = config("wikipedia_dump_url = \"https://m.org/enwiki/20261001/enwiki-20261001-pages-articles-multistream.xml.bz2?x=1\"\nwikipedia_download = true");
    assert_eq!(c.dump_path(root), Some(root.join("enwiki-20261001-pages-articles-multistream.xml.bz2")));
    c.validate().unwrap();
    let c = config("wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\nwikipedia_dump_file = \"mine.bz2\"");
    assert_eq!(c.dump_path(root), Some(root.join("mine.bz2")));
    let c = config("wikipedia_dump_url = \"https://m.org/a.xml.bz2\"");
    assert_eq!(c.dump_path(root), None);
    assert_eq!((c.wikipedia_limit, c.wikipedia_sample, c.bigram_coverage, c.trigram_coverage), (0, 1.0, 0.99, 0.97));
  }

  #[test]
  fn dump_url_accepts_one_or_many() {
    let c = config("wikipedia_dump_url = [\"https://a.org/d/x.xml.bz2\", \" \", \"https://b.org/mirror/d/x.xml.bz2\"]\nwikipedia_download = true");
    assert_eq!(c.dump_urls(), vec!["https://a.org/d/x.xml.bz2", "https://b.org/mirror/d/x.xml.bz2"]);
    assert_eq!(c.dump_path(Path::new("/r")), Some(Path::new("/r").join("x.xml.bz2")));
    c.validate().unwrap();
    // 파일 이름이 다른 주소가 섞이면 거부
    let c = config("wikipedia_dump_url = [\"https://a.org/d/x.xml.bz2\", \"https://b.org/d/y.xml.bz2\"]");
    assert!(c.validate().unwrap_err().to_string().contains("same file name"));
    assert!(config("wikipedia_dump_url = []").validate().is_err());
  }

  #[test]
  fn old_scoring_settings_are_ignored() {
    // 예전 설정 파일의 score_normalization, [rates]가 남아 있어도 읽히고 [costs] 기본값을 씀
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    let mut table: toml::Table = toml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml")).unwrap()).unwrap();
    table.remove("costs");
    table.extend(toml::from_str::<toml::Table>(&format!("{url}score_normalization = \"iqr\"\n[rates]\neffort = 0.2\nsfb = 0.2\nrolling = 0.3\nscissors = 0.3\n")).unwrap());
    // 예전의 [weights] redirect 배율도 무시
    table.get_mut("weights").unwrap().as_table_mut().unwrap().insert("redirect".into(), toml::Value::Float(-2.0));
    let c: Config = toml::Value::Table(table).try_into().unwrap();
    c.validate().unwrap();
    assert_eq!((c.costs.sfb, c.costs.full_scissors, c.costs.inroll, c.costs.effort, c.costs.redirect), (2.5, 8.0, 2.0, 1.0, 1.0));
  }

  #[test]
  fn hand_limits_default_to_old_behavior() {
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    let mut base: toml::Table = toml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml")).unwrap()).unwrap();
    base.retain(|k, _| !k.starts_with("wikipedia_") && !k.starts_with("hand_"));
    let with = |extra: &str| -> Config {
      let mut t = base.clone();
      t.extend(toml::from_str::<toml::Table>(&format!("{url}{extra}")).unwrap());
      toml::Value::Table(t).try_into().unwrap()
    };
    // 지정하지 않으면 실격 기준 = min_hand_usage (예전 동작)
    let c = with("");
    assert_eq!(c.hand_hard_limit(), c.min_hand_usage);
    c.validate().unwrap();
    let c = with("hand_hard_limit = 0.40\nhand_penalty = 2.0");
    assert_eq!((c.hand_hard_limit(), c.hand_penalty), (0.40, 2.0));
    c.validate().unwrap();
    // 실격 기준은 감점 기준보다 클 수 없음
    assert!(with("hand_hard_limit = 0.47").validate().is_err());
    assert!(with("hand_penalty = -1.0").validate().is_err());
  }

  #[test]
  fn validate_rejects_bad_sample_and_coverage() {
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    assert!(config(&format!("{url}wikipedia_sample = 0.0")).validate().is_err());
    assert!(config(&format!("{url}wikipedia_sample = 1.5")).validate().is_err());
    assert!(config(&format!("{url}wikipedia_sample = 0.1")).validate().is_ok());
    assert!(config(&format!("{url}trigram_coverage = 0.0")).validate().is_err());
    assert!(config(&format!("{url}bigram_coverage = 1.0")).validate().is_ok());
    assert!(config("wikipedia_dump_url = \"https://m.org/\"\nwikipedia_download = true").validate().is_err());
  }
}
