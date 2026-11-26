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
  // 가운데 열로 옆으로 뻗기(LSB) 한 번 (중지 홈 → 가운데 홈). 같은 손에서 가운데 열 키와 다른 손가락 키를 잇따라 치는 경우.
  // 다른 LSB는 두 키의 노력값 합과 행 이동(1 + row × 행 차이)에 비례. scissors 항목에 들어감.
  // 없으면 예전 동작 (노력값 합 × center × 행 배율을 시저스와 같은 단위로: 지금 설정이면 약 12.6)
  #[serde(default)]
  pub lateral: Option<f32>,
}

fn default_cost_sfb() -> f32 { 2.5 }
fn default_cost_full_scissors() -> f32 { 8.0 }
fn default_cost_inroll() -> f32 { 2.0 }
fn default_cost_redirect() -> f32 { 1.0 }

// 키마다의 노력값 (검지 홈 = 1.0 기준). 왼손 새끼 → 약지 → 중지 → 검지 → 가운데 열 순서로 5칸씩, 오른손은 좌우 대칭.
// 노력값 비용(글자 빈도 × 노력값)에 쓰이고, SFB·시저스·롤·리다이렉트의 크기도 키 노력값 합에 따라 달라짐.
// [costs]의 대표 사건 비용은 이 표를 바꿔도 그대로 유지됨 (대표 사건 기준으로 다시 맞추므로)
#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct EffortGrid {
  #[serde(default = "default_effort_top")]
  pub top: [f32; 5],
  #[serde(default = "default_effort_home")]
  pub home: [f32; 5],
  #[serde(default = "default_effort_bottom")]
  pub bottom: [f32; 5],
}

fn default_effort_top() -> [f32; 5] { [2.4, 1.3, 1.1, 1.3, 3.5] }
fn default_effort_home() -> [f32; 5] { [1.6, 1.1, 1.0, 1.0, 3.0] }
fn default_effort_bottom() -> [f32; 5] { [2.2, 1.5, 1.3, 1.1, 4.0] }

impl Default for EffortGrid {
  fn default() -> Self { Self { top: default_effort_top(), home: default_effort_home(), bottom: default_effort_bottom() } }
}

impl EffortGrid {
  // 3행 × 10열 전체 (오른손은 왼손을 좌우로 뒤집어 채움)
  pub fn grid(&self) -> [[f32; 10]; 3] {
    let row = |half: &[f32; 5]| {
      let mut r = [0.0f32; 10];
      for (c, &v) in half.iter().enumerate() { r[c] = v; r[9 - c] = v; }
      r
    };
    [row(&self.top), row(&self.home), row(&self.bottom)]
  }
}

impl Default for Costs {
  fn default() -> Self {
    Self {
      effort: 1.0, sfb: default_cost_sfb(), full_scissors: default_cost_full_scissors(),
      inroll: default_cost_inroll(), redirect: default_cost_redirect(), lateral: None,
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
  // 엘리트마다 무작위 배열에서 시작하는 담금질을 이만큼씩 묶어 동시에 돌림 (최소 시작 횟수이기도 함)
  #[serde(default = "default_starts_per_batch")]
  pub starts_per_batch: usize,
  // 같은 최고점이 이만큼 나오면 그 엘리트 탐색을 끝냄
  #[serde(default = "default_stop_after_hits")]
  pub stop_after_hits: usize,
  // 시작 횟수 상한 (최고점이 그만큼 안 나와도 여기서 끝냄)
  #[serde(default = "default_max_starts")]
  pub max_starts: usize,
}

fn default_cooling() -> f64 { 0.998 }
fn default_starts_per_batch() -> usize { 40 }
fn default_stop_after_hits() -> usize { 3 }
fn default_max_starts() -> usize { 2000 }
fn default_bigram_coverage() -> f64 { 0.99 }
fn default_trigram_coverage() -> f64 { 0.97 }
fn default_sample() -> f64 { 1.0 }
fn default_true() -> bool { true }
fn default_min_home_diff() -> u32 { 1 }
fn default_layer_effort() -> f32 { 3.0 }

#[derive(Deserialize, Clone)]
pub struct Config {
  pub corpus_file: String,
  // 글자 배열 코퍼스에 섞을 러스트 코드의 비율 (글자 수 기준, 0~1). 0(기본) = 위키 영어만. 코드는 [code]의 크레이트를
  // 위키와 같은 규칙으로 센 것 (code.letters.bin). 결과 파일은 비율과 상관없이 하나 (result_file)
  #[serde(default)]
  pub code_share: f64,
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
  // 여러 미러에서 덩어리를 나눠 동시에 받기 (미러마다 연결 하나, 같은 서버의 여러 주소는 하나만,
  // 가장 빠른 곳의 20% 미만은 예비). false면 가장 빠른 한 곳에서 받다가 계속 실패하면 다음 곳으로. 기본 true
  #[serde(default = "default_true")]
  pub wikipedia_parallel_download: bool,
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
  // 0(기본) = 제약 없음. 3~6 (센터는 6칸, 키 27개(a-z와 스페이스) 중 나머지가 바깥 24칸에 다 들어가야 하므로 3 이상).
  // 한손 배열에는 쓰지 않음 (center_rare_letters)
  #[serde(default)]
  pub center_rare: usize,
  // 한손 키보드 (왼손 3×5 + 엄지 원샷 레이어 키): 27키를 기본층 15칸과 엄지층 15칸에 둠. 손 비율 규칙은 쓰지 않고,
  // 결과는 result_file 이름에 -one-hand를 붙인 파일에 따로 저장 (result_file_name). 심볼 레이어는 아직 없음. 기본 false
  #[serde(default)]
  pub one_hand: bool,
  // 한손: 엄지층 키의 노력값 배율 (같은 자리 기본층 노력값 × 이 값). 누름 노력값과 SFB·시저스·롤·리다이렉트 크기에 모두 씀.
  // 엄지 탭(노력값 표의 가장 작은 값)은 따로 그대로. 기본 3
  #[serde(default = "default_layer_effort")]
  pub one_hand_layer_effort: f32,
  // 예전 설정의 space_key는 없어짐: 스페이스는 항상 배열의 한 칸(27번째 키). 설정에 남아 있어도 무시됨
  pub weights: Weights,
  // 예전 설정의 score_normalization, [rates]는 더 이상 쓰지 않으며 남아 있어도 무시됨
  #[serde(default)]
  pub costs: Costs,
  #[serde(default)]
  pub effort: EffortGrid,
  pub sa: SaParams,
  // 예전 설정의 [ga](유전 알고리즘)는 더 이상 쓰지 않으며 남아 있어도 무시됨
  // 러스트 코드 코퍼스 (analyze-code): 글자 배열(code_share)과 심볼 레이어가 같이 씀. 없으면 기본값
  #[serde(default)]
  pub code: CodeConfig,
  // 심볼 레이어 최적화 (symbols 명령). 없으면 기본값
  #[serde(default)]
  pub symbols: SymbolsConfig,
}

// 러스트 코드 코퍼스
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct CodeConfig {
  // 코드 코퍼스 파일 (analyze-code가 만듦, result_path 기준)
  pub corpus_file: String,
  // 코드 코퍼스로 쓸 크레이트. crates.io에서 각각 최신 안정 버전만 받음 (의존성은 받지 않음). 크레이트마다 같은 무게
  pub crates: Vec<String>,
}

impl Default for CodeConfig {
  fn default() -> Self {
    Self {
      corpus_file: "code.bin".into(),
      // 라이브러리 20개 + 프로그램 20개 (고른 규칙은 config.toml)
      crates: ["tauri", "sway-core", "typst-layout", "bevy_ecs", "parcel-resolver", "polars-plan", "servo", "swc", "tokio", "surrealdb",
        "yew", "jj-lib", "egui", "iced_widget", "pingora-http", "tree-sitter", "rocket", "turso", "actix-web", "oxc_ast",
        "deno", "uv", "ripgrep", "alacritty", "bat", "starship", "yazi-scheduler", "sniffnet", "zoxide", "just",
        "mise", "rustfs", "zellij", "git-delta", "atuin", "fnm", "difftastic", "eza", "gitui", "rustpython-codegen"]
        .iter().map(|s| (*s).to_string()).collect(),
    }
  }
}

impl CodeConfig {
  pub fn validate(&self) -> anyhow::Result<()> {
    if self.crates.is_empty() { anyhow::bail!("code.crates must name at least one crate"); }
    if let Some(n) = self.crates.iter().find(|n| n.is_empty() || !n.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')) {
      anyhow::bail!("code.crates holds an invalid crate name {n:?}");
    }
    if self.corpus_file.is_empty() { anyhow::bail!("code.corpus_file must not be empty"); }
    Ok(())
  }
}

// 심볼 레이어: 기호 32개를 원샷 심볼 레이어 30칸과 베이스 배열의 빈칸에 배치
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct SymbolsConfig {
  // 베이스 빈칸에 둔 기호의 추가 노력값 (같은 자리 노력값에 더함). 원샷 탭 비용(노력값 표의 가장 작은 값)보다 크면
  // 심볼 레이어가 모자랄 때만 씀
  pub base_slot_extra: f32,
  // 순서 있는 짝: 같은 줄에만. 한 손 안이면 떨어져 있어도 되고 첫 글자가 왼쪽, 양손에 나뉘면 좌우 대칭(같은 손가락)에
  // 첫 글자가 왼손. 순서 있는 짝과 없는 짝은 가리지 않고 모양이 같은 짝끼리 맞춤: 한 손 가로 짝은 같은 두 열(한 칸씩 넓은
  // 짝은 같은 줄에서 감쌀 때만), 한 손 세로 짝은 같은 두 줄, 양손 대칭 짝은 같은 열이나 같은 줄(꺾이면 바로 옆끼리만)
  pub pairs: Vec<String>,
  // 순서 없는 짝: 위 자리에 더해 한 손 같은 열(떨어져 있어도 됨)도 되고, 어느 쪽이 먼저든
  pub loose_pairs: Vec<String>,
  // 담금질 한 번의 이동 횟수 (여러 번 시작하는 규칙은 [sa]의 starts_per_batch, stop_after_hits, max_starts)
  pub steps: usize,
  // 예전 자리 (이제 [code]의 corpus_file, crates). 남아 있으면 옮기라고 알려 줌
  #[serde(rename = "code_corpus_file")]
  old_corpus_file: Option<toml::Value>,
  #[serde(rename = "code_crates")]
  old_crates: Option<toml::Value>,
}

impl Default for SymbolsConfig {
  fn default() -> Self {
    Self {
      base_slot_extra: 3.0,
      pairs: ["()", "[]", "{}", "<>"].iter().map(|s| (*s).to_string()).collect(),
      loose_pairs: ["'\"", ":;", "/\\"].iter().map(|s| (*s).to_string()).collect(),
      steps: 3_000_000,
      old_corpus_file: None,
      old_crates: None,
    }
  }
}

impl SymbolsConfig {
  pub fn validate(&self) -> anyhow::Result<()> {
    let sym = |c: char| c.is_ascii_punctuation();
    if self.old_corpus_file.is_some() || self.old_crates.is_some() {
      anyhow::bail!("[symbols] code_corpus_file and code_crates moved to [code] as corpus_file and crates");
    }
    if self.base_slot_extra < 0.0 { anyhow::bail!("symbols.base_slot_extra must be non-negative"); }
    if self.steps == 0 { anyhow::bail!("symbols.steps must be > 0"); }
    let mut seen = String::new();
    for p in self.pairs.iter().chain(&self.loose_pairs) {
      let cs: Vec<char> = p.chars().collect();
      if cs.len() != 2 || cs[0] == cs[1] || !cs.iter().all(|&c| sym(c)) {
        anyhow::bail!("symbols pairs must be two different ASCII symbols, found {p:?}");
      }
      if let Some(c) = cs.iter().find(|c| seen.contains(**c)) { anyhow::bail!("symbol {c:?} is in more than one pair"); }
      seen.extend(cs);
    }
    Ok(())
  }
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

  // 배열 최적화·채점에 실제로 쓰는 center_rare 글자 수: 한손 배열에는 쓰지 않음 (가운데 열에도 아무 글자나)
  pub fn center_rare_letters(&self) -> usize { if self.one_hand { 0 } else { self.center_rare } }

  // 결과 파일 이름 (result_path 기준): 한손이면 확장자 앞에 -one-hand를 붙여 두 손 결과와 섞이지 않게
  // (result.bin → result-one-hand.bin)
  pub fn result_file_name(&self) -> String {
    let f = &self.result_file;
    if !self.one_hand { return f.clone(); }
    let name_start = f.rfind('/').map_or(0, |i| i + 1);
    match f.rfind('.').filter(|&i| i > name_start) {
      Some(i) => format!("{}-one-hand{}", &f[..i], &f[i..]),
      None => format!("{f}-one-hand"),
    }
  }

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
    if self.center_rare != 0 && !(3..=6).contains(&self.center_rare) {
      anyhow::bail!("center_rare must be 0 (off) or 3..=6 (27 keys, 24 outer slots)");
    }
    let w = &self.weights;
    if w.gap < 0.0 || w.row < 0.0 || w.center < 0.0 || w.outroll < 0.0 || w.roll_row() < 0.0 || w.half_scissors < 0.0 || w.full_scissors_good < 0.0 || w.skip_scissors < 0.0 || w.sfs < 0.0 || w.sfs_same_key() < 0.0 || w.bad_redirect < 0.0 {
      anyhow::bail!("weights.gap, row, roll_row, center, outroll, half_scissors, full_scissors_good, skip_scissors, sfs, sfs_same_key, bad_redirect must be non-negative");
    }
    let c = &self.costs;
    if c.effort < 0.0 || c.sfb < 0.0 || c.full_scissors < 0.0 || c.inroll < 0.0 || c.redirect < 0.0 || c.lateral.is_some_and(|v| v < 0.0) {
      anyhow::bail!("costs.effort, sfb, full_scissors, inroll, redirect, lateral must be non-negative");
    }
    // LSB는 scissors 항목에 시저스 단위로 들어가므로, 시저스 기준값이 0이면 lateral을 나타낼 수 없음
    if c.lateral.is_some_and(|v| v > 0.0) && c.full_scissors <= 0.0 {
      anyhow::bail!("costs.lateral needs costs.full_scissors > 0 (lateral stretches are counted in scissors units)");
    }
    if self.effort.grid().iter().flatten().any(|&v| !(v > 0.0 && v.is_finite())) {
      anyhow::bail!("every [effort] value must be a positive number");
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
    if sa.starts_per_batch == 0 || sa.stop_after_hits == 0 { anyhow::bail!("sa.starts_per_batch and sa.stop_after_hits must be > 0"); }
    if sa.max_starts < sa.starts_per_batch { anyhow::bail!("sa.max_starts must be >= sa.starts_per_batch"); }
    if !(0.0..=1.0).contains(&self.code_share) { anyhow::bail!("code_share must be in 0.0..=1.0"); }
    if !(self.one_hand_layer_effort > 0.0 && self.one_hand_layer_effort.is_finite()) {
      anyhow::bail!("one_hand_layer_effort must be a positive number");
    }
    self.code.validate()?;
    self.symbols.validate()
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
    // 기본 config.toml에서 덤프·커버리지 항목과 one_hand(브랜치에 따라 true)를 빼고 테스트할 값만 넣음
    let mut table: toml::Table = toml::from_str(&base).unwrap();
    table.retain(|k, _| !k.starts_with("wikipedia_") && !k.ends_with("_coverage") && k != "one_hand");
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
  fn effort_grid_defaults_and_mirroring() {
    // 기본값은 예전 고정값과 같음
    let g = EffortGrid::default().grid();
    assert_eq!(g, [
      [2.4, 1.3, 1.1, 1.3, 3.5, 3.5, 1.3, 1.1, 1.3, 2.4],
      [1.6, 1.1, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.1, 1.6],
      [2.2, 1.5, 1.3, 1.1, 4.0, 4.0, 1.1, 1.3, 1.5, 2.2],
    ]);
    // [effort]가 없으면 기본값, 일부 행만 있으면 그 행만 바뀜
    let base: toml::Table = toml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml")).unwrap()).unwrap();
    let with = |extra: &str| -> Result<Config, toml::de::Error> {
      let mut t = base.clone();
      t.remove("effort");
      t.extend(toml::from_str::<toml::Table>(extra).unwrap());
      toml::Value::Table(t).try_into()
    };
    assert_eq!(with("").unwrap().effort, EffortGrid::default());
    let c = with("[effort]\nhome = [2.0, 1.2, 1.0, 1.0, 2.5]").unwrap();
    assert_eq!(c.effort.grid()[1], [2.0, 1.2, 1.0, 1.0, 2.5, 2.5, 1.0, 1.0, 1.2, 2.0]);
    assert_eq!(c.effort.top, default_effort_top());
    c.validate().unwrap();
    // 0 이하는 거부, 5칸이 아니면 읽기 실패
    assert!(with("[effort]\ntop = [2.4, 1.3, 0.0, 1.3, 3.5]").unwrap().validate().is_err());
    assert!(with("[effort]\ntop = [2.4, 1.3, 1.1, 1.3]").is_err());
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

  #[test]
  fn center_rare_leaves_room_for_the_outer_keys() {
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    // 키 27개(a-z와 스페이스)가 바깥 24칸에 다 들어가도록 3부터
    assert!(config(&format!("{url}center_rare = 2")).validate().is_err());
    assert!(config(&format!("{url}center_rare = 3")).validate().is_ok());
    assert!(config(&format!("{url}center_rare = 7")).validate().is_err());
    assert!(config(&format!("{url}center_rare = 0")).validate().is_ok());
    // 예전 설정의 space_key는 남아 있어도 무시
    assert!(config(&format!("{url}space_key = \"thumb\"")).validate().is_ok());
  }

  #[test]
  fn symbols_and_code_sections_defaults_and_validation() {
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    // [symbols]·[code]가 없으면 기본값, 일부만 적으면 나머지는 기본값
    let mut t: toml::Table = toml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml")).unwrap()).unwrap();
    t.remove("symbols");
    t.remove("code");
    t.extend(toml::from_str::<toml::Table>(url).unwrap());
    let c: Config = toml::Value::Table(t).try_into().unwrap();
    c.validate().unwrap();
    assert_eq!((c.symbols.pairs.len(), c.symbols.loose_pairs[0].as_str(), c.code.crates.len(), c.code.crates[0].as_str()), (4, "'\"", 40, "tauri"));
    let c = config(&format!("{url}[code]\ncrates = [\"syn\"]\n[symbols]\nsteps = 5\n"));
    c.validate().unwrap();
    assert_eq!((c.code.crates.len(), c.code.corpus_file.as_str(), c.symbols.base_slot_extra, c.symbols.steps), (1, "code.bin", 3.0, 5));
    // config.toml의 값은 기본값과 같음
    let file = config(url);
    assert_eq!(format!("{:?}", file.symbols), format!("{:?}", SymbolsConfig::default()));
    assert_eq!(format!("{:?}", file.code), format!("{:?}", CodeConfig::default()));
    for bad in [r#"pairs = ["(("]"#, r#"pairs = ["()"]
loose_pairs = ["(\""]"#, "steps = 0", "base_slot_extra = -1.0", r#"pairs = ["(1"]"#] {
      assert!(config(&format!("{url}[symbols]\n{bad}\n")).validate().is_err(), "{bad}");
    }
    for bad in ["crates = []", r#"crates = ["a/b"]"#, "corpus_file = \"\""] {
      assert!(config(&format!("{url}[code]\n{bad}\n")).validate().is_err(), "{bad}");
    }
    // 예전 자리([symbols])에 남은 코드 코퍼스 설정은 옮기라고 알려 줌
    let err = config(&format!("{url}[symbols]\ncode_crates = [\"syn\"]\n")).validate().unwrap_err().to_string();
    assert!(err.contains("moved to [code]"), "{err}");
  }

  #[test]
  fn code_share_is_a_share_and_keeps_one_result_file() {
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    // config.toml은 코드를 25% 섞고, 결과 파일 이름에는 비율을 붙이지 않음
    let c = config(url);
    assert_eq!((c.code_share, c.result_file.as_str()), (0.25, "result.bin"));
    for bad in ["code_share = -0.1", "code_share = 1.5"] { assert!(config(&format!("{url}{bad}")).validate().is_err(), "{bad}"); }
    assert!(config(&format!("{url}code_share = 1.0")).validate().is_ok());
  }

  #[test]
  fn one_hand_keeps_its_own_result_file() {
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    // one_hand가 없으면 두 손
    let c = config(url);
    assert!(!c.one_hand);
    assert_eq!(c.result_file_name(), "result.bin");
    let c = config(&format!("{url}one_hand = true"));
    assert_eq!(c.result_file_name(), "result-one-hand.bin");
    for (file, want) in [("out/res", "out/res-one-hand"), ("a.b/res", "a.b/res-one-hand"), ("x.y.bin", "x.y-one-hand.bin")] {
      let c = config(&format!("{url}one_hand = true\nresult_file = \"{file}\""));
      assert_eq!(c.result_file_name(), want, "{file}");
    }
  }

  #[test]
  fn one_hand_ignores_center_rare() {
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    // 두 손이면 config.toml의 center_rare = 3 그대로
    assert_eq!(config(url).center_rare_letters(), 3);
    let c = config(&format!("{url}one_hand = true"));
    assert_eq!((c.center_rare, c.center_rare_letters()), (3, 0));
  }

  #[test]
  fn one_hand_layer_effort_is_a_positive_factor() {
    let url = "wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n";
    // config.toml은 3배, 없어도 3배
    assert_eq!(config(url).one_hand_layer_effort, 3.0);
    let mut table: toml::Table = toml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml")).unwrap()).unwrap();
    table.remove("one_hand_layer_effort");
    let c: Config = toml::Value::Table(table).try_into().unwrap();
    assert_eq!(c.one_hand_layer_effort, 3.0);
    for bad in ["0.0", "-1.0", "nan"] {
      assert!(config(&format!("{url}one_hand_layer_effort = {bad}")).validate().is_err(), "{bad}");
    }
    assert!(config(&format!("{url}one_hand_layer_effort = 1.0")).validate().is_ok());
  }

  #[test]
  fn lateral_needs_a_scissors_unit() {
    let mut c = config("wikipedia_dump_url = \"https://m.org/a.xml.bz2\"\n");
    c.costs.lateral = Some(9.6);
    c.validate().unwrap();
    c.costs.full_scissors = 0.0;
    assert!(c.validate().unwrap_err().to_string().contains("costs.lateral"));
    c.costs.lateral = Some(0.0);
    c.validate().unwrap();
    c.costs.lateral = Some(-1.0);
    assert!(c.validate().is_err());
    c.costs.lateral = None;
    c.validate().unwrap();
  }
}
