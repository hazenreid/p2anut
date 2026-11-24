#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use crate::score::{Score, ScoreNorm};
use crate::corpus::CorpusArrays;
use crate::precomp::Precomp;

pub const EMPTY: u8 = 255;
pub const CHARS: usize = 27;  // a-z + space(26)
pub const ROWS: usize = 3;
pub const COLS: usize = 10;
// 손 사용 비율 위반 표시. 정상 총점(-(글자당 비용) × ABS_SCALE)은 이 값보다 항상 크게 잘림 (score::total_score)
pub const PENALTY_SCORE: i64 = i64::MIN / 2;


// 가운데 두 열(4·5열, 검지 안쪽)
pub fn is_center_slot(p: usize) -> bool { matches!(p % COLS, 4 | 5) }

pub fn home_row_keys(layout: &Layout) -> u32 {
  let mut mask = 0u32;
  let home = COLS;
  // 양손 비중심 홈행 키 (중앙 두 열 제외)
  for c in (0..COLS / 2 - 1).chain(COLS / 2 + 1..COLS) {
    let id = layout.slots[home + c];
    if id < CHARS as u8 { mask |= 1 << id; }
  }
  mask
}

// 두 홈행 마스크 사이에 서로 다른 글자 수 (빈칸으로 크기가 달라도 큰 쪽 기준)
pub fn home_diff(a: u32, b: u32) -> u32 {
  (a & !b).count_ones().max((b & !a).count_ones())
}

// h가 others 모두와 홈행 글자가 min_diff개 이상 다른지
pub fn is_home_distinct(h: u32, others: &[u32], min_diff: u32) -> bool {
  others.iter().all(|&o| home_diff(h, o) >= min_diff)
}


#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Layout {
  // mapping: slot_index -> letter_idx (0..27, 255 if empty)
  pub slots: [u8; ROWS * COLS],
  // mapping: letter_idx (0..27) -> slot_index (0..30)
  pub pos: [u8; CHARS],
  pub score: Score,
  pub total: i64,
  pub left_usage: u64,
  pub right_usage: u64,
}

impl Layout {
  pub fn new() -> Self {
    Self {
      slots: [EMPTY; ROWS * COLS],
      pos: [EMPTY; CHARS],
      score: Score::default(),
      total: 0,
      left_usage: 0,
      right_usage: 0,
    }
  }

  pub fn make_random(keys: &[char], excluded_homes: &[u32], min_diff: u32) -> Self {
    const SLOTS: usize = ROWS * COLS;
    let nk = keys.len();
    let mut items = [EMPTY; SLOTS];
    for (i, item) in items.iter_mut().take(nk).enumerate() { *item = i as u8; }
    loop {
      for i in (1..SLOTS).rev() { let j = fastrand::usize(..=i); items.swap(i, j); }
      let mut layout = Layout::new();
      layout.slots = items;
      for (p, &id) in items.iter().enumerate() {
        if id < CHARS as u8 { layout.pos[id as usize] = p as u8; }
      }
      let h = home_row_keys(&layout);
      if is_home_distinct(h, excluded_homes, min_diff) {
        return layout;
      }
    }
  }

  // center_mask 글자가 모두 가운데 두 열에 있고, 그 밖의 글자는 가운데에 없는지 (0 = 제약 없음)
  pub fn center_ok(&self, center_mask: u32) -> bool {
    center_mask == 0 || self.slots.iter().enumerate().all(|(p, &id)| {
      id >= CHARS as u8 || is_center_slot(p) == (center_mask >> id & 1 == 1)
    })
  }

  // make_random과 같지만 center_mask 글자는 가운데 칸에, 나머지는 바깥 칸에만 무작위 배치
  pub fn make_random_centered(keys: &[char], center_mask: u32, excluded_homes: &[u32], min_diff: u32) -> Self {
    if center_mask == 0 { return Self::make_random(keys, excluded_homes, min_diff); }
    let (inner, outer): (Vec<usize>, Vec<usize>) = (0..ROWS * COLS).partition(|&p| is_center_slot(p));
    loop {
      let mut layout = Layout::new();
      let mut ins: Vec<u8> = vec![EMPTY; inner.len()];
      let mut outs: Vec<u8> = vec![EMPTY; outer.len()];
      let (mut ni, mut no) = (0, 0);
      for i in 0..keys.len() as u8 {
        if center_mask >> i & 1 == 1 { ins[ni] = i; ni += 1; } else { outs[no] = i; no += 1; }
      }
      fastrand::shuffle(&mut ins);
      fastrand::shuffle(&mut outs);
      for (&p, &id) in inner.iter().zip(&ins).chain(outer.iter().zip(&outs)) {
        layout.slots[p] = id;
        if id < CHARS as u8 { layout.pos[id as usize] = p as u8; }
      }
      if is_home_distinct(home_row_keys(&layout), excluded_homes, min_diff) { return layout; }
    }
  }

  // 배열 텍스트 읽기: 행마다 10칸, 글자는 a-z, 스페이스 키는 '_', 빈칸은 '.'. 공백은 무시 (print() 출력도 그대로 읽힘).
  // 빈 줄은 건너뛰고, 정확히 3행이어야 하며 각 키는 한 번씩만
  pub fn parse(text: &str) -> anyhow::Result<Self> {
    let rows: Vec<Vec<char>> = text.lines()
      .map(|l| l.chars().filter(|c| !c.is_whitespace()).collect::<Vec<char>>())
      .filter(|r| !r.is_empty())
      .collect();
    anyhow::ensure!(rows.len() == ROWS, "layout must have {ROWS} rows, found {}", rows.len());
    let mut l = Layout::new();
    for (r, row) in rows.iter().enumerate() {
      anyhow::ensure!(row.len() == COLS, "row {} must have {COLS} keys, found {}: {}", r + 1, row.len(), row.iter().collect::<String>());
      for (c, &ch) in row.iter().enumerate() {
        let ch = ch.to_ascii_lowercase();
        if ch == '.' { continue; }
        anyhow::ensure!(ch.is_ascii_lowercase() || ch == '_', "unexpected character {ch:?} in row {}", r + 1);
        let id = if ch == '_' { crate::SPACE_ID } else { ch as u8 - b'a' };
        anyhow::ensure!(l.pos[id as usize] == EMPTY, "key {ch} appears more than once");
        let p = r * COLS + c;
        l.slots[p] = id;
        l.pos[id as usize] = p as u8;
      }
    }
    Ok(l)
  }

  // 사람이 읽기 쉬운 비율 지표 (corpus 기준, 0~100%). 분모는 각각 전체 글자·바이그램·트라이그램 빈도.
  // windows: 공백(id 26)까지 포함한 모든 트라이그램 창 (CorpusStats::windows). 커뮤니티 분석기식 창 기준 통계에 씀
  pub fn print_stats(&self, corpus: &CorpusArrays, precomp: &Precomp, windows: &[(u8, u8, u8, u64)]) {
    let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 * 100.0 / b as f64 };
    let mut rows = [0u64; ROWS];
    let mut letters = 0u64;
    for (p, &id) in self.slots.iter().enumerate() {
      if id < CHARS as u8 { rows[p / COLS] += corpus.freq[id as usize]; letters += corpus.freq[id as usize]; }
    }
    let (mut bg, mut sfb, mut roll, mut sciss, mut lsb) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for &(a, b, n) in &corpus.nz_bigrams {
      let (p1, p2) = (self.pos[a as usize], self.pos[b as usize]);
      bg += n;
      if p1 == EMPTY || p2 == EMPTY { continue; }
      let (p1, p2) = (p1 as usize, p2 as usize);
      let e = precomp.bigram(p1, p2);
      if e.sfb_base > 0.0 { sfb += n; }
      if e.rolling_base > 0.0 { roll += n; }
      if e.scissors_base > 0.0 { sciss += n; }
      // 가운데 열로 뻗기(LSB): 가운데 열 키 + 같은 손 다른 손가락 (scissors에 포함)
      let same_hand = (p1 % COLS < COLS / 2) == (p2 % COLS < COLS / 2);
      if same_hand && (is_center_slot(p1) || is_center_slot(p2)) && crate::precomp::finger(p1) != crate::precomp::finger(p2) { lsb += n; }
    }
    let (mut tg, mut sfs, mut redirect, mut roll3) = (0u64, 0u64, 0u64, 0u64);
    for &(a, b, c, n) in &corpus.nz_trigrams {
      let (p1, p2, p3) = (self.pos[a as usize], self.pos[b as usize], self.pos[c as usize]);
      tg += n;
      if p1 == EMPTY || p2 == EMPTY || p3 == EMPTY { continue; }
      let e = precomp.trigram(p1 as usize, p2 as usize, p3 as usize);
      if e.sfs_base > 0.0 && p1 != p3 { sfs += n; } // 같은 키 스킵그램(ere)은 배열로 피할 수 없어 빼고 셈
      if e.roll_base < 0.0 { redirect += n; }
      if e.roll_base > 0.0 { roll3 += n; } // 세 글자 롤
    }
    println!("rows: top={:.1}%  home={:.1}%  bottom={:.1}%", pct(rows[0], letters), pct(rows[1], letters), pct(rows[2], letters));
    println!("bigrams: sfb={:.2}%  roll={:.1}%  scissors={:.2}% (lateral stretch {:.2}%)   trigrams: skipgram(other key)={:.2}%  one-hand-roll={:.1}%  redirect={:.2}%",
      pct(sfb, bg), pct(roll, bg), pct(sciss, bg), pct(lsb, bg), pct(sfs, tg), pct(roll3, tg), pct(redirect, tg));
    self.print_window_stats(precomp, windows);
  }

  // 창 기준 통계: 세 키(공백 포함)를 하나씩 밀며 나눈 분류의 비율. 스킵그램은 다른 분류와 겹쳐서 따로 셈
  fn print_window_stats(&self, precomp: &Precomp, windows: &[(u8, u8, u8, u64)]) {
    use crate::precomp::{finger, Window};
    let at = |id: u8| -> Option<usize> {
      let p = self.pos[id as usize];
      (p != EMPTY).then_some(p as usize)
    };
    let kinds = [Window::Alternate, Window::Roll2In, Window::Roll2Out, Window::Roll3In, Window::Roll3Out,
      Window::Redirect, Window::BadRedirect, Window::SameHandOther];
    let mut counts = [0u64; 8];
    let (mut total, mut skip, mut skip_space) = (0u64, 0u64, 0u64);
    for &(a, b, c, n) in windows {
      let (Some(p1), Some(p2), Some(p3)) = (at(a), at(b), at(c)) else { continue };
      total += n;
      let w = precomp.window(p1, p2, p3);
      counts[kinds.iter().position(|&k| k == w).unwrap_or(0)] += n;
      // 1·3번이 같은 손가락의 다른 키 (가운데는 다른 손가락)
      if p1 != p3 && finger(p1) == finger(p3) && finger(p2) != finger(p1) {
        skip += n;
        if b == crate::SPACE_ID { skip_space += n; }
      }
    }
    let pct = |a: u64| if total == 0 { 0.0 } else { a as f64 * 100.0 / total as f64 };
    println!("windows: alternate={:.1}%  roll2 in={:.1}% out={:.1}%  roll3 in={:.1}% out={:.1}%  redirect={:.2}%  bad-redirect={:.2}%  same-hand-other={:.1}%  skipgram={:.2}% (across space {:.2}%)",
      pct(counts[0]), pct(counts[1]), pct(counts[2]), pct(counts[3]), pct(counts[4]), pct(counts[5]), pct(counts[6]), pct(counts[7]),
      pct(skip), pct(skip_space));
  }

  // 좌우를 뒤집은 배열 (점수는 같음. space_right가 씀). 손 비율도 서로 바뀜
  pub fn mirrored(&self) -> Layout {
    let mut m = *self;
    for r in 0..ROWS { m.slots[r * COLS..(r + 1) * COLS].reverse(); }
    for (p, &id) in m.slots.iter().enumerate() { if id < CHARS as u8 { m.pos[id as usize] = p as u8; } }
    std::mem::swap(&mut m.left_usage, &mut m.right_usage);
    m
  }

  // 스페이스 키가 오른손 쪽에 오게: 왼손에 있으면 좌우를 뒤집음 (점수는 좌우 대칭이라 그대로). 스페이스가 없으면 그대로
  pub fn space_right(&self) -> Layout {
    let p = self.pos[crate::SPACE_ID as usize];
    if p != EMPTY && (p as usize % COLS) < COLS / 2 { self.mirrored() } else { *self }
  }

  pub fn key_count(&self) -> usize {
    self.slots.iter().filter(|&&id| id < CHARS as u8).count()
  }

  pub fn calc_scores(&mut self, corpus: &CorpusArrays, precomp: &Precomp) {
    let mid = COLS / 2;
    let mut left_usage = 0u64;
    let mut right_usage = 0u64;

    for p in 0..(ROWS * COLS) {
      let id = self.slots[p];
      if id < CHARS as u8 {
        let freq = corpus.freq[id as usize];
        if (p % COLS) < mid { left_usage += freq; } else { right_usage += freq; }
      }
    }

    // 페널티 배열도 score/usage는 실제 값으로 채워야 delta 이동(swap2/3)이 올바르게 동작함
    let penalized = precomp.violates_hand_usage(left_usage, right_usage);

    self.left_usage = left_usage;
    self.right_usage = right_usage;

    let mut effort = 0.0f64;
    for p in 0..(ROWS * COLS) {
      let id = self.slots[p];
      if id < CHARS as u8 {
        effort += precomp.effort(p, id as usize);
      }
    }

    let mut sfb = 0.0f64;
    let mut rolling = 0.0f64;
    let mut scissors = 0.0f64;

    for &(a, b, count) in &corpus.nz_bigrams {
      let p1 = self.pos[a as usize];
      let p2 = self.pos[b as usize];
      if p1 == EMPTY || p2 == EMPTY { continue; }
      let e = precomp.bigram(p1 as usize, p2 as usize);
      sfb      += count as f64 * f64::from(e.sfb_base);
      rolling  += count as f64 * f64::from(e.rolling_base);
      scissors += count as f64 * f64::from(e.scissors_base);
    }

    for &(a, b, c, count) in &corpus.nz_trigrams {
      let p1 = self.pos[a as usize];
      let p2 = self.pos[b as usize];
      let p3 = self.pos[c as usize];
      if p1 == EMPTY || p2 == EMPTY || p3 == EMPTY { continue; }
      let e = precomp.trigram(p1 as usize, p2 as usize, p3 as usize);
      sfb      += count as f64 * f64::from(e.sfs_base);
      rolling  += count as f64 * f64::from(e.roll_base);
      scissors += count as f64 * f64::from(e.skip_scissors_base);
    }

    self.score = Score {
      effort,
      sfb,
      rolling,
      scissors,
    };
    // 이전 PENALTY 값이 남아 calc_total_score가 건너뛰지 않도록 항상 초기화
    self.total = if penalized { PENALTY_SCORE } else { 0 };
  }

  pub fn print(&self) {
    for r in 0..ROWS {
      let mut s = String::new();
      for c in 0..COLS {
        let char_idx = self.slots[r * COLS + c];
        let ch = key_char(char_idx);
        if c == 5 { s.push_str("  "); }
        s.push(' ');
        s.push(ch);
      }
      println!("{s}");
    }
  }

  // 스페이스 키 몫의 비용 (글자 100개당, 총점에 이미 들어 있음): 스페이스 누름의 노력과
  // 스페이스가 낀 빅/트라이그램의 sfb·시저스·롤. 배열에 스페이스 키가 없으면 0
  pub fn space_cost(&self, corpus: &CorpusArrays, precomp: &Precomp, norm: &ScoreNorm) -> f64 {
    let sp = crate::SPACE_ID;
    let ps = self.pos[sp as usize];
    if ps == EMPTY { return 0.0; }
    let mut sc = Score { effort: precomp.effort(ps as usize, sp as usize), ..Score::default() };
    for &(a, b, n) in &corpus.nz_bigrams {
      if a != sp && b != sp { continue; }
      let (p1, p2) = (self.pos[a as usize], self.pos[b as usize]);
      if p1 == EMPTY || p2 == EMPTY { continue; }
      let e = precomp.bigram(p1 as usize, p2 as usize);
      sc.sfb += n as f64 * f64::from(e.sfb_base);
      sc.rolling += n as f64 * f64::from(e.rolling_base);
      sc.scissors += n as f64 * f64::from(e.scissors_base);
    }
    for &(a, b, c, n) in &corpus.nz_trigrams {
      if a != sp && b != sp && c != sp { continue; }
      let (p1, p2, p3) = (self.pos[a as usize], self.pos[b as usize], self.pos[c as usize]);
      if p1 == EMPTY || p2 == EMPTY || p3 == EMPTY { continue; }
      let e = precomp.trigram(p1 as usize, p2 as usize, p3 as usize);
      sc.sfb += n as f64 * f64::from(e.sfs_base);
      sc.rolling += n as f64 * f64::from(e.roll_base);
      sc.scissors += n as f64 * f64::from(e.skip_scissors_base);
    }
    let c = norm.cost_breakdown(&sc);
    c.effort + c.sfb + c.scissors + c.rolling
  }

  // 글자 100개당 항목별 비용 (손 비율 감점이 있으면 함께, 스페이스 키의 몫도 따로)
  pub fn print_goodness(&self, norm: &ScoreNorm, corpus: &CorpusArrays, precomp: &Precomp) {
    if self.total <= PENALTY_SCORE {
      println!("DISQUALIFIED: one hand types less than hand_hard_limit (costs below are for reference only)");
    }
    let c = norm.cost_breakdown(&self.score);
    let hand = norm.hand_cost(self.left_usage, self.right_usage);
    let total = c.effort + c.sfb + c.rolling + c.scissors + hand;
    let hand_part = if hand > 0.0 { format!("  hand-balance={hand:.1}") } else { String::new() };
    let space = self.space_cost(corpus, precomp, norm);
    println!("total={total:.1}  effort={:.1}  sfb={:.1}  scissors={:.1}  rolling={:.1}{hand_part}  (space key {space:.1} of the total)", c.effort, c.sfb, c.scissors, c.rolling);
  }

  pub fn calc_total_score(&mut self, norm: &ScoreNorm) {
    if self.total <= PENALTY_SCORE {
      return; // Hand usage penalty already applied
    }
    self.total = norm.total_score(
      self.score.effort,
      self.score.sfb,
      self.score.rolling,
      self.score.scissors,
      self.left_usage,
      self.right_usage,
    );
  }

  #[cfg(any(test, debug_assertions))]
  pub fn assert_invariants(&self) {
    for (p, &id) in self.slots.iter().enumerate() {
      if id < CHARS as u8 {
        assert_eq!(self.pos[id as usize], p as u8, "Pos mismatch for char {id}");
      }
    }
    for (id, &p) in self.pos.iter().enumerate() {
      if p != EMPTY {
        assert_eq!(self.slots[p as usize], id as u8, "Slot mismatch for pos {p}");
      }
    }
  }
}

// 배열 출력용 글자: a-z, 스페이스 '_', 빈칸 '.'
pub fn key_char(id: u8) -> char {
  if id < 26 { (b'a' + id) as char } else if id == crate::SPACE_ID { '_' } else { '.' }
}

// 쓰는 도중에 강제 종료돼도 기존 결과가 깨지지 않게 임시 파일에 쓰고 바꿔치기
pub fn save_results(layouts: &[Layout], path: &str) -> anyhow::Result<()> {
  let bytes = bincode::serialize(layouts)?;
  let tmp = format!("{path}.tmp");
  std::fs::write(&tmp, bytes)?;
  std::fs::rename(&tmp, path)?;
  Ok(())
}

// 저장된 결과. 스페이스가 왼손에 있는 배열(예전 결과)은 좌우를 뒤집어서 (space_right)
pub fn load_results(path: &str) -> anyhow::Result<Vec<Layout>> {
  let bytes = std::fs::read(path)?;
  let layouts: Vec<Layout> = bincode::deserialize(&bytes)?;
  Ok(layouts.iter().map(Layout::space_right).collect())
}

// 총점 기준: [costs]의 대표 사건 비용을 환산한 배율과 손 비율 감점
pub fn init_score_norm(arrays: &CorpusArrays, precomp: &Precomp, costs: &crate::config::Costs, hand: crate::score::HandRule) -> ScoreNorm {
  ScoreNorm { abs: abs_weights(arrays, precomp, costs), hand }
}

// 설정대로 채점할 때의 표와 총점 기준
pub fn scoring(arrays: &CorpusArrays, c: &crate::config::Config) -> (Precomp, ScoreNorm) {
  let precomp = Precomp::new(&arrays.freq, &c.weights, &c.costs, &c.effort, c.hand_hard_limit());
  let norm = init_score_norm(arrays, &precomp, &c.costs, crate::score::HandRule::from_config(c));
  (precomp, norm)
}

// 대표 사건 (왼손): 검지 홈(1,3)·검지 윗줄(0,3)·중지 홈(1,2)·중지 아랫줄(2,2). 롤·리다이렉트의 대표 사건은 Precomp에 있음
const REF_INDEX_HOME: usize = COLS + 3;
const REF_INDEX_TOP: usize = 3;
#[cfg(test)]
const REF_MIDDLE_HOME: usize = COLS + 2;
const REF_MIDDLE_BOTTOM: usize = 2 * COLS + 2;

// [costs]의 대표 사건 비용으로 정한 (sfb, scissors) 원래 점수 1단위당 비용
pub fn event_units(precomp: &Precomp, c: &crate::config::Costs) -> (f64, f64) {
  let per = |cost: f32, base: f32| if base > 0.0 { f64::from(cost) / f64::from(base) } else { 0.0 };
  (per(c.sfb, precomp.bigram(REF_INDEX_HOME, REF_INDEX_TOP).sfb_base),
   per(c.full_scissors, precomp.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).scissors_base))
}

// [costs]의 대표 사건 비용을 원래 점수 1단위당 비용으로 환산
pub fn abs_weights(arrays: &CorpusArrays, precomp: &Precomp, c: &crate::config::Costs) -> crate::score::AbsWeights {
  // 비용은 글자(a-z) 100개당. 스페이스를 배열에 두면 스페이스 누름 비용도 이 분모로 나눔
  let letters: u64 = arrays.freq.iter().take(26).sum();
  let (sfb, scissors) = event_units(precomp, c);
  crate::score::AbsWeights {
    effort: f64::from(c.effort),
    sfb,
    scissors,
    // 롤·리다이렉트는 Precomp가 이미 비용 단위로 계산함 (대표 안쪽 롤 = inroll)
    rolling: 1.0,
    per_letter: if letters > 0 { 1.0 / letters as f64 } else { 0.0 },
  }
}

fn layout_key(l: &Layout) -> impl Ord {
  let ci = |v: f64| v.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
  (
    l.total,
    ci(l.score.rolling * 1e6),
    ci(-l.score.sfb * 1e6),
    ci(-l.score.scissors * 1e6),
    ci(-l.score.effort * 1e6),
  )
}

pub fn sort_layouts(layouts: &mut [Layout]) {
  layouts.sort_by(|a, b| layout_key(b).cmp(&layout_key(a)));
}

// 정렬 후 홈행(양손 비중심 키)이 서로, 그리고 excluded_homes와 min_diff글자 이상 다른 레이아웃만 남김
pub fn sort_unique_layouts(mut layouts: Vec<Layout>, size: usize, excluded_homes: &[u32], min_diff: u32) -> Vec<Layout> {
  sort_layouts(&mut layouts);
  if layouts.is_empty() || size == 0 { return vec![]; }

  // P-22: In-place filtering to reduce allocations
  let mut seen_masks: Vec<u32> = Vec::with_capacity(size + excluded_homes.len());
  seen_masks.extend_from_slice(excluded_homes);

  layouts.retain(|l| {
    if seen_masks.len() >= size + excluded_homes.len() { return false; }
    let h = home_row_keys(l);
    // min_diff=0이어도 완전히 같은 홈행은 중복으로 처리
    if !is_home_distinct(h, &seen_masks, min_diff.max(1)) { return false; }
    seen_masks.push(h);
    true
  });

  layouts.truncate(size);
  layouts
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_util;

  fn identity_layout() -> Layout {
    let mut l = Layout::new();
    for p in 0..CHARS { l.slots[p] = p as u8; l.pos[p] = p as u8; }
    l
  }

  #[test]
  fn home_row_keys_covers_both_hands_without_center() {
    let l = identity_layout();
    // 홈행 = slot 10..20, 중앙 열(14, 15) 제외
    let expect: u32 = [10, 11, 12, 13, 16, 17, 18, 19].iter().map(|&b| 1u32 << b).sum();
    assert_eq!(home_row_keys(&l), expect);
    let mut r = l; r.slots.swap(17, 7);
    assert_ne!(home_row_keys(&r), expect);
  }

  #[test]
  fn home_diff_counts_differing_letters() {
    assert_eq!(home_diff(0b1111, 0b1111), 0);
    assert_eq!(home_diff(0b1111, 0b0111), 1);
    assert_eq!(home_diff(0b0011, 0b1100), 2);
    assert!(is_home_distinct(0b0011, &[0b1100, 0b0111_0000], 2));
    assert!(!is_home_distinct(0b0011, &[0b0110], 2));
  }

  #[test]
  fn sort_unique_respects_min_diff() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let layouts: Vec<Layout> = (0..50).map(|_| {
      let mut l = Layout::make_random(&arrays.keys, &[], 0);
      l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm); l
    }).collect();
    let kept = sort_unique_layouts(layouts, 10, &[], 3);
    for i in 0..kept.len() { for j in i + 1..kept.len() {
      assert!(home_diff(home_row_keys(&kept[i]), home_row_keys(&kept[j])) >= 3);
    }}
    assert!(kept.windows(2).all(|w| w[0].total >= w[1].total));
  }

  #[test]
  fn space_right_mirrors_only_layouts_with_the_space_on_the_left() {
    let stats = test_util::stats();
    let g = stats.to_arrays();
    let (precomp, norm) = test_util::precomp_norm_for(&g);
    let mut seen = [false; 2];
    for _ in 0..40 {
      let mut l = Layout::make_random(&g.keys, &[], 0);
      l.calc_scores(&g, &precomp);
      l.calc_total_score(&norm);
      let left = (l.pos[crate::SPACE_ID as usize] as usize % COLS) < COLS / 2;
      seen[usize::from(left)] = true;
      let r = l.space_right();
      r.assert_invariants();
      assert!(r.pos[crate::SPACE_ID as usize] as usize % COLS >= COLS / 2);
      assert_eq!(r.slots, if left { l.mirrored().slots } else { l.slots });
      // 점수는 좌우 대칭이라 다시 채점해도 같고, 손 비율은 뒤집은 쪽으로 맞음
      let mut again = r;
      again.calc_scores(&g, &precomp);
      again.calc_total_score(&norm);
      assert_eq!((again.total, again.left_usage, again.right_usage), (r.total, r.left_usage, r.right_usage));
    }
    assert!(seen[0] && seen[1]);
  }

  #[test]
  fn parse_reads_files_and_print_output() {
    let l = Layout::parse(".blyx_foup\nmanijzreht\ncgkdq.s.vw\n").unwrap();
    assert_eq!(l.key_count(), 27);
    l.assert_invariants();
    assert_eq!(l.slots[1], b'b' - b'a');
    assert_eq!(l.slots[COLS], b'm' - b'a');
    // '_'는 스페이스 키, '.'는 빈칸
    assert_eq!((l.slots[5], l.pos[crate::SPACE_ID as usize]), (crate::SPACE_ID, 5));
    assert_eq!((key_char(l.slots[5]), key_char(l.slots[0])), ('_', '.'));
    assert_eq!(l.slots[0], EMPTY);
    // print() 형식(공백, 가운데 간격)도 읽힘
    let printed = Layout::parse(" . b l y x   _ f o u p\n m a n i j   z r e h t\n\n c g k d q   . s . v w").unwrap();
    assert_eq!(printed.slots, l.slots);
    assert!(Layout::parse("abc").is_err());
    assert!(Layout::parse(".blyx_foup\nmanijzreht\ncgkdq.s.va\n").err().unwrap().to_string().contains("more than once"));
    assert!(Layout::parse("_blyx_foup\nmanijzreht\ncgkdq.s.vw\n").err().unwrap().to_string().contains("more than once"));
    assert!(Layout::parse(".blyx_foup\nmanijzreht\ncgkdq.s.v\n").is_err());
  }

  #[test]
  fn space_cost_is_the_part_of_the_total_that_the_space_key_adds() {
    let stats = test_util::stats();
    let g = stats.to_arrays();
    let (precomp, norm) = test_util::precomp_norm_for(&g);
    // 비용 분모는 글자(a-z) 수 (스페이스 누름은 분모에 넣지 않음)
    let letters: u64 = g.freq.iter().take(26).sum();
    assert!((abs_weights(&g, &precomp, &crate::config::Costs::default()).per_letter * letters as f64 - 1.0).abs() < 1e-12);
    let cost = |l: &Layout| { let c = norm.cost_breakdown(&l.score); c.effort + c.sfb + c.scissors + c.rolling };
    for _ in 0..20 {
      let mut l = Layout::make_random(&g.keys, &[], 0);
      l.calc_scores(&g, &precomp);
      let space = l.space_cost(&g, &precomp, &norm);
      // 스페이스를 빼면 그 몫만큼 줄어듦
      let mut without = l;
      without.slots[l.pos[crate::SPACE_ID as usize] as usize] = EMPTY;
      without.pos[crate::SPACE_ID as usize] = EMPTY;
      without.calc_scores(&g, &precomp);
      assert!((cost(&l) - cost(&without) - space).abs() < 1e-9 * cost(&l).abs().max(1.0), "{} {} {space}", cost(&l), cost(&without));
      assert_eq!(without.space_cost(&g, &precomp, &norm), 0.0);
      // 손 비율에 스페이스 누름도 들어감
      assert_eq!(l.left_usage + l.right_usage, g.freq.iter().sum::<u64>());
    }
  }

  #[test]
  fn absolute_costs_match_reference_events() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let (precomp, _) = test_util::precomp_norm_for(&arrays);
    let c = crate::config::Costs::default();
    let a = abs_weights(&arrays, &precomp, &c);
    // 대표 사건 하나의 원래 점수 × 환산 배율 = 설정한 비용
    let sfb = f64::from(precomp.bigram(REF_INDEX_HOME, REF_INDEX_TOP).sfb_base) * a.sfb;
    let sci = f64::from(precomp.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).scissors_base) * a.scissors;
    let roll = f64::from(precomp.bigram(REF_MIDDLE_HOME, REF_INDEX_HOME).rolling_base) * a.rolling;
    assert!((sfb - 2.5).abs() < 1e-6 && (sci - 8.0).abs() < 1e-6 && (roll - 2.0).abs() < 1e-6, "{sfb} {sci} {roll}");
    // 대표 사건이 실제로 그 종류인지
    assert!(precomp.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).rolling_base == 0.0);
    assert!(precomp.bigram(REF_MIDDLE_HOME, REF_INDEX_HOME).scissors_base == 0.0);
    assert!((a.per_letter * arrays.freq.iter().take(26).sum::<u64>() as f64 - 1.0).abs() < 1e-12);
  }

  #[test]
  fn changing_effort_keeps_reference_event_costs() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let costs = crate::config::Costs::default();
    let mut effort = crate::config::EffortGrid::default();
    let base = Precomp::new(&arrays.freq, &test_util::weights(), &costs, &effort, 0.45);
    // 검지 윗줄과 중지 홈을 더 힘들게
    effort.top[3] = 2.0;
    effort.home[2] = 1.4;
    let pc = Precomp::new(&arrays.freq, &test_util::weights(), &costs, &effort, 0.45);
    let a = abs_weights(&arrays, &pc, &costs);
    // 대표 사건 비용은 그대로
    let sfb = f64::from(pc.bigram(REF_INDEX_HOME, REF_INDEX_TOP).sfb_base) * a.sfb;
    let sci = f64::from(pc.bigram(REF_MIDDLE_BOTTOM, REF_INDEX_TOP).scissors_base) * a.scissors;
    let roll = f64::from(pc.bigram(REF_MIDDLE_HOME, REF_INDEX_HOME).rolling_base) * a.rolling;
    assert!((sfb - f64::from(costs.sfb)).abs() < 1e-5 && (sci - f64::from(costs.full_scissors)).abs() < 1e-5 && (roll - f64::from(costs.inroll)).abs() < 1e-5, "{sfb} {sci} {roll}");
    // 노력값 비용은 바뀜 (왼손·오른손 모두)
    let e = b'e' - b'a';
    assert!((pc.effort(3, e as usize) / base.effort(3, e as usize) - 2.0 / 1.3).abs() < 1e-6);
    assert!((pc.effort(COLS + 7, e as usize) / base.effort(COLS + 7, e as usize) - 1.4).abs() < 1e-6);
    // 대표 사건이 아닌 사건은 노력값 합에 따라 상대적으로 바뀜 (검지 윗줄 → 중지 홈 하프 시저스)
    assert!(pc.bigram(REF_INDEX_TOP, REF_MIDDLE_HOME).scissors_base > base.bigram(REF_INDEX_TOP, REF_MIDDLE_HOME).scissors_base);
  }

  #[test]
  fn absolute_mode_ranks_by_lower_cost() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let (precomp, _) = test_util::precomp_norm_for(&arrays);
    let norm = init_score_norm(&arrays, &precomp, &crate::config::Costs::default(), crate::score::HandRule::default());
    let mut ls: Vec<Layout> = (0..30).map(|_| {
      let mut l = Layout::make_random(&arrays.keys, &[], 0);
      l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm); l
    }).filter(|l| l.total > PENALTY_SCORE).collect();
    sort_layouts(&mut ls);
    let cost = |l: &Layout| { let c = norm.cost_breakdown(&l.score); c.effort + c.sfb + c.rolling + c.scissors };
    assert!(ls.len() > 5);
    assert!(ls.windows(2).all(|w| cost(&w[0]) <= cost(&w[1]) + 1e-6));
    // 총점 = -(글자당 비용) × ABS_SCALE
    let l = &ls[0];
    assert!(((-l.total as f64) / crate::score::ABS_SCALE * 100.0 - cost(l)).abs() < 1e-4);
  }

  #[test]
  fn soft_hand_limit_penalizes_instead_of_disqualifying() {
    let arrays = test_util::stats().to_arrays();
    let (precomp, norm) = test_util::precomp_norm_soft_hand(&arrays);
    let (mut soft_seen, mut hard_seen) = (false, false);
    for _ in 0..3000 {
      let mut l = Layout::make_random(&arrays.keys, &[], 0);
      l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm);
      let share = l.left_usage.min(l.right_usage) as f64 / (l.left_usage + l.right_usage) as f64;
      let hand = norm.hand_cost(l.left_usage, l.right_usage);
      if share < 0.40 {
        assert!(l.total <= PENALTY_SCORE, "share {share}");
        hard_seen = true;
      } else {
        assert!(l.total > PENALTY_SCORE, "share {share}");
        // 45% 아래는 감점이 붙고, 총점에 그대로 들어감
        assert_eq!(hand > 0.0, share < 0.45, "share {share}");
        let s = &l.score;
        assert_eq!(l.total, norm.total_score(s.effort, s.sfb, s.rolling, s.scissors, l.left_usage, l.right_usage));
        if share < 0.45 {
          soft_seen = true;
          let without = norm.total_score(s.effort, s.sfb, s.rolling, s.scissors, 1, 1);
          assert!(l.total < without);
        }
      }
    }
    assert!(soft_seen && hard_seen, "soft {soft_seen} hard {hard_seen}");
  }

  #[test]
  fn save_results_roundtrip_leaves_no_temp_file() {
    let dir = std::env::temp_dir().join(format!("p2anut-results-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("result.bin");
    let path = path.to_str().unwrap();
    let ls = vec![Layout::make_random(&test_util::stats().to_arrays().keys, &[], 0); 2];
    save_results(&ls, path).unwrap();
    save_results(&ls[..1], path).unwrap();
    assert_eq!(load_results(path).unwrap().len(), 1);
    assert!(!std::path::Path::new(&format!("{path}.tmp")).exists());
    // 스페이스가 왼손에 있던 결과는 읽을 때 좌우를 뒤집음
    let left = Layout::parse("_blyx.foup\nmanijzreht\ncgkdq.s.vw\n").unwrap();
    save_results(&[left], path).unwrap();
    assert_eq!(load_results(path).unwrap()[0].slots, left.mirrored().slots);
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn penalized_layout_keeps_real_scores_and_recovers() {
    let stats = test_util::stats();
    let arrays = stats.to_arrays();
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let (mut checked, mut sides) = (0, [false; 2]);
    for _ in 0..1000 {
      let mut l = Layout::make_random(&arrays.keys, &[], 0);
      l.calc_scores(&arrays, &precomp); l.calc_total_score(&norm);
      let share = l.left_usage as f32 / (l.left_usage + l.right_usage) as f32;
      // 양손 대칭 제약: 어느 쪽이든 45% 미만이면 페널티
      assert_eq!(l.total <= PENALTY_SCORE, !(0.45..=0.55).contains(&share), "left share {share}");
      if l.total > PENALTY_SCORE {
        // 이전 PENALTY 값이 남아 있어도 다시 계산하면 풀려야 함
        let mut m = l;
        m.total = PENALTY_SCORE;
        m.calc_scores(&arrays, &precomp); m.calc_total_score(&norm);
        assert_eq!(m.total, l.total);
        continue;
      }
      checked += 1;
      sides[usize::from(share > 0.5)] = true;
      assert!(l.left_usage + l.right_usage > 0 && l.score.effort > 0.0);
    }
    assert!(sides[0] && sides[1], "both left-heavy and right-heavy layouts should be penalized");
    assert!(checked > 0);
  }
}
