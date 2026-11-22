#![allow(clippy::cast_precision_loss)]
// 민감도 점검: 점수 설정값을 하나씩 ±10/20/35/50% 흔들어서
// (1) 후보 배열들의 1등이 바뀌는지 (재채점), (2) 지금 1등에서 시작한 근처 탐색이 키를 옮기는지 봄.
// 설정 파일과 결과 파일은 읽기만 함
use rayon::prelude::*;
use crate::config::Config;
use crate::corpus::CorpusArrays;
use crate::layout::{abs_weights, Layout, CHARS, COLS, PENALTY_SCORE, ROWS};
use crate::optimizer::Optimizer;
use crate::precomp::Precomp;
use crate::score::{HandRule, Score, ScoreNorm, ABS_SCALE};

pub const LEVELS: [f32; 4] = [0.10, 0.20, 0.35, 0.50];

// 흔들 설정값 하나
pub struct Knob {
  pub name: &'static str,
  pub show: fn(&Config) -> String,
  // factor배로 흔듦 (1.0 = 그대로)
  pub apply: fn(&mut Config, f32),
}

macro_rules! field {
  ($name:expr, $($f:ident).+) => {
    Knob { name: $name, show: |c| format!("{:.2}", c.$($f).+), apply: |c, x| c.$($f).+ *= x }
  };
}

// 노력값 묶음은 기준 칸 대비 추가분만 흔듦: 열 묶음은 같은 행 검지 대비, 행 묶음은 같은 열 홈 대비.
// (값 전체를 흔들면 윗줄이 홈보다 쉬워지는 식의 뜻 없는 표가 나오므로)
const INDEX_COL: usize = 3;

fn scale_col(c: &mut Config, col: usize, f: f32) {
  let e = &mut c.effort;
  for row in [&mut e.top, &mut e.home, &mut e.bottom] { row[col] = row[INDEX_COL] + (row[col] - row[INDEX_COL]) * f; }
}

fn show_col(c: &Config, col: usize) -> String {
  let e = &c.effort;
  format!("+{:.1}/{:.1}/{:.1}", e.top[col] - e.top[INDEX_COL], e.home[col] - e.home[INDEX_COL], e.bottom[col] - e.bottom[INDEX_COL])
}

fn scale_row(row: &mut [f32; 5], home: &[f32; 5], f: f32) {
  for (v, h) in row.iter_mut().zip(home) { *v = h + (*v - h) * f; }
}

fn show_row(row: &[f32; 5], home: &[f32; 5]) -> String {
  let parts: Vec<String> = row.iter().zip(home).map(|(v, h)| format!("{:.1}", v - h)).collect();
  format!("+{}", parts.join("/"))
}

pub fn knobs() -> Vec<Knob> {
  vec![
    field!("costs.effort", costs.effort),
    field!("costs.sfb", costs.sfb),
    field!("costs.full_scissors", costs.full_scissors),
    field!("costs.inroll", costs.inroll),
    field!("costs.redirect", costs.redirect),
    field!("weights.outroll", weights.outroll),
    field!("weights.bad_redirect", weights.bad_redirect),
    // 같은 키 스킵그램 배율이 비어 있으면 sfs를 따라가므로, sfs만 흔들리게 지금 값으로 고정
    Knob { name: "weights.sfs", show: |c| format!("{:.2}", c.weights.sfs),
      apply: |c, f| { c.weights.sfs_same_key = Some(c.weights.sfs_same_key()); c.weights.sfs *= f; } },
    field!("weights.half_scissors", weights.half_scissors),
    field!("weights.full_scissors_good", weights.full_scissors_good),
    field!("weights.skip_scissors", weights.skip_scissors),
    // row는 SFB·시저스의 행 배율만 흔듦 (롤의 행 배율은 roll_row에서 따로)
    Knob { name: "weights.row", show: |c| format!("{:.2}", c.weights.row),
      apply: |c, f| { c.weights.roll_row = Some(c.weights.roll_row()); c.weights.row *= f; } },
    Knob { name: "weights.roll_row", show: |c| format!("{:.2}", c.weights.roll_row()),
      apply: |c, f| c.weights.roll_row = Some(c.weights.roll_row() * f) },
    field!("weights.gap", weights.gap),
    field!("weights.center", weights.center),
    field!("hand_penalty", hand_penalty),
    Knob { name: "effort.pinky", show: |c| show_col(c, 0), apply: |c, f| scale_col(c, 0, f) },
    Knob { name: "effort.ring", show: |c| show_col(c, 1), apply: |c, f| scale_col(c, 1, f) },
    Knob { name: "effort.center_col", show: |c| show_col(c, 4), apply: |c, f| scale_col(c, 4, f) },
    Knob { name: "effort.top_row", show: |c| show_row(&c.effort.top, &c.effort.home),
      apply: |c, f| { let home = c.effort.home; scale_row(&mut c.effort.top, &home, f); } },
    Knob { name: "effort.bottom_row", show: |c| show_row(&c.effort.bottom, &c.effort.home),
      apply: |c, f| { let home = c.effort.home; scale_row(&mut c.effort.bottom, &home, f); } },
  ]
}

// init_score_norm과 같은 기준 (무작위 배열 중앙값은 표시용이라 기준 설정의 값을 그대로 씀)
pub fn scoring(arrays: &CorpusArrays, c: &Config, median: Score) -> (Precomp, ScoreNorm) {
  let precomp = Precomp::new(&arrays.freq, &c.weights, &c.costs, &c.effort, c.hand_hard_limit());
  let norm = ScoreNorm { median, abs: abs_weights(arrays, &precomp, &c.costs), hand: HandRule::from_config(c) };
  (precomp, norm)
}

// 총점 → 글자 100개당 비용
pub fn cost100(total: i64) -> f64 { -(total as f64) / ABS_SCALE * 100.0 }

fn rescore(arrays: &CorpusArrays, precomp: &Precomp, norm: &ScoreNorm, cands: &[Layout]) -> Vec<i64> {
  cands.iter().map(|l| { let mut l = *l; l.calc_scores(arrays, precomp); l.calc_total_score(norm); l.total }).collect()
}

pub struct Outcome {
  // 후보 중 1등 (동점이면 기준 1등 유지)
  pub winner: usize,
  // 기준 1등보다 다른 후보 1등이 몇 점 앞서는지 (글자 100개당 비용, 음수면 기준 1등이 앞섬). 다른 후보가 없으면 None
  pub rival_lead: Option<f64>,
  pub descended: Layout,
  pub moved: Vec<char>,
  // 근처 탐색으로 줄어든 비용 (글자 100개당)
  pub gain: f64,
}

impl Outcome {
  pub fn changed(&self, base: usize) -> bool { self.winner != base || !self.moved.is_empty() }
}

pub fn evaluate(arrays: &CorpusArrays, c: &Config, median: Score, cands: &[Layout], base: usize, center_mask: u32) -> Outcome {
  let (precomp, norm) = scoring(arrays, c, median);
  let totals = rescore(arrays, &precomp, &norm, cands);
  let mut winner = base;
  for (i, &t) in totals.iter().enumerate() { if t > totals[winner] { winner = i; } }
  let rival = totals.iter().enumerate().filter(|&(i, &t)| i != base && t > PENALTY_SCORE).map(|(_, &t)| t).max();
  let rival_lead = rival.map(|t| cost100(totals[base]) - cost100(t));
  let o = Optimizer::new(arrays.clone(), &precomp, &norm, 1, 1, c.sa.clone(), c.ga.clone(), center_mask);
  let start = cands[base];
  let descended = o.descend(start);
  let moved = (0..arrays.keys.len()).filter(|&i| descended.pos[i] != start.pos[i]).map(|i| arrays.keys[i]).collect();
  Outcome { winner, rival_lead, descended, moved, gain: cost100(totals[base]) - cost100(descended.total) }
}

pub fn compact(l: &Layout) -> String {
  (0..ROWS).map(|r| (0..COLS).map(|c| {
    let id = l.slots[r * COLS + c];
    if id < 26 { (b'a' + id) as char } else if id < CHARS as u8 { '_' } else { '.' }
  }).collect::<String>()).collect::<Vec<_>>().join("/")
}

fn mirrored(l: &Layout) -> [u8; ROWS * COLS] {
  let mut s = l.slots;
  for r in 0..ROWS { s[r * COLS..(r + 1) * COLS].reverse(); }
  s
}

// 같은 배열이나 좌우를 뒤집은 배열(점수가 같음)은 하나만 남김
pub fn dedupe(cands: Vec<(String, Layout)>) -> Vec<(String, Layout)> {
  let mut out: Vec<(String, Layout)> = Vec::new();
  for (name, l) in cands {
    if out.iter().any(|(_, o)| o.slots == l.slots || mirrored(o) == l.slots) {
      println!("[Sensitivity] skipping {name}: same layout (or its mirror) as an earlier candidate");
      continue;
    }
    out.push((name, l));
  }
  out
}

fn pct(f: f32) -> String { format!("{:+.0}%", (f - 1.0) * 100.0) }

pub fn run(arrays: &CorpusArrays, config: &Config, median: Score, cands: Vec<(String, Layout)>, center_mask: u32) {
  let mut cands = dedupe(cands);
  let layouts = |c: &[(String, Layout)]| c.iter().map(|(_, l)| *l).collect::<Vec<Layout>>();
  let (precomp, norm) = scoring(arrays, config, median);
  let totals = rescore(arrays, &precomp, &norm, &layouts(&cands));
  let mut base = 0;
  for (i, &t) in totals.iter().enumerate() { if t > totals[base] { base = i; } }

  // 기준 1등이 지금 설정에서 이미 근처 최적이 아니면, 근처 탐색 결과를 기준으로 삼음 (아니면 모든 칸이 바뀜으로 나옴)
  let check = evaluate(arrays, config, median, &layouts(&cands), base, center_mask);
  if !check.moved.is_empty() {
    println!("[Sensitivity] note: [{}] {} is not a local optimum with the current settings: moving {} keys saves {:.2}; using that layout as the base",
      base + 1, cands[base].0, check.moved.len(), check.gain);
    cands.push((format!("local optimum from [{}]", base + 1), check.descended));
    base = cands.len() - 1;
  }
  let lays = layouts(&cands);
  let totals = rescore(arrays, &precomp, &norm, &lays);

  println!("\n[Sensitivity] each setting scaled one at a time by ±{} (effort groups: only the part above the neighbouring key)",
    LEVELS.iter().map(|l| format!("{:.0}", l * 100.0)).collect::<Vec<_>>().join("/") + "%");
  println!("candidates (cost per 100 letters with the current settings, lower is better):");
  for (i, (name, l)) in cands.iter().enumerate() {
    let mark = if i == base { "  <- base" } else { "" };
    let cost = if totals[i] > PENALTY_SCORE { format!("{:.2}", cost100(totals[i])) } else { "DISQUALIFIED".to_string() };
    println!("  [{}] {}  {}  {}{mark}", i + 1, compact(l), cost, name);
  }

  let knobs = knobs();
  let factors: Vec<f32> = LEVELS.iter().rev().map(|l| 1.0 - l).chain(LEVELS.iter().map(|l| 1.0 + l)).collect();
  let cases: Vec<(usize, f32)> = (0..knobs.len()).flat_map(|k| factors.iter().map(move |&f| (k, f))).collect();
  let outcomes: Vec<Outcome> = cases.par_iter().map(|&(k, f)| {
    let mut c = config.clone();
    (knobs[k].apply)(&mut c, f);
    evaluate(arrays, &c, median, &lays, base, center_mask)
  }).collect();
  let at = |k: usize, f: usize| &outcomes[k * factors.len() + f];

  println!("\ncell: . = no change, R = another candidate becomes #1, number = keys moved by local search from the base");
  let head: String = factors.iter().map(|&f| format!("{:>5}", pct(f))).collect();
  println!("{:<28}{:<22}{head}", "setting", "value");
  for (k, knob) in knobs.iter().enumerate() {
    let cells: String = (0..factors.len()).map(|f| {
      let o = at(k, f);
      let mut s = String::new();
      if o.winner != base { s.push('R'); }
      if !o.moved.is_empty() { s.push_str(&o.moved.len().to_string()); }
      if s.is_empty() { s.push('.'); }
      format!("{s:>5}")
    }).collect();
    println!("{:<28}{:<22}{cells}", knob.name, (knob.show)(config));
  }

  // 방향마다 처음 바뀌는 폭과 그때의 내용
  println!("\nfirst change in each direction (cost per 100 letters under the changed setting):");
  let half = LEVELS.len();
  let mut first_change = vec![None::<f32>; knobs.len()];
  for (k, knob) in knobs.iter().enumerate() {
    // 아래쪽은 -10%부터 -50%로, 위쪽은 +10%부터 +50%로
    for dir in [(0..half).rev().collect::<Vec<_>>(), (half..2 * half).collect()] {
      let Some(&f) = dir.iter().find(|&&f| at(k, f).changed(base)) else {
        println!("  {:<28}{}: no change", knob.name, if dir[0] < half { "lower" } else { "higher" });
        continue;
      };
      let o = at(k, f);
      let level = (factors[f] - 1.0).abs();
      first_change[k] = Some(first_change[k].map_or(level, |m: f32| m.min(level)));
      let mut parts = Vec::new();
      if o.winner != base {
        parts.push(format!("[{}] wins by {:.2}", o.winner + 1, o.rival_lead.unwrap_or(0.0)));
      }
      if !o.moved.is_empty() {
        let keys: String = o.moved.iter().map(char::to_string).collect::<Vec<_>>().join(" ");
        parts.push(format!("local search moves {} keys ({keys}), saves {:.2} -> {}", o.moved.len(), o.gain, compact(&o.descended)));
      }
      println!("  {:<28}{}: {}", knob.name, pct(factors[f]), parts.join("; "));
    }
  }

  let names = |pred: &dyn Fn(Option<f32>) -> bool| -> String {
    let v: Vec<&str> = knobs.iter().zip(&first_change).filter(|&(_, &m)| pred(m)).map(|(k, _)| k.name).collect();
    if v.is_empty() { "-".to_string() } else { v.join(", ") }
  };
  let tight = LEVELS[1] + 1e-4;
  println!("\nsummary:");
  println!("  sensitive (changes within ±{:.0}%): {}", LEVELS[1] * 100.0, names(&|m| m.is_some_and(|m| m <= tight)));
  println!("  moderate  (changes at ±{:.0}-{:.0}%): {}", LEVELS[2] * 100.0, LEVELS[3] * 100.0, names(&|m| m.is_some_and(|m| m > tight)));
  println!("  robust    (no change up to ±{:.0}%): {}", LEVELS[3] * 100.0, names(&|m| m.is_none()));
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_util;

  fn config() -> Config {
    crate::config::load_config(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/config.toml"))).unwrap()
  }

  fn random_layouts(arrays: &CorpusArrays, n: usize) -> Vec<Layout> {
    (0..n).map(|_| Layout::make_random(&arrays.keys, &[], 0)).collect()
  }

  fn totals_for(arrays: &CorpusArrays, c: &Config, median: Score, ls: &[Layout]) -> Vec<i64> {
    let (p, n) = scoring(arrays, c, median);
    rescore(arrays, &p, &n, ls)
  }

  #[test]
  fn unscaled_settings_keep_every_score() {
    let arrays = test_util::stats().to_arrays(false);
    let c = config();
    let precomp = Precomp::new(&arrays.freq, &c.weights, &c.costs, &c.effort, c.hand_hard_limit());
    let norm = crate::layout::init_score_norm(&arrays, &precomp, &c.costs, HandRule::from_config(&c), 0);
    let ls = random_layouts(&arrays, 20);
    let want = rescore(&arrays, &precomp, &norm, &ls);
    for knob in knobs() {
      let mut k = c.clone();
      (knob.apply)(&mut k, 1.0);
      let got = totals_for(&arrays, &k, norm.median, &ls);
      for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1000, "{}: {g} vs {w}", knob.name);
      }
    }
  }

  #[test]
  fn every_setting_changes_some_score() {
    let arrays = test_util::stats().to_arrays(false);
    let mut c = config();
    c.min_hand_usage = 0.5; // 손 비율 감점이 거의 모든 배열에 붙게 해서 hand_penalty도 확인
    let ls = random_layouts(&arrays, 30);
    let want = totals_for(&arrays, &c, Score::default(), &ls);
    for knob in knobs() {
      let mut k = c.clone();
      (knob.apply)(&mut k, 1.5);
      let got = totals_for(&arrays, &k, Score::default(), &ls);
      assert!(got.iter().zip(&want).any(|(g, w)| (g - w).abs() > 1000), "{} has no effect", knob.name);
    }
  }

  #[test]
  fn effort_groups_scale_only_the_extra_part() {
    let c = config();
    let mut top = c.clone();
    (knobs().iter().find(|k| k.name == "effort.top_row").unwrap().apply)(&mut top, 2.0);
    assert_eq!(top.effort.home, c.effort.home);
    for i in 0..5 { assert!((top.effort.top[i] - (2.0 * c.effort.top[i] - c.effort.home[i])).abs() < 1e-6); }
    let mut pinky = c.clone();
    (knobs().iter().find(|k| k.name == "effort.pinky").unwrap().apply)(&mut pinky, 0.5);
    assert_eq!(pinky.effort.top[INDEX_COL], c.effort.top[INDEX_COL]);
    assert!((pinky.effort.home[0] - (c.effort.home[0] + c.effort.home[INDEX_COL]) / 2.0).abs() < 1e-6);
    assert_eq!(pinky.effort.top[1..], c.effort.top[1..]);
  }

  #[test]
  fn row_and_sfs_knobs_leave_their_siblings_alone() {
    let c = config();
    let find = |n: &str| knobs().into_iter().find(|k| k.name == n).unwrap();
    let mut r = c.clone();
    (find("weights.row").apply)(&mut r, 1.5);
    assert!((r.weights.roll_row() - c.weights.roll_row()).abs() < 1e-6 && (r.weights.row - c.weights.row * 1.5).abs() < 1e-6);
    let mut s = c.clone();
    s.weights.sfs_same_key = None;
    let before = s.weights.sfs_same_key();
    (find("weights.sfs").apply)(&mut s, 1.5);
    assert!((s.weights.sfs_same_key() - before).abs() < 1e-6);
  }

  #[test]
  fn evaluate_reports_no_change_at_a_local_optimum() {
    let arrays = test_util::stats().to_arrays(false);
    let c = config();
    let mask = arrays.rare_mask(c.center_rare);
    let (p, n) = scoring(&arrays, &c, Score::default());
    let o = Optimizer::new(arrays.clone(), &p, &n, 1, 1, c.sa.clone(), c.ga.clone(), mask);
    let opt = o.descend(Layout::make_random_centered(&arrays.keys, mask, &[], 0));
    let mut worse = Layout::make_random_centered(&arrays.keys, mask, &[], 0);
    worse.calc_scores(&arrays, &p); worse.calc_total_score(&n);
    let r = evaluate(&arrays, &c, Score::default(), &[opt, worse], 0, mask);
    assert!(r.moved.is_empty() && r.gain.abs() < 1e-9);
    assert!(r.winner == 0 || worse.total > opt.total);
    // 다른 배열에서 시작하면 키가 움직이고 비용이 줄어듦
    let r = evaluate(&arrays, &c, Score::default(), &[opt, worse], 1, mask);
    assert!(!r.moved.is_empty() && r.gain > 0.0);
  }

  #[test]
  fn dedupe_drops_mirrors() {
    let arrays = test_util::stats().to_arrays(false);
    let a = Layout::make_random(&arrays.keys, &[], 0);
    let mut m = Layout::new();
    m.slots = mirrored(&a);
    for (p, &id) in m.slots.iter().enumerate() { if id < CHARS as u8 { m.pos[id as usize] = p as u8; } }
    let b = Layout::make_random(&arrays.keys, &[], 0);
    let kept = dedupe(vec![("a".into(), a), ("m".into(), m), ("a2".into(), a), ("b".into(), b)]);
    assert_eq!(kept.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), ["a", "b"]);
  }
}
