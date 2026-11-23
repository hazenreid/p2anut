#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use rayon::prelude::*;
use crate::corpus::CorpusArrays;
use crate::layout::{Layout, ROWS, COLS, EMPTY, CHARS, PENALTY_SCORE, sort_layouts, sort_unique_layouts, save_results, home_row_keys, is_home_distinct, is_center_slot};
use crate::precomp::Precomp;
use crate::score::{cost_per_100, Score, ScoreNorm};

const ALL_POSITIONS: [(usize, usize); ROWS * COLS] = {
  let mut arr = [(0usize, 0usize); ROWS * COLS];
  let mut i = 0;
  while i < ROWS * COLS {
    arr[i] = (i / COLS, i % COLS);
    i += 1;
  }
  arr
};

// 온도가 초기값의 이 비율 아래로 식기 전에는 Phase 1.5(전수 3-cycle, 8120회 평가)를 생략.
// 고온에서는 악화 수락 → 회복이 잦아 매번 전수 탐색하면 시간 대부분을 여기 씀 (벤치마크상 약 4배 차이)
const COLD_FRAC: f64 = 0.02;

const PAIR_COUNT: usize = ROWS * COLS * (ROWS * COLS - 1) / 2;
const PAIR_LIST: [(usize, usize); PAIR_COUNT] = {
  let mut arr = [(0usize, 0usize); PAIR_COUNT];
  let mut idx = 0;
  let mut i = 0;
  while i < ROWS * COLS {
    let mut j = i + 1;
    while j < ROWS * COLS {
      arr[idx] = (i, j);
      idx += 1;
      j += 1;
    }
    i += 1;
  }
  arr
};

// 병렬 작업마다 쓸 시드를 현재 스레드(메인)의 난수에서 미리 뽑아 둠.
// 각 작업이 시작할 때 자기 스레드의 fastrand를 이 값으로 시드하므로, rayon이 작업을
// 어느 스레드에 배치하든 결과가 메인 시드 하나로 결정됨 (작업 안에서 다시 병렬화하지 않는 전제)
fn task_seeds(n: usize) -> Vec<u64> {
  (0..n).map(|_| fastrand::u64(..)).collect()
}

struct SaWorkspace {
  aff_bg_indices: Vec<usize>,
  aff_tg_indices: Vec<usize>,
  aff_cross_indices: Vec<usize>,
  bg_mark: Vec<u32>,
  tg_mark: Vec<u32>,
  cross_mark: Vec<u32>,
  session: u32,
}

impl SaWorkspace {
  fn new(arrays: &CorpusArrays) -> Self {
    Self {
      aff_bg_indices: Vec::with_capacity(1024),
      aff_tg_indices: Vec::with_capacity(8192),
      aff_cross_indices: Vec::with_capacity(1024),
      bg_mark: vec![0; arrays.nz_bigrams.len()],
      tg_mark: vec![0; arrays.nz_trigrams.len()],
      cross_mark: vec![0; arrays.nz_cross.len()],
      session: 0,
    }
  }
}

// 한 엘리트 탐색의 결과 집계: 지금까지의 최고 배열, 그 점수가 나온 횟수, 시작 횟수
#[derive(Default)]
pub struct Tally {
  pub best: Option<Layout>,
  pub hits: usize,
  pub starts: usize,
}

impl Tally {
  // ok = 쓸 수 있는 결과인지 (실격·홈행 제약 위반이면 시작 횟수만 셈)
  fn add(&mut self, l: Layout, ok: bool) {
    self.starts += 1;
    if !ok { return; }
    match self.best {
      Some(b) if l.total < b.total => {}
      Some(b) if l.total == b.total => self.hits += 1,
      _ => { self.best = Some(l); self.hits = 1; }
    }
  }

  fn done(&self, p: &crate::config::SaParams) -> bool {
    self.starts >= p.max_starts || (self.starts >= p.starts_per_batch && self.hits >= p.stop_after_hits)
  }
}

pub struct Optimizer<'a> {
  corpus_arrays: CorpusArrays,
  precomp: &'a Precomp,
  norm: &'a ScoreNorm,
  elites_len: usize,
  min_home_diff: u32,
  sa_params: crate::config::SaParams,
  // center_rare로 고른 글자 비트마스크 (0 = 제약 없음). 제약이 있으면 가운데 두 열과 나머지 칸 사이에서는 글자를 옮기지 않음
  center_mask: u32,
}

impl<'a> Optimizer<'a> {
  pub fn new(corpus_arrays: CorpusArrays, precomp: &'a Precomp, norm: &'a ScoreNorm, elites_len: usize, min_home_diff: u32, sa_params: crate::config::SaParams, center_mask: u32) -> Self {
    Self { corpus_arrays, precomp, norm, elites_len, min_home_diff, sa_params, center_mask }
  }

  // 이동에 쓰는 칸들이 모두 같은 구역(센터/바깥)인지. 제약이 없으면 항상 true
  fn same_zone(&self, ps: &[usize]) -> bool {
    self.center_mask == 0 || ps.iter().all(|&p| is_center_slot(p) == is_center_slot(ps[0]))
  }

  // 엘리트를 하나씩 찾음. 앞서 찾은 엘리트와 홈행이 min_home_diff 이상 다른 배열만. seeds: 저장된 이전 결과 (엘리트마다 첫 시작점으로 씀)
  pub fn run(&self, seeds: &[Layout], result_file: &str) -> Vec<Layout> {
    let elites_len = self.elites_len;
    let mut all_elites: Vec<Layout> = vec![];
    let mut excluded_homes: Vec<u32> = vec![];
    let start = std::time::Instant::now();

    for i in 0..elites_len {
      let (elite, tally) = self.find_one(i + 1, elites_len, &excluded_homes, seeds.get(i).copied(), start);
      excluded_homes.push(home_row_keys(&elite));
      all_elites.push(elite);
      sort_layouts(&mut all_elites);
      // 이전 결과와 합쳐서 저장: 중간에 강제 종료돼도 이전 엘리트를 잃지 않도록
      let merged = sort_unique_layouts(seeds.iter().chain(&all_elites).copied().collect(), elites_len, &[], self.min_home_diff);
      save_results(&merged, result_file).ok();
      println!("\n[Elite {}/{}] best found {} times in {} starts", i + 1, elites_len, tally.hits, tally.starts);
      elite.print();
      elite.print_goodness(self.norm);
      println!();
      if crate::SHUTDOWN.load(std::sync::atomic::Ordering::SeqCst) { break; }
    }
    all_elites
  }

  // 무작위 배열에서 시작하는 담금질을 starts_per_batch개씩 동시에 돌리고, 같은 최고점이 stop_after_hits번 나오면 멈춤
  // (최소 한 묶음, 최대 max_starts번). seed가 있으면 첫 시작점으로 씀 (이전 결과보다 나빠지지 않게)
  fn find_one(&self, elite_idx: usize, total_elites: usize, excluded_homes: &[u32], seed: Option<Layout>, start: std::time::Instant) -> (Layout, Tally) {
    let p = &self.sa_params;
    let seed = seed.filter(|s| s.center_ok(self.center_mask) && is_home_distinct(home_row_keys(s), excluded_homes, self.min_home_diff));
    let mut tally = Tally::default();
    loop {
      let n = p.starts_per_batch.min(p.max_starts - tally.starts);
      let first = tally.starts;
      let found: Vec<Layout> = task_seeds(n).into_par_iter().enumerate()
        .map(|(k, s)| {
          fastrand::seed(s);
          let init = match seed { Some(l) if first == 0 && k == 0 => l, _ => self.make_scored_excl(excluded_homes) };
          self.optimize_sa(init, excluded_homes)
        })
        .collect();
      for l in found {
        let ok = l.total > PENALTY_SCORE && is_home_distinct(home_row_keys(&l), excluded_homes, self.min_home_diff);
        tally.add(l, ok);
      }
      let best = tally.best.map_or(f64::NAN, |b| cost_per_100(b.total));
      print!("\r\x1b[2K[{elite_idx}/{total_elites}] {} starts  best {best:.2} x{}\t{:.1}s", tally.starts, tally.hits, start.elapsed().as_secs_f32());
      { use std::io::Write; let _ = std::io::stdout().flush(); }
      if crate::SHUTDOWN.load(std::sync::atomic::Ordering::SeqCst) || tally.done(p) { break; }
    }
    // 하나도 못 찾았으면(모든 시작이 손 비율 실격 등) 조건을 지키는 무작위 배열이라도 돌려줌
    let best = tally.best.unwrap_or_else(|| self.make_scored_excl(excluded_homes));
    (best, tally)
  }

  #[allow(clippy::too_many_lines)]
  // excluded_homes: 이미 찾은 엘리트의 홈행. 이들과 min_home_diff 미만으로 가까워지는 이동은 거부
  // (그대로 두면 SA가 끝난 뒤 find_one에서 버려져 계산이 낭비됨)
  fn optimize_sa(&self, mut base: Layout, excluded_homes: &[u32]) -> Layout {
    base.calc_scores(&self.corpus_arrays, self.precomp);
    base.calc_total_score(self.norm);
    // 시작부터 가까운 배열이면 막지 않음 (막으면 어떤 이동도 못 해서 멈춤)
    let enforce = !excluded_homes.is_empty()
      && is_home_distinct(home_row_keys(&base), excluded_homes, self.min_home_diff);
    let allowed = |l: &Layout| !enforce || is_home_distinct(home_row_keys(l), excluded_homes, self.min_home_diff);
    let p = &self.sa_params;
    let mut cur = base;
    let mut best = base;
    let mut no_improve = 0usize;
    let mut ws = SaWorkspace::new(&self.corpus_arrays);
    let mut run_descent = true;
    let mut temp = -1.0f64; // 음수 = 아직 미정
    let mut t0 = 0.0f64;

    loop {
      if crate::SHUTDOWN.load(std::sync::atomic::Ordering::SeqCst) { break; }

      let hot = temp > 0.0 && temp > t0 * COLD_FRAC;
      if run_descent {
        run_descent = false;
        // Phase 1: 전수 2-swap 스캔 → 2-opt 로컬 최적 보장
        'descent: loop {
          let mut pairs = PAIR_LIST;
          for i in (1..pairs.len()).rev() { pairs.swap(i, fastrand::usize(..=i)); }
          for &(p1, p2) in &pairs {
            if !self.same_zone(&[p1, p2]) { continue; }
            let candidate = self.swap2_delta(&cur, p1, p2, &mut ws);
            if candidate.total > cur.total && allowed(&candidate) {
              cur = candidate;
              if cur.total > best.total { best = cur; no_improve = 0; }
              continue 'descent;
            }
          }
          break;
        }

        // Phase 1.5: 전수 3-cycle 1회 패스, best-improvement 1개 적용 (고온 구간에서는 생략)
        if !hot {
          let mut best_delta = 0i64;
          let mut best_cand = cur;
          for i in 0..ROWS * COLS {
            for j in i + 1..ROWS * COLS {
              for k in j + 1..ROWS * COLS {
                if !self.same_zone(&[i, j, k]) { continue; }
                for &left in &[true, false] {
                  let cand = self.swap3_cycle_delta(&cur, i, j, k, left, &mut ws);
                  if cand.total > cur.total + best_delta && allowed(&cand) {
                    best_delta = cand.total - cur.total;
                    best_cand = cand;
                  }
                }
              }
            }
          }
          if best_delta > 0 {
            cur = best_cand;
            if cur.total > best.total { best = cur; no_improve = 0; }
            run_descent = true;
            continue;
          }
        }
      }

      if no_improve >= p.max_no_improve { break; }

      // 첫 로컬 최적에 도달한 시점에 초기 온도 결정
      if temp < 0.0 { temp = self.initial_temperature(&cur, &mut ws); t0 = temp; }

      // Phase 2: 80% 랜덤 3-사이클 + 20% k=4,5칸 무작위 순열
      let candidate = if fastrand::u32(..100) < 80 {
        self.random_cycle3(&cur, &mut ws)
      } else {
        let k = if fastrand::bool() { 4 } else { 5 };
        self.random_permute_k(&cur, k, &mut ws)
      };
      // 개선은 수락 후 descent, 악화는 Metropolis 기준(exp(delta/T))으로만 수락.
      // 엘리트와 너무 가까워지는 이동은 둘 다 거부
      let delta = candidate.total - cur.total;
      if allowed(&candidate) {
        if delta > 0 {
          cur = candidate;
          run_descent = true;
        } else if delta < 0 && temp > 0.0 && candidate.total > PENALTY_SCORE
          && fastrand::f64() < (delta as f64 / temp).exp() {
          cur = candidate;
        }
      }
      // 종료 판정은 cur가 아니라 best 기준 (악화 수락 후 회복을 개선으로 세지 않도록)
      if cur.total > best.total {
        best = cur;
        no_improve = 0;
      } else {
        no_improve += 1;
      }
      temp *= p.cooling;
      #[cfg(debug_assertions)]
      best.assert_invariants();
    }
    // delta path는 calc_scores를 거치지 않으므로 반환 전 재정규화 (미러링 포함)
    best.calc_scores(&self.corpus_arrays, self.precomp);
    best.calc_total_score(self.norm);
    best
  }

  // 무작위 없는 근처 탐색 (민감도 점검용): 두 키 맞바꾸기를 전부 보고 가장 좋은 것을 적용, 더 좋아지는 게 없으면
  // 세 키 돌리기를 전부 보고 가장 좋은 것을 적용. 둘 다 없을 때까지 반복. center_rare 구역과 손 비율 규칙은 지킴
  pub fn descend(&self, mut cur: Layout) -> Layout {
    const MAX_STEPS: usize = 1000; // 안전장치 (매 단계 점수가 올라야 하므로 실제로는 훨씬 전에 멈춤)
    const N: usize = ROWS * COLS;
    let rescore = |l: &mut Layout| { l.calc_scores(&self.corpus_arrays, self.precomp); l.calc_total_score(self.norm); };
    rescore(&mut cur);
    let mut ws = SaWorkspace::new(&self.corpus_arrays);
    for _ in 0..MAX_STEPS {
      let mut best = cur;
      for &(p1, p2) in &PAIR_LIST {
        if !self.same_zone(&[p1, p2]) { continue; }
        let cand = self.swap2_delta(&cur, p1, p2, &mut ws);
        if cand.total > best.total { best = cand; }
      }
      if best.total == cur.total {
        for i in 0..N { for j in i + 1..N { for k in j + 1..N {
          if !self.same_zone(&[i, j, k]) { continue; }
          for left in [true, false] {
            let cand = self.swap3_cycle_delta(&cur, i, j, k, left, &mut ws);
            if cand.total > best.total { best = cand; }
          }
        }}}
      }
      if best.total <= cur.total { break; }
      // delta 오차가 쌓이지 않게 단계마다 전체 재계산
      cur = best;
      rescore(&mut cur);
    }
    cur
  }

  fn random_cycle3(&self, cur: &Layout, ws: &mut SaWorkspace) -> Layout {
    let p1 = fastrand::usize(..ROWS * COLS);
    let p2 = loop { let p = fastrand::usize(..ROWS * COLS); if p != p1 && self.same_zone(&[p1, p]) { break p; } };
    let p3 = loop { let p = fastrand::usize(..ROWS * COLS); if p != p1 && p != p2 && self.same_zone(&[p1, p]) { break p; } };
    self.swap3_cycle_delta(cur, p1, p2, p3, fastrand::bool(), ws)
  }

  // 평균 악화폭 d̄에 대해 exp(-d̄/T0) = accept_start 가 되도록 T0 결정 (점수 스케일에 무관)
  fn initial_temperature(&self, cur: &Layout, ws: &mut SaWorkspace) -> f64 {
    const SAMPLES: usize = 200;
    let p0 = self.sa_params.accept_start;
    if p0 <= 0.0 { return 0.0; }
    let (mut sum, mut n) = (0.0f64, 0usize);
    for _ in 0..SAMPLES {
      let cand = self.random_cycle3(cur, ws);
      if cand.total > PENALTY_SCORE && cand.total < cur.total {
        sum += (cur.total - cand.total) as f64;
        n += 1;
      }
    }
    if n == 0 { return 0.0; }
    (sum / n as f64) / -p0.ln()
  }

  #[allow(clippy::similar_names, clippy::many_single_char_names)]
  fn swap2_delta(&self, base: &Layout, p1: usize, p2: usize, ws: &mut SaWorkspace) -> Layout {
    let v1 = base.slots[p1];
    let v2 = base.slots[p2];
    if v1 == v2 { return *base; }

    self.collect_affected(&[v1, v2], ws);

    let mut temp_pos = base.pos;
    if v2 < CHARS as u8 { temp_pos[v2 as usize] = p1 as u8; }
    if v1 < CHARS as u8 { temp_pos[v1 as usize] = p2 as u8; }

    let mid = COLS / 2;
    let mut cur_l = base.left_usage;
    let mut cur_r = base.right_usage;
    if (p1 % COLS) < mid {
      if v1 < CHARS as u8 { cur_l = cur_l.saturating_sub(self.corpus_arrays.freq[v1 as usize]); }
      if v2 < CHARS as u8 { cur_l = cur_l.saturating_add(self.corpus_arrays.freq[v2 as usize]); }
    } else {
      if v1 < CHARS as u8 { cur_r = cur_r.saturating_sub(self.corpus_arrays.freq[v1 as usize]); }
      if v2 < CHARS as u8 { cur_r = cur_r.saturating_add(self.corpus_arrays.freq[v2 as usize]); }
    }
    if (p2 % COLS) < mid {
      if v2 < CHARS as u8 { cur_l = cur_l.saturating_sub(self.corpus_arrays.freq[v2 as usize]); }
      if v1 < CHARS as u8 { cur_l = cur_l.saturating_add(self.corpus_arrays.freq[v1 as usize]); }
    } else {
      if v2 < CHARS as u8 { cur_r = cur_r.saturating_sub(self.corpus_arrays.freq[v2 as usize]); }
      if v1 < CHARS as u8 { cur_r = cur_r.saturating_add(self.corpus_arrays.freq[v1 as usize]); }
    }

    let mut result = *base;
    if self.violates_hand_usage(cur_l, cur_r) {
      result.total = PENALTY_SCORE;
      return result;
    }

    let mut old_bg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_bg_indices {
      let (a, b, cnt) = self.corpus_arrays.nz_bigrams[idx];
      let e = self.precomp.bigram(base.pos[a as usize] as usize, base.pos[b as usize] as usize);
      old_bg.0 += cnt as f64 * f64::from(e.sfb_base);
      old_bg.1 += cnt as f64 * f64::from(e.rolling_base);
      old_bg.2 += cnt as f64 * f64::from(e.scissors_base);
    }
    let mut old_tg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_tg_indices {
      let (a, b, c, cnt) = self.corpus_arrays.nz_trigrams[idx];
      let e = self.precomp.trigram(base.pos[a as usize] as usize, base.pos[b as usize] as usize, base.pos[c as usize] as usize);
      old_tg.0 += cnt as f64 * f64::from(e.sfs_base);
      old_tg.1 += cnt as f64 * f64::from(e.roll_base);
      old_tg.2 += cnt as f64 * f64::from(e.skip_scissors_base);
    }

    let part_sfb      = base.score.sfb      - old_bg.0 - old_tg.0;
    let part_rolling  = base.score.rolling  - old_bg.1 - old_tg.1;
    let part_scissors = base.score.scissors - old_bg.2 - old_tg.2;
    let part_effort   = base.score.effort
      - (if v1 < CHARS as u8 { self.precomp.effort(p1, v1 as usize) } else { 0.0 })
      - (if v2 < CHARS as u8 { self.precomp.effort(p2, v2 as usize) } else { 0.0 });
    let new_effort =
      (if v2 < CHARS as u8 { self.precomp.effort(p1, v2 as usize) } else { 0.0 })
      + (if v1 < CHARS as u8 { self.precomp.effort(p2, v1 as usize) } else { 0.0 });

    let mut new_bg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_bg_indices {
      let (a, b, cnt) = self.corpus_arrays.nz_bigrams[idx];
      let e = self.precomp.bigram(temp_pos[a as usize] as usize, temp_pos[b as usize] as usize);
      new_bg.0 += cnt as f64 * f64::from(e.sfb_base);
      new_bg.1 += cnt as f64 * f64::from(e.rolling_base);
      new_bg.2 += cnt as f64 * f64::from(e.scissors_base);
    }
    let mut new_tg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_tg_indices {
      let (a, b, c, cnt) = self.corpus_arrays.nz_trigrams[idx];
      let e = self.precomp.trigram(temp_pos[a as usize] as usize, temp_pos[b as usize] as usize, temp_pos[c as usize] as usize);
      new_tg.0 += cnt as f64 * f64::from(e.sfs_base);
      new_tg.1 += cnt as f64 * f64::from(e.roll_base);
      new_tg.2 += cnt as f64 * f64::from(e.skip_scissors_base);
    }

    let (old_cs, old_ck) = self.cross_sum(&base.pos, ws);
    let (new_cs, new_ck) = self.cross_sum(&temp_pos, ws);
    let pe = part_effort   + new_effort;
    let ps = part_sfb      + new_bg.0 + new_tg.0 + new_cs - old_cs;
    let pr = part_rolling  + new_bg.1 + new_tg.1;
    let pc = part_scissors + new_bg.2 + new_tg.2 + new_ck - old_ck;
    result.total = self.norm.total_score(pe, ps, pr, pc, cur_l, cur_r);
    result.score = Score { effort: pe, sfb: ps, rolling: pr, scissors: pc };
    result.slots[p1] = v2;
    result.slots[p2] = v1;
    if v2 < CHARS as u8 { result.pos[v2 as usize] = p1 as u8; }
    if v1 < CHARS as u8 { result.pos[v1 as usize] = p2 as u8; }
    result.left_usage  = cur_l;
    result.right_usage = cur_r;
    result
  }

  #[allow(clippy::similar_names, clippy::many_single_char_names, clippy::too_many_lines)]
  fn swap3_cycle_delta(&self, base: &Layout, p1: usize, p2: usize, p3: usize, left: bool, ws: &mut SaWorkspace) -> Layout {
    let v1 = base.slots[p1];
    let v2 = base.slots[p2];
    let v3 = base.slots[p3];
    if v1 == v2 && v2 == v3 { return *base; }

    // left:  p1←v2, p2←v3, p3←v1
    // right: p1←v3, p2←v1, p3←v2
    let (n1, n2, n3) = if left { (v2, v3, v1) } else { (v3, v1, v2) };

    self.collect_affected(&[v1, v2, v3], ws);

    let mut temp_pos = base.pos;
    if n1 < CHARS as u8 { temp_pos[n1 as usize] = p1 as u8; }
    if n2 < CHARS as u8 { temp_pos[n2 as usize] = p2 as u8; }
    if n3 < CHARS as u8 { temp_pos[n3 as usize] = p3 as u8; }

    let mid = COLS / 2;
    let mut cur_l = base.left_usage;
    let mut cur_r = base.right_usage;
    for (pos, old_id, new_id) in [(p1, v1, n1), (p2, v2, n2), (p3, v3, n3)] {
      if old_id == new_id { continue; }
      if (pos % COLS) < mid {
        if old_id < CHARS as u8 { cur_l = cur_l.saturating_sub(self.corpus_arrays.freq[old_id as usize]); }
        if new_id < CHARS as u8 { cur_l = cur_l.saturating_add(self.corpus_arrays.freq[new_id as usize]); }
      } else {
        if old_id < CHARS as u8 { cur_r = cur_r.saturating_sub(self.corpus_arrays.freq[old_id as usize]); }
        if new_id < CHARS as u8 { cur_r = cur_r.saturating_add(self.corpus_arrays.freq[new_id as usize]); }
      }
    }

    let mut result = *base;
    if self.violates_hand_usage(cur_l, cur_r) {
      result.total = PENALTY_SCORE;
      return result;
    }

    let mut old_bg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_bg_indices {
      let (a, b, cnt) = self.corpus_arrays.nz_bigrams[idx];
      let e = self.precomp.bigram(base.pos[a as usize] as usize, base.pos[b as usize] as usize);
      old_bg.0 += cnt as f64 * f64::from(e.sfb_base);
      old_bg.1 += cnt as f64 * f64::from(e.rolling_base);
      old_bg.2 += cnt as f64 * f64::from(e.scissors_base);
    }
    let mut old_tg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_tg_indices {
      let (a, b, c, cnt) = self.corpus_arrays.nz_trigrams[idx];
      let e = self.precomp.trigram(base.pos[a as usize] as usize, base.pos[b as usize] as usize, base.pos[c as usize] as usize);
      old_tg.0 += cnt as f64 * f64::from(e.sfs_base);
      old_tg.1 += cnt as f64 * f64::from(e.roll_base);
      old_tg.2 += cnt as f64 * f64::from(e.skip_scissors_base);
    }

    let part_sfb      = base.score.sfb      - old_bg.0 - old_tg.0;
    let part_rolling  = base.score.rolling  - old_bg.1 - old_tg.1;
    let part_scissors = base.score.scissors - old_bg.2 - old_tg.2;
    let part_effort   = base.score.effort
      - (if v1 < CHARS as u8 { self.precomp.effort(p1, v1 as usize) } else { 0.0 })
      - (if v2 < CHARS as u8 { self.precomp.effort(p2, v2 as usize) } else { 0.0 })
      - (if v3 < CHARS as u8 { self.precomp.effort(p3, v3 as usize) } else { 0.0 });
    let new_effort =
      (if n1 < CHARS as u8 { self.precomp.effort(p1, n1 as usize) } else { 0.0 })
      + (if n2 < CHARS as u8 { self.precomp.effort(p2, n2 as usize) } else { 0.0 })
      + (if n3 < CHARS as u8 { self.precomp.effort(p3, n3 as usize) } else { 0.0 });

    let mut new_bg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_bg_indices {
      let (a, b, cnt) = self.corpus_arrays.nz_bigrams[idx];
      let e = self.precomp.bigram(temp_pos[a as usize] as usize, temp_pos[b as usize] as usize);
      new_bg.0 += cnt as f64 * f64::from(e.sfb_base);
      new_bg.1 += cnt as f64 * f64::from(e.rolling_base);
      new_bg.2 += cnt as f64 * f64::from(e.scissors_base);
    }
    let mut new_tg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_tg_indices {
      let (a, b, c, cnt) = self.corpus_arrays.nz_trigrams[idx];
      let e = self.precomp.trigram(temp_pos[a as usize] as usize, temp_pos[b as usize] as usize, temp_pos[c as usize] as usize);
      new_tg.0 += cnt as f64 * f64::from(e.sfs_base);
      new_tg.1 += cnt as f64 * f64::from(e.roll_base);
      new_tg.2 += cnt as f64 * f64::from(e.skip_scissors_base);
    }

    let (old_cs, old_ck) = self.cross_sum(&base.pos, ws);
    let (new_cs, new_ck) = self.cross_sum(&temp_pos, ws);
    let pe = part_effort   + new_effort;
    let ps = part_sfb      + new_bg.0 + new_tg.0 + new_cs - old_cs;
    let pr = part_rolling  + new_bg.1 + new_tg.1;
    let pc = part_scissors + new_bg.2 + new_tg.2 + new_ck - old_ck;
    result.total = self.norm.total_score(pe, ps, pr, pc, cur_l, cur_r);
    result.score = Score { effort: pe, sfb: ps, rolling: pr, scissors: pc };
    result.slots[p1] = n1; result.slots[p2] = n2; result.slots[p3] = n3;
    if n1 < CHARS as u8 { result.pos[n1 as usize] = p1 as u8; }
    if n2 < CHARS as u8 { result.pos[n2 as usize] = p2 as u8; }
    if n3 < CHARS as u8 { result.pos[n3 as usize] = p3 as u8; }
    result.left_usage  = cur_l;
    result.right_usage = cur_r;
    result
  }

  // 무작위로 고른 k칸의 내용을 무작위 순열로 재배치 (전수 탐색 아님 — 수락 여부는 SA가 판단)
  #[allow(clippy::many_single_char_names, clippy::too_many_lines, clippy::similar_names)]
  fn random_permute_k(&self, base: &Layout, k: usize, ws: &mut SaWorkspace) -> Layout {
    // 제약이 있으면 첫 칸과 같은 구역(센터 6칸 또는 바깥 24칸) 안에서만 고름
    let first = fastrand::usize(..ROWS * COLS);
    let mut pool = ALL_POSITIONS;
    let mut pool_len = 0;
    for &(r, c) in &ALL_POSITIONS {
      if self.same_zone(&[first, r * COLS + c]) { pool[pool_len] = (r, c); pool_len += 1; }
    }
    let mut coords = [(0usize, 0usize); 16];
    let mut remaining = pool_len;
    for coord in coords.iter_mut().take(k) {
      let j = fastrand::usize(..remaining);
      *coord = pool[j];
      pool[j] = pool[remaining - 1];
      remaining -= 1;
    }

    let mut values = [EMPTY; 16];
    let mut swap_pos = [0usize; 16];
    for (i, coord) in coords.iter().enumerate().take(k) {
      let p = coord.0 * COLS + coord.1;
      swap_pos[i] = p;
      values[i] = base.slots[p];
    }

    // 고른 칸이 전부 같은 값(빈칸)이면 바꿀 게 없음
    if values[1..k].iter().all(|&v| v == values[0]) { return *base; }
    // Fisher-Yates. 배치가 그대로인 순열(빈칸끼리 자리바꿈 포함)은 평가해도 낭비이므로 다시 뽑음
    let mut perm = values;
    loop {
      for i in (1..k).rev() { perm.swap(i, fastrand::usize(..=i)); }
      if perm[..k] != values[..k] { break; }
    }

    self.collect_affected(&values[..k], ws);

    // Build new position map from the random permutation
    let mut temp_pos = base.pos;
    for (idx, &id) in perm.iter().enumerate().take(k) {
      if id < CHARS as u8 { temp_pos[id as usize] = swap_pos[idx] as u8; }
    }

    // Compute hand usage delta
    let mid = COLS / 2;
    let mut cur_l = base.left_usage;
    let mut cur_r = base.right_usage;
    for idx in 0..k {
      let old_id = values[idx]; let new_id = perm[idx];
      if old_id == new_id { continue; }
      if (swap_pos[idx] % COLS) < mid {
        if old_id < CHARS as u8 { cur_l = cur_l.saturating_sub(self.corpus_arrays.freq[old_id as usize]); }
        if new_id < CHARS as u8 { cur_l = cur_l.saturating_add(self.corpus_arrays.freq[new_id as usize]); }
      } else {
        if old_id < CHARS as u8 { cur_r = cur_r.saturating_sub(self.corpus_arrays.freq[old_id as usize]); }
        if new_id < CHARS as u8 { cur_r = cur_r.saturating_add(self.corpus_arrays.freq[new_id as usize]); }
      }
    }

    // Old delta contributions (f64 for precision on large corpora)
    let mut old_bg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_bg_indices {
      let (a, b, cnt) = self.corpus_arrays.nz_bigrams[idx];
      let e = self.precomp.bigram(base.pos[a as usize] as usize, base.pos[b as usize] as usize);
      old_bg.0 += cnt as f64 * f64::from(e.sfb_base);
      old_bg.1 += cnt as f64 * f64::from(e.rolling_base);
      old_bg.2 += cnt as f64 * f64::from(e.scissors_base);
    }
    let mut old_tg = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_tg_indices {
      let (a, b, c, cnt) = self.corpus_arrays.nz_trigrams[idx];
      let e = self.precomp.trigram(base.pos[a as usize] as usize, base.pos[b as usize] as usize, base.pos[c as usize] as usize);
      old_tg.0 += cnt as f64 * f64::from(e.sfs_base);
      old_tg.1 += cnt as f64 * f64::from(e.roll_base);
      old_tg.2 += cnt as f64 * f64::from(e.skip_scissors_base);
    }

    let part_sfb      = base.score.sfb - old_bg.0 - old_tg.0;
    let part_rolling  = base.score.rolling - old_bg.1 - old_tg.1;
    let part_scissors = base.score.scissors - old_bg.2 - old_tg.2;
    let part_effort   = base.score.effort
      - values.iter().take(k).enumerate()
          .map(|(i, &id)| if id < CHARS as u8 { self.precomp.effort(swap_pos[i], id as usize) } else { 0.0 })
          .sum::<f64>();
    let new_effort: f64 = perm.iter().enumerate().take(k)
      .map(|(j, &id)| if id < CHARS as u8 { self.precomp.effort(swap_pos[j], id as usize) } else { 0.0 })
      .sum();

    let mut result = *base;
    if self.violates_hand_usage(cur_l, cur_r) {
      result.total = PENALTY_SCORE;
      return result;
    }
    let (mut new_sfb, mut new_roll, mut new_sciss) = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_bg_indices {
      let (a, b, cnt) = self.corpus_arrays.nz_bigrams[idx];
      let e = self.precomp.bigram(temp_pos[a as usize] as usize, temp_pos[b as usize] as usize);
      new_sfb   += cnt as f64 * f64::from(e.sfb_base);
      new_roll  += cnt as f64 * f64::from(e.rolling_base);
      new_sciss += cnt as f64 * f64::from(e.scissors_base);
    }
    let (mut new_sfs, mut new_troll, mut new_tsciss) = (0.0f64, 0.0f64, 0.0f64);
    for &idx in &ws.aff_tg_indices {
      let (a, b, c, cnt) = self.corpus_arrays.nz_trigrams[idx];
      let e = self.precomp.trigram(temp_pos[a as usize] as usize, temp_pos[b as usize] as usize, temp_pos[c as usize] as usize);
      new_sfs   += cnt as f64 * f64::from(e.sfs_base);
      new_troll += cnt as f64 * f64::from(e.roll_base);
      new_tsciss += cnt as f64 * f64::from(e.skip_scissors_base);
    }
    let (old_cs, old_ck) = self.cross_sum(&base.pos, ws);
    let (new_cs, new_ck) = self.cross_sum(&temp_pos, ws);
    let pe = part_effort + new_effort;
    let ps = part_sfb   + new_sfb + new_sfs + new_cs - old_cs;
    let pr = part_rolling + new_roll + new_troll;
    let pc = part_scissors + new_sciss + new_tsciss + new_ck - old_ck;
    result.total = self.norm.total_score(pe, ps, pr, pc, cur_l, cur_r);
    result.score = Score { effort: pe, sfb: ps, rolling: pr, scissors: pc };

    for (idx, &p) in swap_pos.iter().enumerate().take(k) {
      let id = perm[idx];
      result.slots[p] = id;
      if id < CHARS as u8 { result.pos[id as usize] = p as u8; }
    }
    result.left_usage  = cur_l;
    result.right_usage = cur_r;
    result
  }

  fn collect_affected(&self, ids: &[u8], ws: &mut SaWorkspace) {
    ws.aff_bg_indices.clear();
    ws.aff_tg_indices.clear();
    ws.aff_cross_indices.clear();
    ws.session = ws.session.wrapping_add(1);
    if ws.session == 0 { ws.bg_mark.fill(0); ws.tg_mark.fill(0); ws.cross_mark.fill(0); ws.session = 1; }
    let sess = ws.session;
    for &id in ids {
      if id < CHARS as u8 {
        let start_b = self.corpus_arrays.bigram_offsets[id as usize];
        let end_b   = self.corpus_arrays.bigram_offsets[id as usize + 1];
        for &idx in &self.corpus_arrays.bigram_flat[start_b..end_b] {
          if ws.bg_mark[idx] != sess { ws.bg_mark[idx] = sess; ws.aff_bg_indices.push(idx); }
        }
        let start_t = self.corpus_arrays.trigram_offsets[id as usize];
        let end_t   = self.corpus_arrays.trigram_offsets[id as usize + 1];
        for &idx in &self.corpus_arrays.trigram_flat[start_t..end_t] {
          if ws.tg_mark[idx] != sess { ws.tg_mark[idx] = sess; ws.aff_tg_indices.push(idx); }
        }
        let start_c = self.corpus_arrays.cross_offsets[id as usize];
        let end_c   = self.corpus_arrays.cross_offsets[id as usize + 1];
        for &idx in &self.corpus_arrays.cross_flat[start_c..end_c] {
          if ws.cross_mark[idx] != sess { ws.cross_mark[idx] = sess; ws.aff_cross_indices.push(idx); }
        }
      }
    }
  }

  // 영향받는 공백 건너 쌍들의 (스킵그램, 스킵 시저스) 합. pos: 글자 → 칸
  fn cross_sum(&self, pos: &[u8; CHARS], ws: &SaWorkspace) -> (f64, f64) {
    let mut sum = (0.0f64, 0.0f64);
    for &idx in &ws.aff_cross_indices {
      let (a, c, cnt) = self.corpus_arrays.nz_cross[idx];
      let e = self.precomp.cross(pos[a as usize] as usize, pos[c as usize] as usize);
      sum.0 += cnt as f64 * f64::from(e.sfs_base);
      sum.1 += cnt as f64 * f64::from(e.skip_scissors_base);
    }
    sum
  }

  fn violates_hand_usage(&self, cur_l: u64, cur_r: u64) -> bool {
    self.precomp.violates_hand_usage(cur_l, cur_r)
  }

  fn make_scored_excl(&self, excluded_homes: &[u32]) -> Layout {
    loop {
      let mut l = Layout::make_random_centered(&self.corpus_arrays.keys, self.center_mask, excluded_homes, self.min_home_diff);
      l.calc_scores(&self.corpus_arrays, self.precomp);
      l.calc_total_score(self.norm);
      if l.total > PENALTY_SCORE { return l; }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_util;

  fn assert_matches_full_recompute(cand: &Layout, arrays: &CorpusArrays, precomp: &Precomp, norm: &ScoreNorm) {
    cand.assert_invariants();
    let mut full = *cand;
    full.calc_scores(arrays, precomp);
    full.calc_total_score(norm);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs().max(1.0);
    assert!(close(cand.score.effort, full.score.effort), "effort {} vs {}", cand.score.effort, full.score.effort);
    assert!(close(cand.score.sfb, full.score.sfb), "sfb {} vs {}", cand.score.sfb, full.score.sfb);
    assert!(close(cand.score.rolling, full.score.rolling), "rolling {} vs {}", cand.score.rolling, full.score.rolling);
    assert!(close(cand.score.scissors, full.score.scissors), "scissors {} vs {}", cand.score.scissors, full.score.scissors);
    assert_eq!((cand.left_usage, cand.right_usage), (full.left_usage, full.right_usage));
    assert!((cand.total - full.total).abs() <= 1000, "total {} vs {}", cand.total, full.total);
  }

  #[test]
  fn delta_moves_match_full_recompute() {
    for soft in [false, true] { check_delta_moves(soft); }
  }

  // soft_hand: 손 비율 감점이 붙는 구간(40~45%)에서도 delta 계산이 전체 재계산과 같은지
  fn check_delta_moves(soft_hand: bool) {
    let arrays = test_util::stats().to_arrays();
    let (precomp, norm) = if soft_hand { test_util::precomp_norm_soft_hand(&arrays) } else { test_util::precomp_norm_for(&arrays) };
    let o = Optimizer::new(arrays, &precomp, &norm, 1, 1, test_util::sa(), 0);
    let mut ws = SaWorkspace::new(&o.corpus_arrays);
    let mut cur = o.make_scored_excl(&[]);
    let mut checked = 0;
    for step in 0..3000 {
      let cand = match step % 3 {
        0 => {
          let p1 = fastrand::usize(..ROWS * COLS);
          let p2 = (p1 + 1 + fastrand::usize(..ROWS * COLS - 1)) % (ROWS * COLS);
          o.swap2_delta(&cur, p1, p2, &mut ws)
        }
        1 => o.random_cycle3(&cur, &mut ws),
        _ => o.random_permute_k(&cur, 4 + fastrand::usize(..2), &mut ws),
      };
      if cand.total <= PENALTY_SCORE { continue; }
      assert_matches_full_recompute(&cand, &o.corpus_arrays, &precomp, &norm);
      checked += 1;
      // 누적 delta 오차까지 검증하려고 후보를 계속 이어서 사용
      cur = cand;
    }
    assert!(checked > 1000);
  }

  #[test]
  fn center_rare_constraint_holds() {
    let arrays = test_util::stats().to_arrays();
    let (precomp, norm) = test_util::precomp_norm_for(&arrays);
    let mask = arrays.rare_mask(4);
    assert_eq!(mask.count_ones(), 4);
    let o = Optimizer::new(arrays, &precomp, &norm, 1, 1, test_util::sa(), mask);
    let mut ws = SaWorkspace::new(&o.corpus_arrays);
    let mut cur = o.make_scored_excl(&[]);
    assert!(cur.center_ok(mask));
    for _ in 0..2000 {
      let cand = if fastrand::bool() { o.random_cycle3(&cur, &mut ws) } else { o.random_permute_k(&cur, 4 + fastrand::usize(..2), &mut ws) };
      assert!(cand.center_ok(mask));
      if cand.total > PENALTY_SCORE { assert_matches_full_recompute(&cand, &o.corpus_arrays, &precomp, &norm); cur = cand; }
    }
    let a = o.optimize_sa(o.make_scored_excl(&[]), &[]);
    let b = o.optimize_sa(o.make_scored_excl(&[]), &[]);
    assert!(a.center_ok(mask) && b.center_ok(mask) && a.total > PENALTY_SCORE);
    // 센터에는 제약 글자만: 4글자 + 빈칸 2개
    let n_center = (0..ROWS * COLS).filter(|&p| is_center_slot(p) && a.slots[p] < CHARS as u8).count();
    assert_eq!(n_center, 4);
  }

  #[test]
  fn descend_reaches_a_deterministic_local_optimum() {
    let arrays = test_util::stats().to_arrays();
    let (precomp, norm) = test_util::precomp_norm_soft_hand(&arrays);
    let mask = arrays.rare_mask(4);
    let o = Optimizer::new(arrays, &precomp, &norm, 1, 1, test_util::sa(), mask);
    let mut ws = SaWorkspace::new(&o.corpus_arrays);
    for _ in 0..3 {
      let start = o.make_scored_excl(&[]);
      let out = o.descend(start);
      assert!(out.total >= start.total);
      assert!(out.center_ok(mask));
      assert_matches_full_recompute(&out, &o.corpus_arrays, &precomp, &norm);
      assert!(o.descend(start).slots == out.slots, "same start must give the same result");
      // 어떤 두 키 맞바꾸기도 더 좋아지지 않음
      for &(p1, p2) in &PAIR_LIST {
        if o.same_zone(&[p1, p2]) { assert!(o.swap2_delta(&out, p1, p2, &mut ws).total <= out.total); }
      }
    }
  }

  #[test]
  fn sa_improves_and_repairs_penalized_layouts() {
    let stats = test_util::stats();
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let o = Optimizer::new(stats.to_arrays(), &precomp, &norm, 1, 1, test_util::sa(), 0);
    let start = o.make_scored_excl(&[]);
    let out = o.optimize_sa(start, &[]);
    assert!(out.total >= start.total);

    // 제약을 조금 벗어난 배열(한 손 55~62%)은 SA가 고쳐야 함.
    // 훨씬 치우친 배열은 단일 이동으로 못 고칠 수 있어 find_one에서 버림
    let pen = loop {
      let mut l = Layout::make_random(&o.corpus_arrays.keys, &[], 0);
      l.calc_scores(&o.corpus_arrays, &precomp);
      let share = l.left_usage as f64 / (l.left_usage + l.right_usage) as f64;
      if l.total <= PENALTY_SCORE && (0.38..0.62).contains(&share) { break l; }
    };
    assert!(o.optimize_sa(pen, &[]).total > PENALTY_SCORE);
  }

  #[test]
  fn space_off_grid_places_only_letters() {
    let arrays = test_util::stats().to_arrays();
    let (precomp, norm) = test_util::precomp_norm_for(&arrays);
    let o = Optimizer::new(arrays, &precomp, &norm, 1, 1, test_util::sa(), 0);
    let a = o.optimize_sa(o.make_scored_excl(&[]), &[]);
    let b = o.optimize_sa(o.make_scored_excl(&[]), &[]);
    for l in [a, b] {
      l.assert_invariants();
      assert_eq!(l.key_count(), 26);
      assert_eq!(l.pos[26], EMPTY);
    }
    assert!(a.total > PENALTY_SCORE);
  }

  #[test]
  fn sa_stays_away_from_excluded_homes() {
    let stats = test_util::stats();
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let o = Optimizer::new(stats.to_arrays(), &precomp, &norm, 1, 3, test_util::sa(), 0);
    for _ in 0..5 {
      let start = o.make_scored_excl(&[]);
      // 제외 없이 찾은 결과의 홈행을 막고 같은 시작점에서 다시 돌리면, 그 홈행과 3글자 이상 달라야 함
      let free = o.optimize_sa(start, &[]);
      let excluded = [home_row_keys(&free)];
      if !is_home_distinct(home_row_keys(&start), &excluded, 3) { continue; }
      let constrained = o.optimize_sa(start, &excluded);
      assert!(is_home_distinct(home_row_keys(&constrained), &excluded, 3));
      assert!(constrained.total >= start.total);
    }
  }

  #[test]
  fn same_seed_gives_same_elites() {
    let stats = test_util::stats();
    fastrand::seed(7); // 정규화 표본도 고정해야 실행(프로세스)마다 같은 목적함수가 됨
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let o = Optimizer::new(stats.to_arrays(), &precomp, &norm, 1, 2, test_util::sa(), 0);
    let path = std::env::temp_dir().join(format!("p2anut_seed_{}.bin", std::process::id()));
    let run = |seed: u64| {
      fastrand::seed(seed);
      let elites = o.run(&[], path.to_str().unwrap());
      elites.iter().map(|l| (l.slots, l.total)).collect::<Vec<_>>()
    };
    // 다른 시드끼리 결과가 다른지는 확인하지 않음: 작은 코퍼스에서는 같은 최적해로 수렴할 수 있음
    let a = run(42);
    let b = run(42);
    let _ = std::fs::remove_file(&path);
    assert!(a == b, "same seed must reproduce identical elites");
  }

  #[test]
  fn elites_respect_min_home_diff() {
    let stats = test_util::stats();
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let o = Optimizer::new(stats.to_arrays(), &precomp, &norm, 3, 2, test_util::sa(), 0);
    let path = std::env::temp_dir().join(format!("p2anut_test_{}.bin", std::process::id()));
    let elites = o.run(&[], path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    assert_eq!(elites.len(), 3);
    for i in 0..elites.len() { for j in i + 1..elites.len() {
      assert!(crate::layout::home_diff(home_row_keys(&elites[i]), home_row_keys(&elites[j])) >= 2);
    }}
  }

  fn scored(total: i64) -> Layout { let mut l = Layout::new(); l.total = total; l }

  #[test]
  fn tally_counts_how_often_the_best_comes_back() {
    let p = crate::config::SaParams { starts_per_batch: 4, stop_after_hits: 2, max_starts: 10, ..test_util::sa() };
    let mut t = Tally::default();
    t.add(scored(-10), true);
    t.add(scored(-5), true);
    t.add(scored(-5), false); // 실격이나 홈행 위반은 시작으로만 셈
    t.add(scored(-7), true);
    assert_eq!((t.best.unwrap().total, t.hits, t.starts), (-5, 1, 4));
    assert!(!t.done(&p));
    t.add(scored(-5), true);
    assert_eq!(t.hits, 2);
    assert!(t.done(&p));
    // 더 좋은 점수가 나오면 횟수를 처음부터
    t.add(scored(-1), true);
    assert_eq!((t.best.unwrap().total, t.hits), (-1, 1));
    // 최소 한 묶음은 채움, 상한에서는 멈춤
    let mut u = Tally::default();
    u.add(scored(-1), true); u.add(scored(-1), true);
    assert!(!u.done(&p));
    for _ in 0..8 { u.add(scored(-9), true); }
    assert!(u.done(&p));
  }

  #[test]
  fn each_elite_stops_once_its_best_repeats() {
    let arrays = test_util::stats().to_arrays();
    let (precomp, norm) = test_util::precomp_norm_for(&arrays);
    // 최고점이 한 번만 나와도 되면 첫 묶음(4번)을 채우고 멈춤 (최소 한 묶음)
    let sa = crate::config::SaParams { starts_per_batch: 4, stop_after_hits: 1, max_starts: 400, ..test_util::sa() };
    let o = Optimizer::new(arrays, &precomp, &norm, 1, 1, sa, 0);
    let (best, t) = o.find_one(1, 1, &[], None, std::time::Instant::now());
    assert_eq!(t.starts, 4);
    assert!(t.hits >= 1);
    assert_eq!(t.best.unwrap().total, best.total);
    // 이전 결과를 첫 시작점으로 쓰면 그보다 나빠지지 않음
    let (again, _) = o.find_one(1, 1, &[], Some(best), std::time::Instant::now());
    assert!(again.total >= best.total);
    // 최고점이 다시 안 나오면(목표가 너무 높음) 상한에서 멈춤
    let sa = crate::config::SaParams { starts_per_batch: 4, stop_after_hits: 1000, max_starts: 12, ..test_util::sa() };
    let o = Optimizer::new(test_util::stats().to_arrays(), &precomp, &norm, 1, 1, sa, 0);
    let (_, t) = o.find_one(1, 1, &[], None, std::time::Instant::now());
    assert_eq!(t.starts, 12);
  }
}
