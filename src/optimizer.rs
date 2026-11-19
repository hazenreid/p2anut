#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]
use rayon::prelude::*;
use crate::corpus::CorpusArrays;
use crate::layout::{Layout, ROWS, COLS, EMPTY, CHARS, PENALTY_SCORE, sort_layouts, sort_unique_layouts, save_results, fmt_num, home_row_keys, is_home_distinct, is_center_slot};
use crate::precomp::Precomp;
use crate::score::{Score, ScoreNorm};

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
  bg_mark: Vec<u32>,
  tg_mark: Vec<u32>,
  session: u32,
}

impl SaWorkspace {
  fn new(bg_len: usize, tg_len: usize) -> Self {
    Self {
      aff_bg_indices: Vec::with_capacity(1024),
      aff_tg_indices: Vec::with_capacity(8192),
      bg_mark: vec![0; bg_len],
      tg_mark: vec![0; tg_len],
      session: 0,
    }
  }
}

pub struct Optimizer<'a> {
  corpus_arrays: CorpusArrays,
  precomp: &'a Precomp,
  norm: &'a ScoreNorm,
  elites_len: usize,
  min_home_diff: u32,
  sa_params: crate::config::SaParams,
  ga_params: crate::config::GaParams,
  // center_rare로 고른 글자 비트마스크 (0 = 제약 없음). 제약이 있으면 가운데 두 열과 나머지 칸 사이에서는 글자를 옮기지 않음
  center_mask: u32,
}

impl<'a> Optimizer<'a> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(corpus_arrays: CorpusArrays, precomp: &'a Precomp, norm: &'a ScoreNorm, elites_len: usize, min_home_diff: u32, sa_params: crate::config::SaParams, ga_params: crate::config::GaParams, center_mask: u32) -> Self {
    Self { corpus_arrays, precomp, norm, elites_len, min_home_diff, sa_params, ga_params, center_mask }
  }

  // 이동에 쓰는 칸들이 모두 같은 구역(센터/바깥)인지. 제약이 없으면 항상 true
  fn same_zone(&self, ps: &[usize]) -> bool {
    self.center_mask == 0 || ps.iter().all(|&p| is_center_slot(p) == is_center_slot(ps[0]))
  }

  #[allow(clippy::needless_pass_by_value)]
  pub fn run(&self, base_layouts: Vec<Layout>, result_file: &str) -> (Vec<Layout>, bool) {
    let elites_len = self.elites_len;
    let mut all_elites: Vec<Layout> = vec![];
    let mut excluded_homes: Vec<u32> = vec![];
    let seeds: Vec<Option<Layout>> = (0..elites_len).map(|i| base_layouts.get(i).copied()).collect();
    let mut any_improved = false;
    let start = std::time::Instant::now();

    for (i, seed) in seeds.iter().enumerate().take(elites_len) {
      let (elite, improved) = self.find_one(i + 1, elites_len, &excluded_homes, *seed, start);
      if improved { any_improved = true; }
      excluded_homes.push(home_row_keys(&elite));
      all_elites.push(elite);
      sort_layouts(&mut all_elites);
      save_results(&all_elites, result_file).ok();
      println!("\n[Elite {}/{}] total={}", i + 1, elites_len, fmt_num(elite.total));
      elite.print();
      println!();
      if crate::SHUTDOWN.load(std::sync::atomic::Ordering::SeqCst) { break; }
    }
    (all_elites, any_improved)
  }

  fn find_one(&self, elite_idx: usize, total_elites: usize, excluded_homes: &[u32], seed: Option<Layout>, start: std::time::Instant) -> (Layout, bool) {
    let pop_size = self.ga_params.population;
    let max_stall = self.ga_params.max_stall();

    let mut population: Vec<Layout> = (0..pop_size).map(|_| self.make_scored_excl(excluded_homes)).collect();
    // 이미 찾은 엘리트와 홈행이 너무 비슷한 시드는 쓰지 않음 (그대로 반환되면 중복 엘리트가 됨)
    if let Some(mut s) = seed.filter(|s| is_home_distinct(home_row_keys(s), excluded_homes, self.min_home_diff)) {
      s.calc_scores(&self.corpus_arrays, self.precomp);
      s.calc_total_score(self.norm);
      population[0] = s;
    }
    sort_layouts(&mut population);

    let mut best = population[0];
    let mut prev_total = best.total;
    let mut generation = 1usize;
    let mut improved = false;

    let print_status = |g: usize, total: i64| {
      use std::io::Write;
      print!("\r\x1b[2K[{}/{}];{}/{} Best: {}\t{:.1}s",
        elite_idx, total_elites, g, max_stall,
        fmt_num(total), start.elapsed().as_secs_f32());
      let _ = std::io::stdout().flush();
    };
    print_status(generation, prev_total);

    while generation <= max_stall {
      let pool: Vec<&Layout> = population.iter().collect();

      let mut next: Vec<Layout> = task_seeds(pop_size - 1).into_par_iter()
        .map(|seed| {
          fastrand::seed(seed);
          let p1 = self.tournament(&pool);
          let p2 = self.tournament(&pool);
          let mut child = self.crossover(p1, p2);
          child.calc_scores(&self.corpus_arrays, self.precomp);
          child.calc_total_score(self.norm);
          child
        })
        .collect();
      next.push(best);

      population = sort_unique_layouts(next, pop_size, excluded_homes, self.min_home_diff);
      while population.len() < pop_size {
        population.push(self.make_scored_excl(excluded_homes));
      }

      let seeds = task_seeds(population.len());
      population = population.into_par_iter().zip(seeds)
        .map(|(l, seed)| { fastrand::seed(seed); self.optimize_sa(l, excluded_homes) })
        .collect();

      if crate::SHUTDOWN.load(std::sync::atomic::Ordering::SeqCst) { break; }

      // 한 손에 크게 치우친 시작 배열은 단일 이동으로 손 비율을 못 맞춰 페널티로 남을 수 있음 → 버리고 새로 채움
      population.retain(|l| {
        l.total > PENALTY_SCORE && is_home_distinct(home_row_keys(l), excluded_homes, self.min_home_diff)
      });
      while population.len() < pop_size {
        population.push(self.make_scored_excl(excluded_homes));
      }
      sort_layouts(&mut population);

      let cur_best = &population[0];
      let cur_total = cur_best.total;

      if cur_total > prev_total {
        best = *cur_best;
        prev_total = cur_total;
        generation = 1;
        improved = true;
        let inject = pop_size / 5;
        for entry in population.iter_mut().rev().take(inject) {
          *entry = self.make_scored_excl(excluded_homes);
        }
      } else {
        generation += 1;
      }
      print_status(generation, prev_total);
    }


    (best, improved)
  }

  fn tournament<'b>(&self, pool: &[&'b Layout]) -> &'b Layout {
    debug_assert!(!pool.is_empty(), "tournament pool is empty");
    let n = pool.len();
    let mut best = pool[fastrand::usize(..n)];
    for _ in 1..self.ga_params.tournament {
      let candidate = pool[fastrand::usize(..n)];
      if candidate.total > best.total { best = candidate; }
    }
    best
  }

  fn crossover(&self, p1: &Layout, p2: &Layout) -> Layout {
    let nk = CHARS;
    let mut tags1 = [0u8; ROWS * COLS];
    let mut tags2 = [0u8; ROWS * COLS];
    // 빈칸 태그는 구역별로 따로 번호를 매김: 센터 빈칸끼리, 바깥 빈칸끼리만 대응되어
    // PMX가 구역을 넘나들지 않음 (두 부모가 제약을 지키면 자식도 지킴)
    let constrained = self.center_mask != 0;
    let fill_tags = |layout: &Layout, tags: &mut [u8; ROWS * COLS]| {
      let mut blank_id = nk + 1;
      let mut center_blank = nk + 1 + ROWS * COLS;
      for (i, &id) in layout.slots.iter().enumerate() {
        tags[i] = if id < CHARS as u8 { id + 1 }
          else if constrained && is_center_slot(i) { let b = center_blank; center_blank += 1; b as u8 }
          else { let b = blank_id; blank_id += 1; b as u8 };
      }
    };
    fill_tags(p1, &mut tags1);
    fill_tags(p2, &mut tags2);

    let lo = fastrand::usize(..ROWS * COLS - 1);
    let hi = lo + 1 + fastrand::usize(..ROWS * COLS - lo - 1);
    let mut child_tags = [0u8; ROWS * COLS];
    child_tags[lo..=hi].copy_from_slice(&tags1[lo..=hi]);

    let mut reverse_map = [0u8; 256];
    let mut in_segment = [false; 256];
    for i in lo..=hi {
      reverse_map[tags1[i] as usize] = tags2[i];
      in_segment[tags1[i] as usize] = true;
    }
    for i in (0..lo).chain(hi + 1..ROWS * COLS) {
      let mut val = tags2[i];
      let mut steps = 0usize;
      while in_segment[val as usize] { val = reverse_map[val as usize]; steps += 1; if steps > hi - lo + 1 { break; } }
      child_tags[i] = val;
    }

    let mut layout = Layout::new();
    for (i, &v) in child_tags.iter().enumerate() {
      if (1..=CHARS as u8).contains(&v) {
        let id = v - 1;
        layout.slots[i] = id;
        layout.pos[id as usize] = i as u8;
      }
    }
    layout
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
    let mut ws = SaWorkspace::new(self.corpus_arrays.nz_bigrams.len(), self.corpus_arrays.nz_trigrams.len());
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

    let pe = part_effort   + new_effort;
    let ps = part_sfb      + new_bg.0 + new_tg.0;
    let pr = part_rolling  + new_bg.1 + new_tg.1;
    let pc = part_scissors + new_bg.2 + new_tg.2;
    result.total = self.norm.total_score(pe, ps, pr, pc);
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

    let pe = part_effort   + new_effort;
    let ps = part_sfb      + new_bg.0 + new_tg.0;
    let pr = part_rolling  + new_bg.1 + new_tg.1;
    let pc = part_scissors + new_bg.2 + new_tg.2;
    result.total = self.norm.total_score(pe, ps, pr, pc);
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
    let pe = part_effort + new_effort;
    let ps = part_sfb   + new_sfb + new_sfs;
    let pr = part_rolling + new_roll + new_troll;
    let pc = part_scissors + new_sciss + new_tsciss;
    result.total = self.norm.total_score(pe, ps, pr, pc);
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
    ws.session = ws.session.wrapping_add(1);
    if ws.session == 0 { ws.bg_mark.fill(0); ws.tg_mark.fill(0); ws.session = 1; }
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
      }
    }
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
    for space in [true, false] { check_delta_moves(space); }
  }

  fn check_delta_moves(space_on_grid: bool) {
    let arrays = test_util::stats().to_arrays(space_on_grid);
    let (precomp, norm) = test_util::precomp_norm_for(&arrays);
    let o = Optimizer::new(arrays, &precomp, &norm, 1, 1, test_util::sa(), test_util::ga(), 0);
    let mut ws = SaWorkspace::new(o.corpus_arrays.nz_bigrams.len(), o.corpus_arrays.nz_trigrams.len());
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
    let arrays = test_util::stats().to_arrays(false);
    let (precomp, norm) = test_util::precomp_norm_for(&arrays);
    let mask = arrays.rare_mask(4);
    assert_eq!(mask.count_ones(), 4);
    let o = Optimizer::new(arrays, &precomp, &norm, 1, 1, test_util::sa(), test_util::ga(), mask);
    let mut ws = SaWorkspace::new(o.corpus_arrays.nz_bigrams.len(), o.corpus_arrays.nz_trigrams.len());
    let mut cur = o.make_scored_excl(&[]);
    assert!(cur.center_ok(mask));
    for _ in 0..2000 {
      let cand = if fastrand::bool() { o.random_cycle3(&cur, &mut ws) } else { o.random_permute_k(&cur, 4 + fastrand::usize(..2), &mut ws) };
      assert!(cand.center_ok(mask));
      if cand.total > PENALTY_SCORE { assert_matches_full_recompute(&cand, &o.corpus_arrays, &precomp, &norm); cur = cand; }
    }
    let a = o.optimize_sa(o.make_scored_excl(&[]), &[]);
    let b = o.optimize_sa(o.make_scored_excl(&[]), &[]);
    for _ in 0..50 { let c = o.crossover(&a, &b); c.assert_invariants(); assert!(c.center_ok(mask)); }
    assert!(a.center_ok(mask) && b.center_ok(mask) && a.total > PENALTY_SCORE);
    // 센터에는 제약 글자만: 4글자 + 빈칸 2개
    let n_center = (0..ROWS * COLS).filter(|&p| is_center_slot(p) && a.slots[p] < CHARS as u8).count();
    assert_eq!(n_center, 4);
  }

  #[test]
  fn sa_improves_and_repairs_penalized_layouts() {
    let stats = test_util::stats();
    let (precomp, norm) = test_util::precomp_norm(&stats);
    let o = Optimizer::new(stats.to_arrays(true), &precomp, &norm, 1, 1, test_util::sa(), test_util::ga(), 0);
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
    let arrays = test_util::stats().to_arrays(false);
    let (precomp, norm) = test_util::precomp_norm_for(&arrays);
    let o = Optimizer::new(arrays, &precomp, &norm, 1, 1, test_util::sa(), test_util::ga(), 0);
    let a = o.optimize_sa(o.make_scored_excl(&[]), &[]);
    let b = o.optimize_sa(o.make_scored_excl(&[]), &[]);
    for l in [a, b, o.crossover(&a, &b)] {
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
    let o = Optimizer::new(stats.to_arrays(true), &precomp, &norm, 1, 3, test_util::sa(), test_util::ga(), 0);
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
    let o = Optimizer::new(stats.to_arrays(true), &precomp, &norm, 1, 2, test_util::sa(), test_util::ga(), 0);
    let path = std::env::temp_dir().join(format!("p2anut_seed_{}.bin", std::process::id()));
    let run = |seed: u64| {
      fastrand::seed(seed);
      let (elites, _) = o.run(vec![], path.to_str().unwrap());
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
    let o = Optimizer::new(stats.to_arrays(true), &precomp, &norm, 3, 2, test_util::sa(), test_util::ga(), 0);
    let path = std::env::temp_dir().join(format!("p2anut_test_{}.bin", std::process::id()));
    let (elites, _) = o.run(vec![], path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    assert_eq!(elites.len(), 3);
    for i in 0..elites.len() { for j in i + 1..elites.len() {
      assert!(crate::layout::home_diff(home_row_keys(&elites[i]), home_row_keys(&elites[j])) >= 2);
    }}
  }
}
