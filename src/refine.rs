//! Greedy refinement of the control set targeting genomic control λ → 1.
//!
//! Operates on the LD-pruned site set with **incremental χ²** updates per
//! swap. The refinement objective is `(log λ)²`, which penalizes both
//! inflation (`λ > 1`) and deflation (`λ < 1`) symmetrically — the prior
//! approach minimized λ outright, which over-corrects in practice.
//!
//! Per iteration we sample a small batch of swap proposals, evaluate each by
//! provisionally updating χ² for the affected sites, then apply only the
//! single best improving swap. Proposal sampling is biased by transport-mass
//! relevance: drops ∝ `(1 - r_norm)`, adds ∝ `r_norm`, with a small uniform
//! floor against starvation. When sex-split, swaps are constrained
//! within-sex (round-robin across strata) so the female/male ratio is
//! preserved exactly.

use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand_chacha::ChaCha8Rng;

use crate::candidates::{Candidate, Stratum};
use crate::error::{Error, Result};
use crate::io::db_pack::{DOSAGE_MISSING, DbPack};
use crate::io::query::Query;
use crate::stats::{chi_square, lambda, log_lambda_sq};

#[derive(Debug, Clone, Copy)]
pub struct RefineParams {
    pub n_controls: usize,
    pub max_iter: usize,
    pub batch: usize,
    /// Stop when |log λ| < tol.
    pub tol: f64,
    /// Stop when no improving swap is found in this many consecutive batches.
    pub plateau: usize,
    /// Floor added to weights to prevent starvation; in [0, 1] scale.
    pub uniform_floor: f64,
    pub seed: u64,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            n_controls: 500,
            max_iter: 10_000,
            batch: 50,
            tol: 0.01,
            plateau: 200,
            uniform_floor: 0.05,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefineResult {
    /// Final selected db sample indices (sorted ascending).
    pub selected: Vec<usize>,
    pub initial_lambda: f64,
    pub final_lambda: f64,
    pub iterations: usize,
    pub accepted_swaps: usize,
}

// --- site alignment -------------------------------------------------------

/// Inner-join between query.counts and the db's pruned site subset, keyed by
/// (chrom, pos, ref, alt). Non-pruned sites are skipped.
struct SiteAlignment {
    /// For each active site k: (site_idx, query_count_idx).
    pairs: Vec<(usize, usize)>,
    /// Per full site_idx: Some(active_k) if pruned AND joined to query, else None.
    /// Length == n_sites (full, not pruned-only).
    sites_to_active: Vec<Option<usize>>,
}

fn build_site_alignment(query: &Query, pack: &DbPack) -> SiteAlignment {
    let mut by_key: HashMap<(String, u64, String, String), usize> =
        HashMap::with_capacity(query.counts.len());
    // glad-prep convention: effect_allele = alt, other_allele = ref.
    for (i, c) in query.counts.iter().enumerate() {
        by_key.insert(
            (
                c.chrom.clone(),
                c.pos,
                c.other_allele.clone(),
                c.effect_allele.clone(),
            ),
            i,
        );
    }

    let n_sites = pack.sites.chrom.len();
    let mut pairs = Vec::new();
    let mut sites_to_active = vec![None; n_sites];
    for (s, slot) in sites_to_active.iter_mut().enumerate() {
        if !pack.sites.in_pruned[s] {
            continue;
        }
        let key = (
            pack.sites.chrom[s].clone(),
            pack.sites.pos[s],
            pack.sites.ref_allele[s].clone(),
            pack.sites.alt_allele[s].clone(),
        );
        if let Some(&q_idx) = by_key.get(&key) {
            let active_k = pairs.len();
            pairs.push((s, q_idx));
            *slot = Some(active_k);
        }
    }
    SiteAlignment {
        pairs,
        sites_to_active,
    }
}

// --- per-stratum candidate state -----------------------------------------

struct StratumState {
    /// Currently selected: (db_idx, normalized_score in [0,1]).
    selected: Vec<(usize, f32)>,
    /// Currently unselected pool members.
    unselected: Vec<(usize, f32)>,
}

impl StratumState {
    fn new(mut pool: Vec<(usize, f32)>) -> Self {
        normalize_in_place(&mut pool);
        Self {
            selected: Vec::new(),
            unselected: pool,
        }
    }
}

fn normalize_in_place(items: &mut [(usize, f32)]) {
    if items.is_empty() {
        return;
    }
    let (mut min_v, mut max_v) = (f32::MAX, f32::MIN);
    for (_, s) in items.iter() {
        if *s < min_v {
            min_v = *s;
        }
        if *s > max_v {
            max_v = *s;
        }
    }
    let range = (max_v - min_v).max(1e-12);
    for (_, s) in items.iter_mut() {
        *s = (*s - min_v) / range;
    }
}

// --- swap evaluation ------------------------------------------------------

/// A site change produced by a candidate swap, evaluated but not yet applied.
#[derive(Clone)]
struct SiteChange {
    active_idx: usize,
    new_ac: u32,
    new_an: u32,
    new_chi2: f64,
}

/// Compute the (active site → new state) deltas for swapping in `add_idx` and
/// out `drop_idx` against the current `ac_ctrl` / `an_ctrl`. Does NOT mutate.
#[allow(clippy::too_many_arguments)]
fn compute_swap_changes(
    drop_idx: usize,
    add_idx: usize,
    pack: &DbPack,
    alignment: &SiteAlignment,
    ac_query: &[u32],
    an_query: &[u32],
    ac_ctrl: &[u32],
    an_ctrl: &[u32],
) -> Vec<SiteChange> {
    // (active_idx) → (delta_ac, delta_an), summing contributions of drop and add.
    let mut deltas: HashMap<usize, (i32, i32)> = HashMap::new();
    for (site_idx, dosage) in pack.geno.non_ref_sites(drop_idx) {
        if let Some(active_idx) = alignment.sites_to_active[site_idx as usize] {
            let entry = deltas.entry(active_idx).or_insert((0, 0));
            if dosage == DOSAGE_MISSING {
                entry.1 += 2; // recover the 2 alleles
            } else {
                entry.0 -= dosage as i32;
            }
        }
    }
    for (site_idx, dosage) in pack.geno.non_ref_sites(add_idx) {
        if let Some(active_idx) = alignment.sites_to_active[site_idx as usize] {
            let entry = deltas.entry(active_idx).or_insert((0, 0));
            if dosage == DOSAGE_MISSING {
                entry.1 -= 2;
            } else {
                entry.0 += dosage as i32;
            }
        }
    }

    let mut changes = Vec::with_capacity(deltas.len());
    for (active_idx, (d_ac, d_an)) in deltas {
        let new_ac = (ac_ctrl[active_idx] as i32 + d_ac).max(0) as u32;
        let new_an = (an_ctrl[active_idx] as i32 + d_an).max(0) as u32;
        let new_chi2 = chi_square(
            ac_query[active_idx],
            an_query[active_idx],
            new_ac,
            new_an,
        );
        changes.push(SiteChange {
            active_idx,
            new_ac,
            new_an,
            new_chi2,
        });
    }
    changes
}

/// Temporarily apply changes to `chi2`, compute λ, then revert.
fn lambda_after(chi2: &mut [f64], changes: &[SiteChange]) -> f64 {
    let saved: Vec<(usize, f64)> = changes.iter().map(|c| (c.active_idx, chi2[c.active_idx])).collect();
    for c in changes {
        chi2[c.active_idx] = c.new_chi2;
    }
    let l = lambda(chi2);
    for (idx, old) in saved {
        chi2[idx] = old;
    }
    l
}

fn apply_changes(
    changes: &[SiteChange],
    ac_ctrl: &mut [u32],
    an_ctrl: &mut [u32],
    chi2: &mut [f64],
) {
    for c in changes {
        ac_ctrl[c.active_idx] = c.new_ac;
        an_ctrl[c.active_idx] = c.new_an;
        chi2[c.active_idx] = c.new_chi2;
    }
}

// --- weighted sampling ----------------------------------------------------

fn sample_weighted(
    items: &[(usize, f32)],
    rng: &mut ChaCha8Rng,
    floor: f64,
    invert: bool,
) -> Option<usize> {
    if items.is_empty() {
        return None;
    }
    let weights: Vec<f64> = items
        .iter()
        .map(|(_, s)| {
            let raw = if invert { 1.0 - *s as f64 } else { *s as f64 };
            raw.max(0.0) + floor
        })
        .collect();
    match WeightedIndex::new(&weights) {
        Ok(d) => Some(d.sample(rng)),
        Err(_) => Some(rng.random_range(0..items.len())),
    }
}

// --- top-level driver -----------------------------------------------------

pub fn run(
    pack: &DbPack,
    query: &Query,
    candidates: &[Candidate],
    params: RefineParams,
) -> Result<RefineResult> {
    if params.n_controls == 0 {
        return Err(Error::Schema("n_controls must be > 0".into()));
    }
    if candidates.len() < params.n_controls {
        return Err(Error::Schema(format!(
            "candidate pool size {} < requested controls {}",
            candidates.len(),
            params.n_controls
        )));
    }

    let alignment = build_site_alignment(query, pack);
    if alignment.pairs.is_empty() {
        return Err(Error::Schema(
            "no overlap between query counts and pruned db sites".into(),
        ));
    }
    let n_active = alignment.pairs.len();

    let mut rng = ChaCha8Rng::seed_from_u64(params.seed);

    // Query allele counts at active sites.
    let mut ac_query = Vec::with_capacity(n_active);
    let mut an_query = Vec::with_capacity(n_active);
    for &(_, q_idx) in &alignment.pairs {
        let c = &query.counts[q_idx];
        ac_query.push(c.alt_count);
        an_query.push(c.n_alleles);
    }

    // Bucket candidates by stratum.
    let mut bucket_all: Vec<(usize, f32)> = Vec::new();
    let mut bucket_f: Vec<(usize, f32)> = Vec::new();
    let mut bucket_m: Vec<(usize, f32)> = Vec::new();
    for c in candidates {
        match c.stratum {
            Stratum::All => bucket_all.push((c.db_idx, c.score)),
            Stratum::Female => bucket_f.push((c.db_idx, c.score)),
            Stratum::Male => bucket_m.push((c.db_idx, c.score)),
        }
    }

    let mut strata: Vec<StratumState> = Vec::new();
    if !bucket_all.is_empty() {
        strata.push(StratumState::new(bucket_all));
    }
    if !bucket_f.is_empty() {
        strata.push(StratumState::new(bucket_f));
    }
    if !bucket_m.is_empty() {
        strata.push(StratumState::new(bucket_m));
    }

    // Allocate n_controls across strata proportional to their pool sizes.
    let total_pool: usize = strata.iter().map(|s| s.unselected.len()).sum();
    let mut allocs: Vec<usize> = strata
        .iter()
        .map(|s| (params.n_controls * s.unselected.len()) / total_pool.max(1))
        .collect();
    let mut total_alloc: usize = allocs.iter().sum();
    // Hand any remainder to the largest stratum.
    while total_alloc < params.n_controls {
        let largest = (0..strata.len())
            .max_by_key(|&i| strata[i].unselected.len())
            .unwrap();
        allocs[largest] += 1;
        total_alloc += 1;
    }

    // Initial selection: weighted draw within each stratum.
    for (s_idx, n_pick) in allocs.iter().enumerate() {
        for _ in 0..*n_pick {
            if let Some(pos) = sample_weighted(
                &strata[s_idx].unselected,
                &mut rng,
                params.uniform_floor,
                false,
            ) {
                let entry = strata[s_idx].unselected.swap_remove(pos);
                strata[s_idx].selected.push(entry);
            }
        }
    }

    // Build initial AC/AN/χ².
    let n_selected: u32 = strata.iter().map(|s| s.selected.len() as u32).sum();
    let mut ac_ctrl = vec![0u32; n_active];
    let mut an_ctrl = vec![2 * n_selected; n_active];
    for state in &strata {
        for &(db_idx, _) in &state.selected {
            for (site_idx, dosage) in pack.geno.non_ref_sites(db_idx) {
                if let Some(active_idx) = alignment.sites_to_active[site_idx as usize] {
                    if dosage == DOSAGE_MISSING {
                        an_ctrl[active_idx] = an_ctrl[active_idx].saturating_sub(2);
                    } else {
                        ac_ctrl[active_idx] += dosage as u32;
                    }
                }
            }
        }
    }
    let mut chi2: Vec<f64> = (0..n_active)
        .map(|i| chi_square(ac_query[i], an_query[i], ac_ctrl[i], an_ctrl[i]))
        .collect();
    let initial_lambda = lambda(&chi2);
    let mut current_obj = log_lambda_sq(initial_lambda);

    // Greedy loop with batched best-improvement.
    let mut accepted = 0usize;
    let mut plateau_counter = 0usize;
    let mut iter = 0usize;
    let mut rr = 0usize;

    while iter < params.max_iter {
        iter += 1;
        let l = lambda(&chi2);
        if l > 0.0 && l.ln().abs() < params.tol {
            break;
        }

        let mut best: Option<(usize, usize, usize, f64, Vec<SiteChange>)> = None;

        for _ in 0..params.batch {
            // Pick a stratum (round-robin across non-empty strata).
            let s_idx = if strata.len() == 1 {
                0
            } else {
                rr = (rr + 1) % strata.len();
                rr
            };
            let state = &strata[s_idx];
            if state.selected.is_empty() || state.unselected.is_empty() {
                continue;
            }

            let drop_pos = match sample_weighted(
                &state.selected,
                &mut rng,
                params.uniform_floor,
                true,
            ) {
                Some(p) => p,
                None => continue,
            };
            let add_pos = match sample_weighted(
                &state.unselected,
                &mut rng,
                params.uniform_floor,
                false,
            ) {
                Some(p) => p,
                None => continue,
            };

            let drop_idx = state.selected[drop_pos].0;
            let add_idx = state.unselected[add_pos].0;

            let changes = compute_swap_changes(
                drop_idx, add_idx, pack, &alignment, &ac_query, &an_query, &ac_ctrl, &an_ctrl,
            );
            let new_lambda = lambda_after(&mut chi2, &changes);
            let new_obj = log_lambda_sq(new_lambda);

            if new_obj < current_obj - 1e-12
                && best.as_ref().is_none_or(|b| new_obj < b.3)
            {
                best = Some((s_idx, drop_pos, add_pos, new_obj, changes));
            }
        }

        if let Some((s_idx, drop_pos, add_pos, new_obj, changes)) = best {
            apply_changes(&changes, &mut ac_ctrl, &mut an_ctrl, &mut chi2);
            let drop_entry = strata[s_idx].selected.swap_remove(drop_pos);
            let add_entry = strata[s_idx].unselected.swap_remove(add_pos);
            strata[s_idx].selected.push(add_entry);
            strata[s_idx].unselected.push(drop_entry);
            current_obj = new_obj;
            accepted += 1;
            plateau_counter = 0;
        } else {
            plateau_counter += 1;
            if plateau_counter >= params.plateau {
                break;
            }
        }
    }

    let final_lambda = lambda(&chi2);
    let mut selected_vec: Vec<usize> = strata
        .iter()
        .flat_map(|s| s.selected.iter().map(|(idx, _)| *idx))
        .collect();
    selected_vec.sort_unstable();
    debug_assert_eq!(
        selected_vec.iter().collect::<HashSet<_>>().len(),
        selected_vec.len(),
        "selected set must contain unique db indices"
    );

    Ok(RefineResult {
        selected: selected_vec,
        initial_lambda,
        final_lambda,
        iterations: iter,
        accepted_swaps: accepted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::{Candidate, Stratum};

    #[test]
    fn normalize_maps_to_zero_one() {
        let mut items = vec![(0, 1.0_f32), (1, 3.0), (2, 5.0)];
        normalize_in_place(&mut items);
        approx::assert_abs_diff_eq!(items[0].1, 0.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(items[1].1, 0.5, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(items[2].1, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn normalize_handles_constant() {
        let mut items = vec![(0, 0.5_f32), (1, 0.5), (2, 0.5)];
        normalize_in_place(&mut items);
        // Range is clamped to 1e-12; all become large values, but no panic.
        for (_, s) in &items {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn refine_rejects_too_few_candidates() {
        // Build a minimal valid query + pack, but pass too few candidates.
        use crate::io::db_pack;
        use crate::io::query::{Distributions, FittedGmm, Mode, Query, SnpCount};

        let tmp = tempfile::tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), 4, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();

        let query = Query {
            version: "1.0".into(),
            reference_build: "GRCh38".into(),
            n_samples: 100,
            n_snps_attempted: 4,
            n_snps_found: 4,
            per_sex_counts: None,
            counts: (0..4)
                .map(|i| SnpCount {
                    chrom: "1".into(),
                    pos: 100 + i as u64 * 10,
                    effect_allele: "G".into(),
                    other_allele: "A".into(),
                    alt_count: 50,
                    n_alleles: 200,
                })
                .collect(),
            distributions: Distributions {
                mode: Mode::None,
                n_dims: 2,
                all: Some(FittedGmm {
                    n_components: 1,
                    weights: vec![1.0],
                    means: vec![vec![0.0, 0.0]],
                    covariances: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
                }),
                female: None,
                male: None,
            },
        };

        let candidates: Vec<Candidate> = (0..2)
            .map(|i| Candidate {
                db_idx: i,
                score: 0.5,
                stratum: Stratum::All,
            })
            .collect();
        let params = RefineParams {
            n_controls: 5,
            ..Default::default()
        };
        assert!(run(&pack, &query, &candidates, params).is_err());
    }

    #[test]
    fn refine_terminates_on_synthetic_data() {
        use crate::io::db_pack;
        use crate::io::query::{Distributions, FittedGmm, Mode, Query, SnpCount};

        let tmp = tempfile::tempdir().unwrap();
        // 20 samples, 2 PCs, 4 sites — enough to run, too small for meaningful λ.
        db_pack::fixture::build(tmp.path(), 20, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();

        let query = Query {
            version: "1.0".into(),
            reference_build: "GRCh38".into(),
            n_samples: 100,
            n_snps_attempted: 4,
            n_snps_found: 4,
            per_sex_counts: None,
            counts: (0..4)
                .map(|i| SnpCount {
                    chrom: "1".into(),
                    pos: 100 + i as u64 * 10,
                    effect_allele: "G".into(),
                    other_allele: "A".into(),
                    alt_count: 50,
                    n_alleles: 200,
                })
                .collect(),
            distributions: Distributions {
                mode: Mode::None,
                n_dims: 2,
                all: Some(FittedGmm {
                    n_components: 1,
                    weights: vec![1.0],
                    means: vec![vec![0.0, 0.0]],
                    covariances: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
                }),
                female: None,
                male: None,
            },
        };

        let candidates: Vec<Candidate> = (0..15)
            .map(|i| Candidate {
                db_idx: i,
                score: (i as f32) / 14.0,
                stratum: Stratum::All,
            })
            .collect();
        let params = RefineParams {
            n_controls: 5,
            max_iter: 200,
            batch: 10,
            tol: 0.001,
            plateau: 50,
            uniform_floor: 0.05,
            seed: 42,
        };
        let result = run(&pack, &query, &candidates, params).expect("refine runs");
        assert_eq!(result.selected.len(), 5);
        assert!(result.iterations <= params.max_iter);
        // Final λ should be finite (or could be 0 if all χ²s are 0 — both are valid).
        assert!(result.final_lambda.is_finite());
    }

    #[test]
    fn refine_is_deterministic_with_seed() {
        use crate::io::db_pack;
        use crate::io::query::{Distributions, FittedGmm, Mode, Query, SnpCount};

        let tmp = tempfile::tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), 20, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();

        let query = Query {
            version: "1.0".into(),
            reference_build: "GRCh38".into(),
            n_samples: 100,
            n_snps_attempted: 4,
            n_snps_found: 4,
            per_sex_counts: None,
            counts: (0..4)
                .map(|i| SnpCount {
                    chrom: "1".into(),
                    pos: 100 + i as u64 * 10,
                    effect_allele: "G".into(),
                    other_allele: "A".into(),
                    alt_count: 50,
                    n_alleles: 200,
                })
                .collect(),
            distributions: Distributions {
                mode: Mode::None,
                n_dims: 2,
                all: Some(FittedGmm {
                    n_components: 1,
                    weights: vec![1.0],
                    means: vec![vec![0.0, 0.0]],
                    covariances: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
                }),
                female: None,
                male: None,
            },
        };
        let candidates: Vec<Candidate> = (0..15)
            .map(|i| Candidate {
                db_idx: i,
                score: (i as f32) / 14.0,
                stratum: Stratum::All,
            })
            .collect();
        let params = RefineParams {
            n_controls: 5,
            max_iter: 50,
            batch: 5,
            tol: 0.0,
            plateau: 1000,
            uniform_floor: 0.05,
            seed: 12345,
        };
        let r1 = run(&pack, &query, &candidates, params).unwrap();
        let r2 = run(&pack, &query, &candidates, params).unwrap();
        assert_eq!(r1.selected, r2.selected);
        assert_eq!(r1.iterations, r2.iterations);
        approx::assert_abs_diff_eq!(r1.final_lambda, r2.final_lambda, epsilon = 1e-12);
    }
}
