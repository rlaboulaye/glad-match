//! Top-level orchestrator: query + db_pack → matched control set + diagnostics.
//!
//! Branches on `query.distributions.mode`:
//! - **sex_and_age / sex_only**: runs Sinkhorn twice (per-sex db × per-sex
//!   GMM), allocates pool per sex in proportion to query per-sex counts,
//!   then refines with within-sex swap constraints (the refinement module
//!   handles the stratification automatically based on Candidate tags).
//! - **age_only / none**: single Sinkhorn pass over all db samples.
//!
//! All randomness flows from a single user-provided seed.

use std::collections::HashSet;

use crate::candidates::{self, Candidate, Stratum};
use crate::cost;
use crate::error::{Error, Result};
use crate::features::{self, FeatureLayout};
use crate::filter::{self, FilterSpec};
use crate::io::db_pack::DbPack;
use crate::io::query::{FittedGmm, Query};
use crate::io::summary_json::{
    self, InputSummary, PerSexSummary, RefinementSummary, SCHEMA_VERSION, SelectedSummary,
    SinkhornSummary, Summary,
};
use crate::ot::{self, SinkhornParams};
use crate::refine::{self, RefineParams, RefineResult};

#[derive(Debug, Clone)]
pub struct MatchParams {
    pub n_controls: usize,
    /// Multiplier on `n_controls` for the candidate pool (default 4).
    pub pool_factor: usize,
    pub seed: u64,
    pub sinkhorn: SinkhornParams,
    pub refine: RefineParams,
    /// Optional pre-Sinkhorn db sample filter.
    pub filter: FilterSpec,
}

impl Default for MatchParams {
    fn default() -> Self {
        let refine = RefineParams::default();
        Self {
            n_controls: refine.n_controls,
            pool_factor: 4,
            seed: refine.seed,
            sinkhorn: SinkhornParams::default(),
            refine,
            filter: FilterSpec::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SinkhornRunInfo {
    /// "all" | "female" | "male".
    pub group: String,
    pub iters: usize,
    pub objective: f32,
    pub eps_used: f32,
    pub n_db_in_stratum: usize,
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub selected: Vec<usize>,
    pub initial_lambda: f64,
    pub final_lambda: f64,
    pub iterations: usize,
    pub accepted_swaps: usize,
    pub sinkhorn_runs: Vec<SinkhornRunInfo>,
}

pub fn match_controls(query: &Query, pack: &DbPack, params: &MatchParams) -> Result<MatchResult> {
    if params.n_controls == 0 {
        return Err(Error::Schema("n_controls must be > 0".into()));
    }
    if params.pool_factor == 0 {
        return Err(Error::Schema("pool_factor must be > 0".into()));
    }

    let pool_size = params.n_controls * params.pool_factor;
    let mode = query.distributions.mode;

    // Apply pre-Sinkhorn filter (population exclusion etc.) once. The
    // surviving indices form the universe used for sex partitioning and
    // single-GMM index sets below.
    let allowed: HashSet<usize> = filter::apply(&pack.samples, &params.filter)
        .into_iter()
        .collect();
    if allowed.is_empty() {
        return Err(Error::Schema("filter excluded every db sample".into()));
    }

    let mut sinkhorn_runs: Vec<SinkhornRunInfo> = Vec::new();
    let mut all_candidates: Vec<Candidate> = Vec::new();

    if mode.has_sex() {
        let female_gmm = query
            .distributions
            .female
            .as_ref()
            .ok_or_else(|| Error::Schema("sex-split mode requires female GMM".into()))?;
        let male_gmm = query
            .distributions
            .male
            .as_ref()
            .ok_or_else(|| Error::Schema("sex-split mode requires male GMM".into()))?;
        let per_sex = query.per_sex_counts.ok_or_else(|| {
            Error::Schema("sex-split mode requires per_sex_counts in query".into())
        })?;

        let layout_f = FeatureLayout::from_gmm(female_gmm, mode)?;
        let layout_m = FeatureLayout::from_gmm(male_gmm, mode)?;
        if layout_f.n_dims() != layout_m.n_dims() {
            return Err(Error::Schema(format!(
                "female GMM n_dims={} differs from male GMM n_dims={}",
                layout_f.n_dims(),
                layout_m.n_dims()
            )));
        }

        let (female_idx, male_idx) = features::indices_by_sex(&pack.samples);
        let female_idx: Vec<usize> = female_idx
            .into_iter()
            .filter(|i| allowed.contains(i))
            .collect();
        let male_idx: Vec<usize> = male_idx
            .into_iter()
            .filter(|i| allowed.contains(i))
            .collect();
        if female_idx.is_empty() && male_idx.is_empty() {
            return Err(Error::Schema(
                "db_pack has no samples with valid sex labels surviving filter".into(),
            ));
        }

        let (pool_f, pool_m) =
            candidates::allocate_pool_sex(pool_size, per_sex.female, per_sex.male);

        if !female_idx.is_empty() && pool_f > 0 {
            let (cs, info) = stratum_pipeline(
                pack,
                female_gmm,
                &female_idx,
                layout_f,
                Stratum::Female,
                pool_f,
                params,
            )?;
            sinkhorn_runs.push(info);
            all_candidates.extend(cs);
        }
        if !male_idx.is_empty() && pool_m > 0 {
            let (cs, info) = stratum_pipeline(
                pack,
                male_gmm,
                &male_idx,
                layout_m,
                Stratum::Male,
                pool_m,
                params,
            )?;
            sinkhorn_runs.push(info);
            all_candidates.extend(cs);
        }
    } else {
        let gmm = query
            .distributions
            .all
            .as_ref()
            .ok_or_else(|| Error::Schema("non-sex mode requires `all` GMM".into()))?;
        let layout = FeatureLayout::from_gmm(gmm, mode)?;
        let all_idx: Vec<usize> = features::all_indices(&pack.samples)
            .into_iter()
            .filter(|i| allowed.contains(i))
            .collect();
        if all_idx.is_empty() {
            return Err(Error::Schema("no db samples survive filter".into()));
        }
        let (cs, info) =
            stratum_pipeline(pack, gmm, &all_idx, layout, Stratum::All, pool_size, params)?;
        sinkhorn_runs.push(info);
        all_candidates.extend(cs);
    }

    if all_candidates.len() < params.n_controls {
        return Err(Error::Schema(format!(
            "candidate pool size {} < requested controls {}",
            all_candidates.len(),
            params.n_controls
        )));
    }

    let refine_params = RefineParams {
        n_controls: params.n_controls,
        seed: params.seed,
        ..params.refine
    };
    let RefineResult {
        selected,
        initial_lambda,
        final_lambda,
        iterations,
        accepted_swaps,
    } = refine::run(pack, query, &all_candidates, refine_params)?;

    Ok(MatchResult {
        selected,
        initial_lambda,
        final_lambda,
        iterations,
        accepted_swaps,
        sinkhorn_runs,
    })
}

/// Build a Summary from pipeline outputs, ready for JSON serialization.
///
/// `db_n_samples_used` is whatever was effectively in scope (after any
/// pre-filtering); for now this is just `pack.samples.sample_ids.len()`.
pub fn build_summary(
    query: &Query,
    pack: &DbPack,
    result: &MatchResult,
    seed: u64,
    db_n_samples_used: usize,
    age_bin_edges: &[f32],
    k_anon_min: u32,
) -> Summary {
    let input = InputSummary {
        query_n_samples: query.n_samples,
        query_mode: query.distributions.mode.label().to_string(),
        query_per_sex: query.per_sex_counts.map(|p| PerSexSummary {
            female: p.female,
            male: p.male,
        }),
        query_n_snps_found: query.n_snps_found,
        db_n_samples_used,
    };

    let sinkhorn: Vec<SinkhornSummary> = result
        .sinkhorn_runs
        .iter()
        .map(|r| SinkhornSummary {
            group: r.group.clone(),
            iters: r.iters,
            objective: r.objective,
            eps_used: r.eps_used,
        })
        .collect();

    let refinement = RefinementSummary {
        initial_lambda: result.initial_lambda,
        final_lambda: result.final_lambda,
        iterations: result.iterations,
        accepted_swaps: result.accepted_swaps,
    };

    let selected: SelectedSummary = summary_json::build_selected_summary(
        &result.selected,
        &pack.samples,
        age_bin_edges,
        k_anon_min,
    );

    Summary {
        version: SCHEMA_VERSION.into(),
        glad_match_version: env!("CARGO_PKG_VERSION").into(),
        rng_seed: seed,
        input,
        sinkhorn,
        refinement,
        selected,
        warnings: vec![],
    }
}

fn stratum_pipeline(
    pack: &DbPack,
    gmm: &FittedGmm,
    db_indices: &[usize],
    layout: FeatureLayout,
    stratum: Stratum,
    pool_size: usize,
    params: &MatchParams,
) -> Result<(Vec<Candidate>, SinkhornRunInfo)> {
    let features_mat = features::build(
        &pack.samples,
        db_indices,
        layout,
        pack.manifest.age_mean,
        pack.manifest.age_sd,
    )?;
    let cost_mat = cost::mahalanobis(&features_mat, gmm)?;
    let a = ot::uniform_source(db_indices.len());
    let b = ot::gmm_target(&gmm.weights);
    let sink = ot::run(&cost_mat, &a, &b, params.sinkhorn)?;

    let scores = candidates::relevance_scores(&sink.plan);
    let cs = candidates::select_top(&scores, db_indices, pool_size, stratum);

    let group = match stratum {
        Stratum::All => "all",
        Stratum::Female => "female",
        Stratum::Male => "male",
    }
    .to_string();

    let info = SinkhornRunInfo {
        group,
        iters: sink.iterations,
        objective: sink.objective,
        eps_used: sink.eps_used,
        n_db_in_stratum: db_indices.len(),
    };
    Ok((cs, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::db_pack;
    use crate::io::query::{Distributions, FittedGmm, Mode, PerSexCounts, Query, SnpCount};

    fn dummy_query_none(n_pcs: usize) -> Query {
        Query {
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
                n_dims: n_pcs,
                all: Some(FittedGmm {
                    n_components: 1,
                    weights: vec![1.0],
                    means: vec![vec![0.0; n_pcs]],
                    covariances: vec![
                        vec![vec![0.0; n_pcs]; n_pcs]
                            .into_iter()
                            .enumerate()
                            .map(|(i, mut row)| {
                                row[i] = 1.0;
                                row
                            })
                            .collect(),
                    ],
                }),
                female: None,
                male: None,
            },
        }
    }

    fn identity_cov(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; n]; n];
        for (i, row) in m.iter_mut().enumerate().take(n) {
            row[i] = 1.0;
        }
        m
    }

    fn dummy_query_sex(n_pcs_with_age: usize, query_f: u32, query_m: u32) -> Query {
        let gmm = FittedGmm {
            n_components: 1,
            weights: vec![1.0],
            means: vec![vec![0.0; n_pcs_with_age]],
            covariances: vec![identity_cov(n_pcs_with_age)],
        };
        Query {
            version: "1.0".into(),
            reference_build: "GRCh38".into(),
            n_samples: (query_f + query_m) as usize,
            n_snps_attempted: 4,
            n_snps_found: 4,
            per_sex_counts: Some(PerSexCounts {
                female: query_f,
                male: query_m,
            }),
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
                mode: Mode::SexAndAge,
                n_dims: n_pcs_with_age,
                all: None,
                female: Some(gmm.clone()),
                male: Some(gmm),
            },
        }
    }

    fn small_params(n_controls: usize, pool_factor: usize) -> MatchParams {
        MatchParams {
            n_controls,
            pool_factor,
            refine: RefineParams {
                n_controls,
                batch: 5,
                max_iter: 50,
                plateau: 20,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn filter_narrows_db_universe() {
        let tmp = tempfile::tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), 30, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();
        // Fixture sets every sample's population to "MXL"; excluding it
        // leaves zero samples → pipeline must error cleanly.
        let q = dummy_query_none(2);
        let params = MatchParams {
            filter: FilterSpec {
                exclude_populations: vec!["MXL".into()],
            },
            ..small_params(5, 4)
        };
        let err = match_controls(&q, &pack, &params).unwrap_err();
        assert!(matches!(err, Error::Schema(_)));
    }

    #[test]
    fn end_to_end_single_gmm_path() {
        let tmp = tempfile::tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), 30, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();

        let q = dummy_query_none(2);
        let params = small_params(5, 4);

        let result = match_controls(&q, &pack, &params).expect("pipeline runs");
        assert_eq!(result.selected.len(), 5);
        assert_eq!(result.sinkhorn_runs.len(), 1);
        assert_eq!(result.sinkhorn_runs[0].group, "all");
        assert_eq!(result.sinkhorn_runs[0].n_db_in_stratum, 30);
        assert!(result.iterations <= params.refine.max_iter);
    }

    #[test]
    fn end_to_end_sex_split_path() {
        let tmp = tempfile::tempdir().unwrap();
        // 30 samples, 2 PCs, 4 sites; fixture sex pattern is i%2 → 15 female, 15 male.
        db_pack::fixture::build(tmp.path(), 30, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();

        // GMM uses n_pcs_with_age = 2 (1 PC + age) so layout.n_pcs_used = 1
        // and the synthetic 2-PC db_pack accommodates it.
        let q = dummy_query_sex(2, 6, 4); // 60% female / 40% male

        let params = small_params(10, 2);

        let result = match_controls(&q, &pack, &params).expect("pipeline runs");
        assert_eq!(result.selected.len(), 10);
        assert_eq!(result.sinkhorn_runs.len(), 2);
        let groups: Vec<&str> = result
            .sinkhorn_runs
            .iter()
            .map(|r| r.group.as_str())
            .collect();
        assert!(groups.contains(&"female"));
        assert!(groups.contains(&"male"));

        // Selected sex ratio should match query allocation 6:4.
        let female_selected = result
            .selected
            .iter()
            .filter(|&&i| pack.samples.sex[i] == 0)
            .count();
        let male_selected = result.selected.len() - female_selected;
        // pool_size = 20; allocate_pool_sex(20, 6, 4) = (12, 8); after refine
        // selecting n=10 from a 20-pool with within-sex swaps preserves the
        // initial allocation ratio. With proportional initial picks we expect
        // ~6 female and ~4 male.
        assert_eq!(female_selected + male_selected, 10);
        // Ratio should be in the ballpark
        assert!((female_selected as i32 - 6).abs() <= 1);
        assert!((male_selected as i32 - 4).abs() <= 1);
    }

    #[test]
    fn rejects_sex_mode_without_per_sex_counts() {
        let tmp = tempfile::tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), 30, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();

        let mut q = dummy_query_sex(2, 6, 4);
        q.per_sex_counts = None;

        let params = small_params(10, 2);
        let err = match_controls(&q, &pack, &params).unwrap_err();
        assert!(matches!(err, Error::Schema(_)));
    }

    #[test]
    fn deterministic_with_seed() {
        let tmp = tempfile::tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), 30, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();
        let q = dummy_query_none(2);

        let params = MatchParams {
            n_controls: 5,
            pool_factor: 4,
            seed: 12345,
            refine: RefineParams {
                n_controls: 5,
                seed: 12345,
                batch: 5,
                max_iter: 30,
                plateau: 10,
                ..Default::default()
            },
            ..Default::default()
        };

        let r1 = match_controls(&q, &pack, &params).unwrap();
        let r2 = match_controls(&q, &pack, &params).unwrap();
        assert_eq!(r1.selected, r2.selected);
        assert_eq!(r1.iterations, r2.iterations);
        approx::assert_abs_diff_eq!(r1.final_lambda, r2.final_lambda, epsilon = 1e-12);
    }
}
