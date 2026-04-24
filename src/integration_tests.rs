//! End-to-end pipeline smoke tests.
//!
//! These exercise the modules together against the real `query.glad.gz`
//! fixture in the repository root and a synthetic `db_pack` built on the fly.

use std::path::Path;
use tempfile::tempdir;

use crate::cost;
use crate::features::{self, FeatureLayout};
use crate::io::{db_pack, output_tsv, query, summary_json};
use crate::ot::{self, SinkhornParams};
use crate::pipeline::{self, MatchParams};
use crate::refine::RefineParams;

/// Pick the first available GMM regardless of split mode. For mode=sex_*, this
/// is the female GMM; for mode=age_only or none, this is the `all` GMM.
fn pick_gmm(q: &query::Query) -> &query::FittedGmm {
    q.distributions
        .female
        .as_ref()
        .or(q.distributions.male.as_ref())
        .or(q.distributions.all.as_ref())
        .expect("validated query has at least one GMM")
}

#[test]
fn single_gmm_pipeline_smoke() {
    let query_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("query.glad.gz");
    if !query_path.exists() {
        eprintln!("skipping: query.glad.gz fixture not present");
        return;
    }
    let q = query::read(&query_path).expect("parse query.glad.gz");
    let gmm = pick_gmm(&q);
    let layout =
        FeatureLayout::from_gmm(gmm, q.distributions.mode).expect("build layout from GMM");

    let tmp = tempdir().unwrap();
    db_pack::fixture::build(tmp.path(), 200, layout.n_pcs_used, 4);
    let pack = db_pack::load(tmp.path()).expect("load synthetic db_pack");

    // Use the female-indexed half of the synthetic samples; this exercises
    // the path that the sex-split pipeline will use later (one stratum at a
    // time). For non-sex modes, indices_by_sex still partitions cleanly.
    let (female_idx, _male_idx) = features::indices_by_sex(&pack.samples);
    assert!(!female_idx.is_empty());

    let f = features::build(
        &pack.samples,
        &female_idx,
        layout,
        pack.manifest.age_mean,
        pack.manifest.age_sd,
    )
    .expect("build features");
    assert_eq!(f.shape(), &[female_idx.len(), layout.n_dims()]);

    let cost_mat = cost::mahalanobis(&f, gmm).expect("build cost");
    assert_eq!(cost_mat.shape(), &[female_idx.len(), gmm.n_components]);
    // Cost entries finite & non-negative.
    for &c in cost_mat.iter() {
        assert!(c.is_finite() && c >= 0.0, "non-finite or negative cost: {c}");
    }

    let a = ot::uniform_source(female_idx.len());
    let b = ot::gmm_target(&gmm.weights);
    assert!(
        (a.iter().sum::<f32>() - 1.0).abs() < 1e-5,
        "source mass should sum to 1"
    );
    assert!(
        (b.iter().sum::<f32>() - 1.0).abs() < 1e-3,
        "GMM weights should sum to ~1"
    );

    let result = ot::run(&cost_mat, &a, &b, SinkhornParams::default()).expect("sinkhorn");
    assert_eq!(result.plan.shape(), &[female_idx.len(), gmm.n_components]);
    assert!(result.iterations > 0);
    assert!(result.eps_used > 0.0);

    // Plan entries must be non-negative and finite. Total mass should be in
    // a sane range for unbalanced OT with both marginals at unit mass.
    let mut total = 0.0_f32;
    for &p in result.plan.iter() {
        assert!(p.is_finite() && p >= 0.0, "non-finite/negative plan entry: {p}");
        total += p;
    }
    assert!(
        total > 0.0 && total <= 1.5,
        "plan total mass out of range: {total}"
    );
}

/// End-to-end on the real `query.glad.gz` (sex_and_age, 31 dims, 4 + 8
/// component GMMs). Builds a synthetic db_pack with sites taken from the
/// query so the TSV writer emits joined rows; runs the full pipeline,
/// asserts sex-ratio preservation and that both output artifacts are written.
#[test]
fn real_query_full_pipeline() {
    let query_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("query.glad.gz");
    if !query_path.exists() {
        eprintln!("skipping: query.glad.gz fixture not present");
        return;
    }
    let q = query::read(&query_path).expect("parse query.glad.gz");
    assert!(q.distributions.mode.has_sex(), "fixture is sex_and_age");
    let psc = q.per_sex_counts.expect("regenerated query carries per_sex_counts");

    let female_gmm = q.distributions.female.as_ref().unwrap();
    let layout = FeatureLayout::from_gmm(female_gmm, q.distributions.mode).unwrap();
    let n_pcs_used = layout.n_pcs_used;

    // Sites: take the first 50 entries from query.counts so the TSV writer
    // produces non-trivial output (otherwise the inner-join is empty).
    // Note db_pack stores (chrom, pos, ref, alt); query has effect_allele=alt,
    // other_allele=ref, so we swap accordingly.
    let sites: Vec<(String, u64, String, String)> = q
        .counts
        .iter()
        .take(50)
        .map(|c| {
            (
                c.chrom.clone(),
                c.pos,
                c.other_allele.clone(),
                c.effect_allele.clone(),
            )
        })
        .collect();
    assert!(sites.len() >= 20, "fixture should yield at least 20 sites");

    let tmp = tempdir().unwrap();
    // 100 samples, 50 female + 50 male via fixture's i%2 sex pattern.
    db_pack::fixture::build_with_sites(tmp.path(), 100, n_pcs_used, &sites);
    let pack = db_pack::load(tmp.path()).unwrap();

    let n_controls = 20;
    let params = MatchParams {
        n_controls,
        pool_factor: 2,
        seed: 7,
        refine: RefineParams {
            n_controls,
            seed: 7,
            batch: 5,
            max_iter: 30,
            plateau: 10,
            ..Default::default()
        },
        ..Default::default()
    };

    let result = pipeline::match_controls(&q, &pack, &params).expect("pipeline runs");
    assert_eq!(result.selected.len(), n_controls);
    assert_eq!(result.sinkhorn_runs.len(), 2);
    let groups: Vec<&str> = result
        .sinkhorn_runs
        .iter()
        .map(|r| r.group.as_str())
        .collect();
    assert!(groups.contains(&"female") && groups.contains(&"male"));

    // Sex-ratio check: with within-sex swaps, the selected ratio matches the
    // initial allocation, which itself comes from the per-sex query counts.
    let total = (psc.female + psc.male) as f32;
    let target_female = (n_controls as f32 * psc.female as f32 / total).round() as i32;
    let actual_female = result
        .selected
        .iter()
        .filter(|&&i| pack.samples.sex[i] == 0)
        .count() as i32;
    assert!(
        (actual_female - target_female).abs() <= 1,
        "selected female count {actual_female} not within ±1 of target {target_female}"
    );

    // Write outputs and verify they are non-empty.
    let tsv_path = tmp.path().join("out.tsv.gz");
    output_tsv::write(&tsv_path, &pack, &result.selected).expect("write TSV");
    assert!(tsv_path.metadata().unwrap().len() > 0);

    let summary = pipeline::build_summary(
        &q,
        &pack,
        &result,
        params.seed,
        pack.samples.sample_ids.len(),
        &summary_json::default_age_bins(),
        5,
    );
    let json_path = tmp.path().join("summary.json");
    summary_json::write(&json_path, &summary).expect("write summary");
    let json_bytes = std::fs::read(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
    assert_eq!(v["selected"]["n"], n_controls);
    assert_eq!(v["sinkhorn"].as_array().unwrap().len(), 2);
    assert_eq!(v["input"]["query_mode"], "sex_and_age");
    assert!(v["input"]["query_per_sex"]["female"].as_u64().unwrap() > 0);
}
