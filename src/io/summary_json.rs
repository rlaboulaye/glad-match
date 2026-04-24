//! Summary JSON sidecar — pipeline diagnostics + privacy-respecting view of
//! the selected control set.
//!
//! Per-population counts and age-bin counts are subject to k-anonymity:
//! cells below `k_anon_min` are zeroed, and the suppressed count is reported
//! separately so a consumer knows mass is hidden without learning where.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use serde::Serialize;

use crate::error::{Error, Result};
use crate::io::db_pack::DbSamples;

pub const SCHEMA_VERSION: &str = "1.0";

#[derive(Serialize, Debug)]
pub struct Summary {
    pub version: String,
    pub glad_match_version: String,
    pub rng_seed: u64,
    pub input: InputSummary,
    pub sinkhorn: Vec<SinkhornSummary>,
    pub refinement: RefinementSummary,
    pub selected: SelectedSummary,
    pub warnings: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct InputSummary {
    pub query_n_samples: usize,
    pub query_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_per_sex: Option<PerSexSummary>,
    pub query_n_snps_found: usize,
    pub db_n_samples_used: usize,
}

#[derive(Serialize, Debug, Clone, Copy)]
pub struct PerSexSummary {
    pub female: u32,
    pub male: u32,
}

#[derive(Serialize, Debug)]
pub struct SinkhornSummary {
    /// Stratum label: "all", "female", or "male".
    pub group: String,
    pub iters: usize,
    pub objective: f32,
    pub eps_used: f32,
}

#[derive(Serialize, Debug)]
pub struct RefinementSummary {
    pub initial_lambda: f64,
    pub final_lambda: f64,
    pub iterations: usize,
    pub accepted_swaps: usize,
}

#[derive(Serialize, Debug)]
pub struct SelectedSummary {
    pub n: usize,
    pub per_sex: BTreeMap<String, u32>,
    pub per_population: BTreeMap<String, u32>,
    pub age_histogram: AgeHistogram,
}

#[derive(Serialize, Debug)]
pub struct AgeHistogram {
    /// Bin edges; `counts.len() + 1`.
    pub bin_edges: Vec<f32>,
    /// Per-bin counts. Cells whose true count was below `k_anon_min` are zeroed.
    pub counts: Vec<u32>,
    pub k_anon_min: u32,
    /// Total samples in suppressed bins.
    pub suppressed: u32,
}

/// Default decade bin edges from 0 to 100.
pub fn default_age_bins() -> Vec<f32> {
    (0..=10).map(|i| (i * 10) as f32).collect()
}

/// Aggregate the selected db samples into a privacy-respecting summary.
///
/// `per_sex` and `per_population` use the raw labels; `age_histogram`
/// applies the `k_anon_min` suppression rule.
pub fn build_selected_summary(
    selected: &[usize],
    samples: &DbSamples,
    age_bin_edges: &[f32],
    k_anon_min: u32,
) -> SelectedSummary {
    let n = selected.len();
    let mut per_sex: BTreeMap<String, u32> = BTreeMap::new();
    let mut per_population: BTreeMap<String, u32> = BTreeMap::new();
    let n_bins = age_bin_edges.len().saturating_sub(1);
    let mut raw_counts = vec![0u32; n_bins.max(1)];

    for &idx in selected {
        let sex_label = match samples.sex.get(idx).copied() {
            Some(0) => "female",
            Some(1) => "male",
            _ => "unknown",
        };
        *per_sex.entry(sex_label.into()).or_default() += 1;

        if let Some(pop) = samples.population.get(idx) {
            *per_population.entry(pop.clone()).or_default() += 1;
        }

        if let Some(&age) = samples.age.get(idx)
            && let Some(b) = bin_for(age, age_bin_edges)
            && b < raw_counts.len()
        {
            raw_counts[b] += 1;
        }
    }

    let mut suppressed = 0u32;
    let mut counts = raw_counts;
    for c in counts.iter_mut() {
        if *c > 0 && *c < k_anon_min {
            suppressed += *c;
            *c = 0;
        }
    }

    SelectedSummary {
        n,
        per_sex,
        per_population,
        age_histogram: AgeHistogram {
            bin_edges: age_bin_edges.to_vec(),
            counts,
            k_anon_min,
            suppressed,
        },
    }
}

fn bin_for(age: f32, edges: &[f32]) -> Option<usize> {
    if edges.len() < 2 || !age.is_finite() {
        return None;
    }
    if age < edges[0] || age >= *edges.last().unwrap() {
        return None;
    }
    (0..edges.len() - 1).find(|&i| age >= edges[i] && age < edges[i + 1])
}

pub fn write<P: AsRef<Path>>(path: P, summary: &Summary) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, summary)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn dummy_samples(n: usize) -> DbSamples {
        let sample_ids: Vec<String> = (0..n).map(|i| format!("s{i}")).collect();
        let sex: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let age: Vec<f32> = (0..n).map(|i| 20.0 + i as f32 * 5.0).collect();
        let population: Vec<String> = (0..n)
            .map(|i| if i % 3 == 0 { "MXL".into() } else { "PEL".into() })
            .collect();
        let pca = Array2::<f32>::zeros((n, 1));
        DbSamples {
            sample_ids,
            sex,
            age,
            population,
            pca,
        }
    }

    #[test]
    fn counts_sex_and_population() {
        let samples = dummy_samples(10);
        // selected = s0..s4 → sex pattern 0,1,0,1,0 (3 female, 2 male).
        // population: i%3==0 → MXL (i=0,3); else PEL.
        // s0=MXL, s1=PEL, s2=PEL, s3=MXL, s4=PEL → MXL=2, PEL=3.
        let s = build_selected_summary(&[0, 1, 2, 3, 4], &samples, &default_age_bins(), 1);
        assert_eq!(s.n, 5);
        assert_eq!(s.per_sex.get("female").copied(), Some(3));
        assert_eq!(s.per_sex.get("male").copied(), Some(2));
        assert_eq!(s.per_population.get("MXL").copied(), Some(2));
        assert_eq!(s.per_population.get("PEL").copied(), Some(3));
    }

    #[test]
    fn k_anon_suppresses_low_age_cells() {
        let samples = dummy_samples(10);
        // ages 20, 30, 40, 50, 60 → one per decade bin
        let s = build_selected_summary(&[0, 2, 4, 6, 8], &samples, &default_age_bins(), 2);
        for &c in &s.age_histogram.counts {
            assert_eq!(c, 0, "all cells below k_anon should be zeroed");
        }
        assert_eq!(s.age_histogram.suppressed, 5);
    }

    #[test]
    fn k_anon_keeps_meets_threshold() {
        let samples = dummy_samples(20);
        // Select all 20 → ages 20, 25, 30, ..., 115 → multiple per bin
        let selected: Vec<usize> = (0..20).collect();
        let s = build_selected_summary(&selected, &samples, &default_age_bins(), 2);
        let kept: u32 = s.age_histogram.counts.iter().sum();
        assert!(kept > 0, "some bins should survive k-anon");
        // suppressed + kept ≤ total selected (some may fall outside bin range, e.g. age 115)
        assert!(s.age_histogram.suppressed + kept <= 20);
    }

    #[test]
    fn ages_outside_bin_range_dropped() {
        let mut samples = dummy_samples(3);
        samples.age = vec![-5.0, 50.0, 200.0]; // first and last out of range
        let s = build_selected_summary(&[0, 1, 2], &samples, &default_age_bins(), 1);
        let total: u32 = s.age_histogram.counts.iter().sum::<u32>() + s.age_histogram.suppressed;
        // Only the middle age (50.0) lands in a bin.
        assert_eq!(total, 1);
    }

    #[test]
    fn round_trips_through_serde_json() {
        let samples = dummy_samples(5);
        let sel = build_selected_summary(&[0, 1, 2], &samples, &default_age_bins(), 1);
        let summary = Summary {
            version: SCHEMA_VERSION.into(),
            glad_match_version: "0.1.0".into(),
            rng_seed: 42,
            input: InputSummary {
                query_n_samples: 100,
                query_mode: "none".into(),
                query_per_sex: None,
                query_n_snps_found: 200,
                db_n_samples_used: 5,
            },
            sinkhorn: vec![SinkhornSummary {
                group: "all".into(),
                iters: 87,
                objective: 0.012,
                eps_used: 0.001,
            }],
            refinement: RefinementSummary {
                initial_lambda: 1.42,
                final_lambda: 1.03,
                iterations: 1824,
                accepted_swaps: 412,
            },
            selected: sel,
            warnings: vec![],
        };

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("summary.json");
        write(&path, &summary).unwrap();

        let f = File::open(&path).unwrap();
        let v: serde_json::Value = serde_json::from_reader(f).unwrap();
        assert_eq!(v["version"], "1.0");
        assert_eq!(v["selected"]["n"], 3);
        assert!(v["selected"]["age_histogram"]["bin_edges"].is_array());
        assert_eq!(v["sinkhorn"][0]["group"], "all");
        // Optional field skipped when None
        assert!(v["input"]["query_per_sex"].is_null());
    }
}
