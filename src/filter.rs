//! Pre-Sinkhorn filtering of db samples.
//!
//! Currently supports excluding samples by population label. The phs-cohort
//! exclusion case is wired through the same `FilterSpec` shape but is a
//! no-op until the corresponding column is added to the db_pack samples
//! table by the preprocessing step.

use std::collections::HashSet;

use crate::io::db_pack::DbSamples;

#[derive(Debug, Clone, Default)]
pub struct FilterSpec {
    /// Population labels (e.g. "MXL", "PEL") whose samples should be excluded.
    pub exclude_populations: Vec<String>,
}

impl FilterSpec {
    pub fn is_empty(&self) -> bool {
        self.exclude_populations.is_empty()
    }
}

/// Return the db sample indices that pass the filter, in original order.
pub fn apply(samples: &DbSamples, filter: &FilterSpec) -> Vec<usize> {
    let n = samples.sample_ids.len();
    if filter.is_empty() {
        return (0..n).collect();
    }
    let exclude: HashSet<&str> = filter
        .exclude_populations
        .iter()
        .map(String::as_str)
        .collect();
    (0..n)
        .filter(|&i| {
            samples
                .population
                .get(i)
                .map(|p| !exclude.contains(p.as_str()))
                .unwrap_or(true)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn samples(populations: &[&str]) -> DbSamples {
        let n = populations.len();
        DbSamples {
            sample_ids: (0..n).map(|i| format!("s{i}")).collect(),
            sex: vec![0; n],
            age: vec![50.0; n],
            population: populations.iter().map(|s| s.to_string()).collect(),
            pca: Array2::<f32>::zeros((n, 1)),
        }
    }

    #[test]
    fn empty_filter_returns_all() {
        let s = samples(&["MXL", "PEL", "PUR"]);
        let kept = apply(&s, &FilterSpec::default());
        assert_eq!(kept, vec![0, 1, 2]);
    }

    #[test]
    fn excludes_single_population() {
        let s = samples(&["MXL", "PEL", "PUR", "MXL"]);
        let kept = apply(
            &s,
            &FilterSpec {
                exclude_populations: vec!["MXL".into()],
            },
        );
        assert_eq!(kept, vec![1, 2]);
    }

    #[test]
    fn excludes_multiple_populations() {
        let s = samples(&["MXL", "PEL", "PUR", "CLM", "MXL"]);
        let kept = apply(
            &s,
            &FilterSpec {
                exclude_populations: vec!["MXL".into(), "PUR".into()],
            },
        );
        assert_eq!(kept, vec![1, 3]);
    }

    #[test]
    fn excludes_unknown_population_keeps_all() {
        let s = samples(&["MXL", "PEL"]);
        let kept = apply(
            &s,
            &FilterSpec {
                exclude_populations: vec!["AFR".into()],
            },
        );
        assert_eq!(kept, vec![0, 1]);
    }

    #[test]
    fn excluding_all_populations_yields_empty() {
        let s = samples(&["MXL", "PEL"]);
        let kept = apply(
            &s,
            &FilterSpec {
                exclude_populations: vec!["MXL".into(), "PEL".into()],
            },
        );
        assert!(kept.is_empty());
    }
}
