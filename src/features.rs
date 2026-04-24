//! Feature matrix construction for db samples.
//!
//! Mirrors `glad-prep/src/gmm.rs:89-117`: features are `[PC_0, …, PC_{k-1}]`
//! where `k = n_pcs_used`, optionally followed by a z-scored age using
//! glad_meta's `age_mean` / `age_sd`. This matches the standardization the
//! query side applied when fitting the GMM, so db and GMM live in the same
//! feature space.

use ndarray::Array2;

use crate::error::{Error, Result};
use crate::io::db_pack::DbSamples;
use crate::io::query::{FittedGmm, Mode};

#[derive(Debug, Clone, Copy)]
pub struct FeatureLayout {
    /// Number of leading PCs used as features (must be ≤ db's n_pcs).
    pub n_pcs_used: usize,
    /// Whether to append a z-scored age column.
    pub include_age: bool,
}

impl FeatureLayout {
    pub fn from_gmm(gmm: &FittedGmm, mode: Mode) -> Result<Self> {
        let n_dims = gmm
            .means
            .first()
            .map(Vec::len)
            .ok_or_else(|| Error::Schema("GMM has no components".into()))?;
        let include_age = mode.has_age();
        let age_dim = if include_age { 1 } else { 0 };
        let n_pcs_used = n_dims.checked_sub(age_dim).ok_or_else(|| {
            Error::Schema(format!(
                "GMM n_dims={n_dims} is inconsistent with age dim {age_dim}"
            ))
        })?;
        Ok(Self {
            n_pcs_used,
            include_age,
        })
    }

    pub fn n_dims(&self) -> usize {
        self.n_pcs_used + usize::from(self.include_age)
    }
}

/// Build the `(n_rows × n_dims)` feature matrix for the given db-sample indices.
pub fn build(
    samples: &DbSamples,
    indices: &[usize],
    layout: FeatureLayout,
    age_mean: f64,
    age_sd: f64,
) -> Result<Array2<f32>> {
    let available_pcs = samples.pca.shape()[1];
    if layout.n_pcs_used > available_pcs {
        return Err(Error::Schema(format!(
            "GMM requires {} PCs but db_pack has {available_pcs}",
            layout.n_pcs_used
        )));
    }
    if layout.include_age && (!age_sd.is_finite() || age_sd == 0.0) {
        return Err(Error::Schema(format!(
            "age_sd={age_sd} is invalid for z-scoring"
        )));
    }

    let n = indices.len();
    let d = layout.n_dims();
    let mut features = Array2::<f32>::zeros((n, d));
    for (row, &idx) in indices.iter().enumerate() {
        for j in 0..layout.n_pcs_used {
            features[(row, j)] = samples.pca[(idx, j)];
        }
        if layout.include_age {
            let age = samples.age[idx] as f64;
            let z = (age - age_mean) / age_sd;
            features[(row, layout.n_pcs_used)] = z as f32;
        }
    }
    Ok(features)
}

/// Partition db sample indices by sex (0 = female, 1 = male).
pub fn indices_by_sex(samples: &DbSamples) -> (Vec<usize>, Vec<usize>) {
    let mut female = Vec::new();
    let mut male = Vec::new();
    for (i, &s) in samples.sex.iter().enumerate() {
        match s {
            0 => female.push(i),
            1 => male.push(i),
            _ => {} // out-of-range sex values are dropped (shouldn't occur in validated db_pack)
        }
    }
    (female, male)
}

pub fn all_indices(samples: &DbSamples) -> Vec<usize> {
    (0..samples.sample_ids.len()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::db_pack::{self, DbPack};
    use tempfile::tempdir;

    fn load_small(n_samples: usize, n_pcs: usize, n_sites: usize) -> (tempfile::TempDir, DbPack) {
        let tmp = tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), n_samples, n_pcs, n_sites);
        let pack = db_pack::load(tmp.path()).unwrap();
        (tmp, pack)
    }

    fn gmm(n_components: usize, n_dims: usize) -> FittedGmm {
        FittedGmm {
            n_components,
            weights: vec![1.0 / n_components as f64; n_components],
            means: vec![vec![0.0; n_dims]; n_components],
            covariances: vec![vec![vec![0.0; n_dims]; n_dims]; n_components],
        }
    }

    #[test]
    fn layout_without_age_matches_pc_count() {
        let layout = FeatureLayout::from_gmm(&gmm(2, 5), Mode::None).unwrap();
        assert_eq!(layout.n_pcs_used, 5);
        assert!(!layout.include_age);
        assert_eq!(layout.n_dims(), 5);
    }

    #[test]
    fn layout_with_age_strips_last_dim_as_pc_count() {
        let layout = FeatureLayout::from_gmm(&gmm(1, 6), Mode::SexAndAge).unwrap();
        assert_eq!(layout.n_pcs_used, 5);
        assert!(layout.include_age);
        assert_eq!(layout.n_dims(), 6);
    }

    #[test]
    fn build_no_age_copies_pcs() {
        let (_tmp, pack) = load_small(4, 3, 2);
        let layout = FeatureLayout {
            n_pcs_used: 2,
            include_age: false,
        };
        let f = build(&pack.samples, &[0, 2], layout, 50.0, 10.0).unwrap();
        assert_eq!(f.shape(), &[2, 2]);
        // fixture PCA: pca[i, j] = (i * n_pcs + j) * 0.01
        approx::assert_abs_diff_eq!(f[(0, 0)], 0.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(f[(0, 1)], 0.01, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(f[(1, 0)], 0.06, epsilon = 1e-6); // i=2, j=0 → 6*0.01
        approx::assert_abs_diff_eq!(f[(1, 1)], 0.07, epsilon = 1e-6);
    }

    #[test]
    fn build_with_age_zscored() {
        let (_tmp, pack) = load_small(3, 2, 2);
        let layout = FeatureLayout {
            n_pcs_used: 2,
            include_age: true,
        };
        let age_mean = 40.0_f64;
        let age_sd = 1.0_f64;
        let f = build(&pack.samples, &[0, 1, 2], layout, age_mean, age_sd).unwrap();
        assert_eq!(f.shape(), &[3, 3]);
        // fixture age[i] = 40.0 + i → z = i
        approx::assert_abs_diff_eq!(f[(0, 2)], 0.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(f[(1, 2)], 1.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(f[(2, 2)], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn build_rejects_too_many_pcs() {
        let (_tmp, pack) = load_small(3, 2, 2);
        let layout = FeatureLayout {
            n_pcs_used: 5,
            include_age: false,
        };
        assert!(build(&pack.samples, &[0], layout, 0.0, 1.0).is_err());
    }

    #[test]
    fn indices_by_sex_partitions() {
        let (_tmp, pack) = load_small(6, 2, 2);
        // fixture sex = i % 2, so female=[0,2,4], male=[1,3,5]
        let (f, m) = indices_by_sex(&pack.samples);
        assert_eq!(f, vec![0, 2, 4]);
        assert_eq!(m, vec![1, 3, 5]);
    }
}
