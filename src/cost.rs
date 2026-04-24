//! Squared Mahalanobis distance from db sample features to GMM components.
//!
//! For each db sample i and component k:
//!   `C[i, k] = (f_i - μ_k)^T Σ_k^{-1} (f_i - μ_k)`
//!
//! We avoid forming `Σ_k^{-1}` explicitly. For each component we Cholesky-
//! decompose `Σ_k = L L^T` (pure Rust — no BLAS dependency), then for each
//! sample solve `L z = (f_i - μ_k)` by forward substitution and compute
//! `||z||^2`. Numerically stable; cheap at our sizes (n_dims ≤ ~31).

use ndarray::Array2;

use crate::error::{Error, Result};
use crate::io::query::FittedGmm;

/// Build the (n_samples × n_components) Mahalanobis cost matrix.
pub fn mahalanobis(features: &Array2<f32>, gmm: &FittedGmm) -> Result<Array2<f32>> {
    let (n, d) = features.dim();
    let k = gmm.n_components;

    if gmm.weights.len() != k || gmm.means.len() != k || gmm.covariances.len() != k {
        return Err(Error::Schema(format!(
            "GMM has inconsistent component counts: weights={}, means={}, covs={}",
            gmm.weights.len(),
            gmm.means.len(),
            gmm.covariances.len()
        )));
    }
    if let Some(first_mean) = gmm.means.first()
        && first_mean.len() != d
    {
        return Err(Error::Schema(format!(
            "GMM means have dim {} but features have dim {d}",
            first_mean.len()
        )));
    }

    let chols: Vec<CholeskyL> = gmm
        .covariances
        .iter()
        .enumerate()
        .map(|(c, cov)| {
            build_chol(cov, d).map_err(|e| Error::Schema(format!("component {c}: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut cost = Array2::<f32>::zeros((n, k));
    let mut diff = vec![0.0_f64; d];
    let mut z = vec![0.0_f64; d];
    for comp_idx in 0..k {
        let mu = &gmm.means[comp_idx];
        let l = &chols[comp_idx];
        for i in 0..n {
            for j in 0..d {
                diff[j] = features[(i, j)] as f64 - mu[j];
            }
            forward_solve(l, &diff, &mut z);
            let s: f64 = z.iter().map(|&x| x * x).sum();
            cost[(i, comp_idx)] = s as f32;
        }
    }

    Ok(cost)
}

/// Lower-triangular Cholesky factor stored in row-major dense Vec.
struct CholeskyL {
    data: Vec<f64>,
    n: usize,
}

impl CholeskyL {
    #[inline]
    fn at(&self, row: usize, col: usize) -> f64 {
        debug_assert!(col <= row);
        self.data[row * self.n + col]
    }
}

fn build_chol(cov: &[Vec<f64>], n: usize) -> std::result::Result<CholeskyL, String> {
    if cov.len() != n {
        return Err(format!("cov has {} rows, expected {n}", cov.len()));
    }
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        if cov[i].len() != n {
            return Err(format!(
                "cov row {i} has {} cols, expected {n}",
                cov[i].len()
            ));
        }
        for j in 0..=i {
            let mut sum = cov[i][j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "covariance not positive-definite at index {i} (pivot {sum})"
                    ));
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(CholeskyL { data: l, n })
}

/// Solve `L z = b` in place into `z`.
fn forward_solve(l: &CholeskyL, b: &[f64], z: &mut [f64]) {
    let n = l.n;
    for i in 0..n {
        let mut sum = b[i];
        for (j, &zj) in z.iter().enumerate().take(i) {
            sum -= l.at(i, j) * zj;
        }
        z[i] = sum / l.at(i, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_covariance_is_squared_euclidean() {
        let gmm = FittedGmm {
            n_components: 1,
            weights: vec![1.0],
            means: vec![vec![0.0, 0.0]],
            covariances: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
        };
        let features = array![[3.0_f32, 4.0], [1.0, 0.0], [0.0, 0.0]];
        let cost = mahalanobis(&features, &gmm).unwrap();
        approx::assert_abs_diff_eq!(cost[(0, 0)], 25.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(cost[(1, 0)], 1.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(cost[(2, 0)], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn diagonal_covariance_weights_dimensions() {
        let gmm = FittedGmm {
            n_components: 1,
            weights: vec![1.0],
            means: vec![vec![0.0, 0.0]],
            covariances: vec![vec![vec![4.0, 0.0], vec![0.0, 1.0]]],
        };
        let features = array![[2.0_f32, 1.0], [4.0, 0.0]];
        let cost = mahalanobis(&features, &gmm).unwrap();
        // (2,1)^T diag(1/4, 1) (2,1) = 1 + 1
        approx::assert_abs_diff_eq!(cost[(0, 0)], 2.0, epsilon = 1e-5);
        // (4,0)^T diag(1/4, 1) (4,0) = 4
        approx::assert_abs_diff_eq!(cost[(1, 0)], 4.0, epsilon = 1e-5);
    }

    #[test]
    fn full_covariance_correlated() {
        // Σ = [[2,1],[1,2]] → Σ^{-1} = (1/3)[[2,-1],[-1,2]]
        // For x = (1, 0): x^T Σ^{-1} x = 2/3
        let gmm = FittedGmm {
            n_components: 1,
            weights: vec![1.0],
            means: vec![vec![0.0, 0.0]],
            covariances: vec![vec![vec![2.0, 1.0], vec![1.0, 2.0]]],
        };
        let features = array![[1.0_f32, 0.0]];
        let cost = mahalanobis(&features, &gmm).unwrap();
        approx::assert_abs_diff_eq!(cost[(0, 0)], 2.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn multiple_components_independent() {
        // Two components with distinct means; verify per-column distances.
        let gmm = FittedGmm {
            n_components: 2,
            weights: vec![0.5, 0.5],
            means: vec![vec![0.0, 0.0], vec![10.0, 10.0]],
            covariances: vec![
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            ],
        };
        let features = array![[0.0_f32, 0.0], [10.0, 10.0]];
        let cost = mahalanobis(&features, &gmm).unwrap();
        approx::assert_abs_diff_eq!(cost[(0, 0)], 0.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(cost[(0, 1)], 200.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(cost[(1, 0)], 200.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(cost[(1, 1)], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn rejects_non_psd_covariance() {
        let gmm = FittedGmm {
            n_components: 1,
            weights: vec![1.0],
            means: vec![vec![0.0]],
            covariances: vec![vec![vec![-1.0]]],
        };
        let features = array![[1.0_f32]];
        assert!(mahalanobis(&features, &gmm).is_err());
    }

    #[test]
    fn rejects_dim_mismatch() {
        let gmm = FittedGmm {
            n_components: 1,
            weights: vec![1.0],
            means: vec![vec![0.0, 0.0, 0.0]],
            covariances: vec![vec![vec![1.0; 3]; 3]],
        };
        let features = array![[1.0_f32, 0.0]];
        assert!(mahalanobis(&features, &gmm).is_err());
    }
}
