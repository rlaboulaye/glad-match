//! Sinkhorn-based optimal transport between db samples and GMM components.
//!
//! Unbalanced (KL-penalized marginals) Sinkhorn in log space, via the `wass`
//! crate. Unbalanced is important here: the marginal-KL term lets the
//! algorithm "drop" mass from db samples that don't fit any GMM component
//! well, so the candidate pool isn't dragged around by outliers.
//!
//! Both source and target masses are interpreted on the **probability scale**:
//! `a` is uniform `1/N_db`, `b` is the GMM's mixture weights (already
//! probabilities). This is a deliberate choice — wass's "do not normalize"
//! guidance says don't *erase* meaningful scale, and we choose probability
//! scale up front rather than rescaling something that came in differently.
//! `eps` and `rho` are tuned against this scale.

use ndarray::{Array1, Array2};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct SinkhornParams {
    /// Entropic regularization. If <= 0, uses `median(C) / 50`.
    pub eps: f32,
    /// Marginal-KL penalty (mass-deletion strength).
    pub rho: f32,
    pub max_iter: usize,
    pub tol: f32,
}

impl Default for SinkhornParams {
    fn default() -> Self {
        Self {
            eps: 0.0,
            rho: 0.1,
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

#[derive(Debug)]
pub struct SinkhornResult {
    /// Transport plan, shape (n_samples, n_components).
    pub plan: Array2<f32>,
    pub objective: f32,
    pub iterations: usize,
    pub eps_used: f32,
}

/// Run unbalanced Sinkhorn against a precomputed cost matrix.
pub fn run(
    cost: &Array2<f32>,
    a: &Array1<f32>,
    b: &Array1<f32>,
    params: SinkhornParams,
) -> Result<SinkhornResult> {
    if cost.dim() != (a.len(), b.len()) {
        return Err(Error::Schema(format!(
            "cost shape {:?} does not match (|a|={}, |b|={})",
            cost.dim(),
            a.len(),
            b.len()
        )));
    }
    let eps = if params.eps > 0.0 {
        params.eps
    } else {
        auto_eps(cost)
    };
    let (plan, objective, iterations) = wass::unbalanced_sinkhorn_log_with_convergence(
        a,
        b,
        cost,
        eps,
        params.rho,
        params.max_iter,
        params.tol,
    )
    .map_err(|e| Error::Parse {
        what: "sinkhorn".into(),
        source: anyhow::anyhow!("{e:?}"),
    })?;
    Ok(SinkhornResult {
        plan,
        objective,
        iterations,
        eps_used: eps,
    })
}

/// Uniform source-mass vector summing to 1.
pub fn uniform_source(n: usize) -> Array1<f32> {
    Array1::from_elem(n, 1.0 / n as f32)
}

/// GMM mixture weights cast to f32.
pub fn gmm_target(weights: &[f64]) -> Array1<f32> {
    Array1::from_iter(weights.iter().map(|&w| w as f32))
}

/// Default ε = median(C) / 50, lower-bounded.
fn auto_eps(cost: &Array2<f32>) -> f32 {
    let n = cost.len();
    if n == 0 {
        return 1e-6;
    }
    let mut all: Vec<f32> = cost.iter().copied().collect();
    let mid = n / 2;
    all.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    (all[mid] / 50.0).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn diagonal_cost_concentrates_plan_on_diagonal() {
        // 3 source rows, 3 target cols; cheap on diagonal, expensive off-diag.
        let cost = array![
            [0.0_f32, 5.0, 5.0],
            [5.0, 0.0, 5.0],
            [5.0, 5.0, 0.0]
        ];
        let a = uniform_source(3);
        let b = Array1::from(vec![1.0_f32 / 3.0; 3]);
        let result = run(
            &cost,
            &a,
            &b,
            SinkhornParams {
                eps: 0.5,
                rho: 1.0,
                max_iter: 500,
                tol: 1e-6,
            },
        )
        .unwrap();
        assert_eq!(result.plan.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(
                        result.plan[(i, i)] > result.plan[(i, j)],
                        "row {i}: diagonal {} should exceed off-diagonal {} at col {j}",
                        result.plan[(i, i)],
                        result.plan[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_shape_mismatch() {
        let cost = array![[0.0_f32, 1.0], [1.0, 0.0]];
        let a = Array1::from(vec![0.5_f32, 0.5]);
        let b = Array1::from(vec![1.0_f32]);
        let err = run(&cost, &a, &b, SinkhornParams::default()).unwrap_err();
        assert!(matches!(err, Error::Schema(_)));
    }

    #[test]
    fn auto_eps_is_positive_for_typical_cost() {
        let cost = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let eps = auto_eps(&cost);
        assert!(eps > 0.0);
        assert!(eps < 1.0);
    }
}
