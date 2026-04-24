//! Candidate-pool selection from the Sinkhorn transport plan.
//!
//! For each db sample i, the row sum of the transport plan
//! `r_i = Σ_k P[i, k]` measures how much mass the GMM "sent" to that sample —
//! a clean relevance score. We rank by `r_i` (descending), allocate per-sex
//! pool sizes proportional to the query's per-sex counts (when sex-split),
//! and emit a tagged candidate set that downstream refinement uses.

use ndarray::Array2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stratum {
    All,
    Female,
    Male,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    /// Index into the original `DbPack::samples` table.
    pub db_idx: usize,
    /// Relevance score = row sum of P at this sample's row in its stratum.
    pub score: f32,
    pub stratum: Stratum,
}

/// Per-sample relevance scores (row sums of the transport plan).
pub fn relevance_scores(plan: &Array2<f32>) -> Vec<f32> {
    plan.rows().into_iter().map(|row| row.sum()).collect()
}

/// Allocate a pool of `pool_size` candidates between female/male strata in
/// proportion to the query's per-sex counts. Sums to `pool_size` (rounding
/// any remainder to the male side to keep totals exact).
pub fn allocate_pool_sex(
    pool_size: usize,
    n_query_female: u32,
    n_query_male: u32,
) -> (usize, usize) {
    let total = (n_query_female as u64) + (n_query_male as u64);
    if total == 0 || pool_size == 0 {
        return (0, 0);
    }
    let pool_female = ((pool_size as u128 * n_query_female as u128 + total as u128 / 2)
        / total as u128) as usize;
    let pool_female = pool_female.min(pool_size);
    let pool_male = pool_size - pool_female;
    (pool_female, pool_male)
}

/// Select the top `pool_size` indices by descending score.
///
/// `relevances[i]` is the score for the i-th sample in this stratum, and
/// `index_map[i]` maps that to the global db_idx.
pub fn select_top(
    relevances: &[f32],
    index_map: &[usize],
    pool_size: usize,
    stratum: Stratum,
) -> Vec<Candidate> {
    assert_eq!(
        relevances.len(),
        index_map.len(),
        "relevance and index_map length must match"
    );
    let n = relevances.len();
    let take = pool_size.min(n);
    if take == 0 {
        return Vec::new();
    }

    // Pair (local_idx, score), sort by descending score, take top `take`.
    let mut indexed: Vec<(usize, f32)> = relevances
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(take);

    indexed
        .into_iter()
        .map(|(local_i, score)| Candidate {
            db_idx: index_map[local_i],
            score,
            stratum,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn relevance_is_row_sum() {
        let plan = array![
            [0.1_f32, 0.2, 0.3],
            [0.0, 0.5, 0.0],
            [0.4, 0.4, 0.4]
        ];
        let r = relevance_scores(&plan);
        assert_eq!(r.len(), 3);
        approx::assert_abs_diff_eq!(r[0], 0.6, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(r[1], 0.5, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(r[2], 1.2, epsilon = 1e-6);
    }

    #[test]
    fn allocate_pool_sex_preserves_total() {
        for (pool, f, m) in [(100, 60, 40), (500, 250, 250), (7, 3, 4), (100, 1, 99)] {
            let (pf, pm) = allocate_pool_sex(pool, f, m);
            assert_eq!(pf + pm, pool, "pool={pool}, f={f}, m={m}");
        }
    }

    #[test]
    fn allocate_pool_sex_proportional() {
        let (pf, pm) = allocate_pool_sex(1000, 700, 300);
        assert!((pf as i32 - 700).abs() <= 1);
        assert!((pm as i32 - 300).abs() <= 1);
    }

    #[test]
    fn allocate_pool_sex_zero_inputs() {
        assert_eq!(allocate_pool_sex(0, 100, 100), (0, 0));
        assert_eq!(allocate_pool_sex(100, 0, 0), (0, 0));
    }

    #[test]
    fn select_top_picks_highest_scores() {
        let relevances = vec![0.1_f32, 0.9, 0.5, 0.2, 0.7];
        let index_map = vec![10, 20, 30, 40, 50];
        let candidates = select_top(&relevances, &index_map, 3, Stratum::All);
        assert_eq!(candidates.len(), 3);
        // Expected order: idx 20 (0.9), idx 50 (0.7), idx 30 (0.5)
        assert_eq!(candidates[0].db_idx, 20);
        assert_eq!(candidates[1].db_idx, 50);
        assert_eq!(candidates[2].db_idx, 30);
        approx::assert_abs_diff_eq!(candidates[0].score, 0.9, epsilon = 1e-6);
    }

    #[test]
    fn select_top_clamps_to_available() {
        let relevances = vec![0.5_f32, 0.7];
        let index_map = vec![1, 2];
        let candidates = select_top(&relevances, &index_map, 10, Stratum::Female);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].stratum, Stratum::Female);
    }

    #[test]
    fn select_top_empty_pool_returns_empty() {
        let candidates = select_top(&[1.0_f32], &[0], 0, Stratum::All);
        assert!(candidates.is_empty());
    }
}
