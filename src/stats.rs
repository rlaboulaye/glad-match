//! Per-site χ² for case/control allele counts and genomic control λ.
//!
//! For a 2×2 allele table:
//! ```text
//!             | Alt | Ref | Total
//!   Query     |  a  |  b  | a + b
//!   Control   |  c  |  d  | c + d
//!   Total     | a+c | b+d |   N
//! ```
//! `χ² = N · (ad − bc)² / ((a+b)(c+d)(a+c)(b+d))` (1 df). Returns 0 for any
//! degenerate or monomorphic table.
//!
//! `λ = median(χ²) / 0.4549…`, where 0.4549 is the median of the χ²(1)
//! distribution. We target `λ → 1` (not `λ → 0`) — over-correction is just as
//! bad as under-correction.

/// Median of the χ²(df=1) distribution. λ_GC = median(χ²) / this constant.
pub const MEDIAN_CHI2_DF1: f64 = 0.454_936_423_119_572_7;

pub fn chi_square(ac_query: u32, an_query: u32, ac_ctrl: u32, an_ctrl: u32) -> f64 {
    if an_query == 0 || an_ctrl == 0 {
        return 0.0;
    }
    if ac_query > an_query || ac_ctrl > an_ctrl {
        // Defensive: ill-formed counts contribute no information.
        return 0.0;
    }
    let a = ac_query as f64;
    let b = (an_query - ac_query) as f64;
    let c = ac_ctrl as f64;
    let d = (an_ctrl - ac_ctrl) as f64;
    let n = a + b + c + d;
    let n_alt = a + c;
    let n_ref = b + d;
    if n_alt == 0.0 || n_ref == 0.0 {
        return 0.0; // monomorphic across both groups → no information
    }
    let n_q = a + b;
    let n_c = c + d;
    let denom = n_q * n_c * n_alt * n_ref;
    if denom == 0.0 {
        return 0.0;
    }
    let num = (a * d - b * c).powi(2);
    n * num / denom
}

/// Genomic control λ from a slice of per-site χ² values.
pub fn lambda(chi2: &[f64]) -> f64 {
    if chi2.is_empty() {
        return 0.0;
    }
    let mut buf: Vec<f64> = chi2.to_vec();
    let mid = buf.len() / 2;
    buf.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let med = if buf.len() % 2 == 1 {
        buf[mid]
    } else {
        // After select_nth at mid, buf[..mid] are the smaller half.
        // The median (even length) is the mean of the (mid-1)-th and mid-th
        // smallest. The (mid-1)-th smallest is the max of buf[..mid].
        let lower_max = buf[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lower_max + buf[mid]) / 2.0
    };
    med / MEDIAN_CHI2_DF1
}

/// Refinement objective: `(log λ)²`. Returns +∞ for non-positive or non-finite λ.
pub fn log_lambda_sq(lambda_value: f64) -> f64 {
    if lambda_value <= 0.0 || !lambda_value.is_finite() {
        return f64::INFINITY;
    }
    let l = lambda_value.ln();
    l * l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_square_zero_for_identical_frequencies() {
        let chi = chi_square(50, 100, 50, 100);
        approx::assert_abs_diff_eq!(chi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn chi_square_known_value() {
        // a=80, b=20, c=40, d=60, N=200.
        // χ² = 200 * (80*60 - 20*40)^2 / (100 * 100 * 120 * 80) = 200/6 ≈ 33.33
        let chi = chi_square(80, 100, 40, 100);
        approx::assert_abs_diff_eq!(chi, 100.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn chi_square_monomorphic_returns_zero() {
        approx::assert_abs_diff_eq!(chi_square(0, 100, 0, 100), 0.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(chi_square(100, 100, 100, 100), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn chi_square_zero_n_returns_zero() {
        assert_eq!(chi_square(0, 0, 50, 100), 0.0);
        assert_eq!(chi_square(50, 100, 0, 0), 0.0);
    }

    #[test]
    fn chi_square_invalid_counts_returns_zero() {
        assert_eq!(chi_square(150, 100, 50, 100), 0.0);
    }

    #[test]
    fn lambda_unit_when_median_matches_null() {
        let chis = vec![MEDIAN_CHI2_DF1; 1000];
        approx::assert_abs_diff_eq!(lambda(&chis), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn lambda_inflated_when_median_one() {
        let chis = vec![1.0_f64; 1000];
        approx::assert_abs_diff_eq!(lambda(&chis), 1.0 / MEDIAN_CHI2_DF1, epsilon = 1e-10);
    }

    #[test]
    fn lambda_deflated_when_median_low() {
        let chis = vec![0.1_f64; 100];
        assert!(lambda(&chis) < 1.0);
    }

    #[test]
    fn lambda_handles_even_length() {
        let chis = vec![0.0_f64, MEDIAN_CHI2_DF1 * 2.0];
        // Median = (0 + 2*M) / 2 = M → λ = 1
        approx::assert_abs_diff_eq!(lambda(&chis), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn lambda_empty_returns_zero() {
        assert_eq!(lambda(&[]), 0.0);
    }

    #[test]
    fn log_lambda_sq_zero_at_one() {
        assert_eq!(log_lambda_sq(1.0), 0.0);
    }

    #[test]
    fn log_lambda_sq_symmetric_around_one() {
        let a = log_lambda_sq(2.0);
        let b = log_lambda_sq(0.5);
        approx::assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }

    #[test]
    fn log_lambda_sq_handles_zero_and_negative() {
        assert!(log_lambda_sq(0.0).is_infinite());
        assert!(log_lambda_sq(-1.0).is_infinite());
        assert!(log_lambda_sq(f64::NAN).is_infinite());
    }
}
