//! Parser for glad-prep's `.glad.gz` output.
//!
//! Mirrors the schema defined in glad-prep/src/output.rs and glad-prep/src/gmm.rs.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use flate2::read::GzDecoder;
use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Deserialize)]
pub struct Query {
    pub version: String,
    pub reference_build: String,
    pub n_samples: usize,
    pub n_snps_attempted: usize,
    pub n_snps_found: usize,

    /// Added in glad-prep when mode is sex_and_age or sex_only.
    /// Optional here so we tolerate older fixtures; downstream code checks
    /// presence when sex-split is required.
    #[serde(default)]
    pub per_sex_counts: Option<PerSexCounts>,

    pub counts: Vec<SnpCount>,
    pub distributions: Distributions,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct PerSexCounts {
    pub female: u32,
    pub male: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SnpCount {
    pub chrom: String,
    pub pos: u64,
    pub effect_allele: String,
    pub other_allele: String,
    pub alt_count: u32,
    pub n_alleles: u32,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    SexAndAge,
    SexOnly,
    AgeOnly,
    None,
}

impl Mode {
    pub fn has_sex(self) -> bool {
        matches!(self, Mode::SexAndAge | Mode::SexOnly)
    }

    pub fn has_age(self) -> bool {
        matches!(self, Mode::SexAndAge | Mode::AgeOnly)
    }

    /// snake_case label, matching glad-prep's wire format.
    pub fn label(self) -> &'static str {
        match self {
            Mode::SexAndAge => "sex_and_age",
            Mode::SexOnly => "sex_only",
            Mode::AgeOnly => "age_only",
            Mode::None => "none",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Distributions {
    pub mode: Mode,
    pub n_dims: usize,
    #[serde(default)]
    pub all: Option<FittedGmm>,
    #[serde(default)]
    pub female: Option<FittedGmm>,
    #[serde(default)]
    pub male: Option<FittedGmm>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FittedGmm {
    pub n_components: usize,
    pub weights: Vec<f64>,
    /// Shape: n_components × n_dims
    pub means: Vec<Vec<f64>>,
    /// Shape: n_components × n_dims × n_dims (full covariances)
    pub covariances: Vec<Vec<Vec<f64>>>,
}

/// Read a gzipped JSON query file produced by glad-prep.
pub fn read<P: AsRef<Path>>(path: P) -> Result<Query> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(GzDecoder::new(file));
    let query: Query = serde_json::from_reader(reader)?;
    validate(&query)?;
    Ok(query)
}

fn validate(q: &Query) -> Result<()> {
    match q.distributions.mode {
        Mode::SexAndAge | Mode::SexOnly => {
            if q.distributions.female.is_none() || q.distributions.male.is_none() {
                return Err(Error::Schema(
                    "sex-split mode requires both female and male GMMs".into(),
                ));
            }
        }
        Mode::AgeOnly | Mode::None => {
            if q.distributions.all.is_none() {
                return Err(Error::Schema(
                    "non-sex-split mode requires the `all` GMM".into(),
                ));
            }
        }
    }

    let expected_dims_min = 1;
    if q.distributions.n_dims < expected_dims_min {
        return Err(Error::Schema(format!(
            "n_dims must be >= {expected_dims_min}, got {}",
            q.distributions.n_dims
        )));
    }

    // Spot-check one GMM's dimensional consistency.
    let sample = q
        .distributions
        .all
        .as_ref()
        .or(q.distributions.female.as_ref())
        .or(q.distributions.male.as_ref())
        .expect("at least one GMM present (checked above)");
    if let Some(first_mean) = sample.means.first()
        && first_mean.len() != q.distributions.n_dims
    {
        return Err(Error::Schema(format!(
            "GMM mean dim {} does not match distributions.n_dims {}",
            first_mean.len(),
            q.distributions.n_dims
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;

    fn roundtrip_write(json: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let mut gz = GzEncoder::new(f.as_file_mut(), Compression::default());
        gz.write_all(json.as_bytes()).unwrap();
        gz.finish().unwrap();
        f
    }

    #[test]
    fn parses_mode_none_query() {
        let json = r#"{
            "version": "1.0",
            "reference_build": "GRCh38",
            "n_samples": 100,
            "n_snps_attempted": 10,
            "n_snps_found": 9,
            "counts": [
                {"chrom": "1", "pos": 18, "effect_allele": "A", "other_allele": "T", "alt_count": 12, "n_alleles": 200}
            ],
            "distributions": {
                "mode": "none",
                "n_dims": 2,
                "all": {
                    "n_components": 1,
                    "weights": [1.0],
                    "means": [[0.1, 0.2]],
                    "covariances": [[[1.0, 0.0], [0.0, 1.0]]]
                }
            }
        }"#;
        let f = roundtrip_write(json);
        let q = read(f.path()).unwrap();
        assert_eq!(q.distributions.mode, Mode::None);
        assert!(q.distributions.all.is_some());
        assert_eq!(q.counts.len(), 1);
    }

    #[test]
    fn rejects_sex_mode_without_both_gmms() {
        let json = r#"{
            "version": "1.0",
            "reference_build": "GRCh38",
            "n_samples": 100,
            "n_snps_attempted": 0,
            "n_snps_found": 0,
            "counts": [],
            "distributions": {
                "mode": "sex_only",
                "n_dims": 2,
                "female": {
                    "n_components": 1,
                    "weights": [1.0],
                    "means": [[0.0, 0.0]],
                    "covariances": [[[1.0, 0.0], [0.0, 1.0]]]
                }
            }
        }"#;
        let f = roundtrip_write(json);
        let err = read(f.path()).unwrap_err();
        matches!(err, Error::Schema(_));
    }

    #[test]
    fn parses_real_fixture() {
        // The working-dir query.glad.gz is produced by glad-prep from simulated data.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("query.glad.gz");
        if !path.exists() {
            // Skip silently in environments where the fixture isn't available.
            return;
        }
        let q = read(&path).unwrap();
        assert_eq!(q.version, "1.0");
        assert!(q.n_samples > 0);
        assert!(!q.counts.is_empty());
        // The fixture is sex_and_age, so per_sex_counts must be present
        // (regenerated after glad-prep gained the field).
        if q.distributions.mode.has_sex() {
            let psc = q
                .per_sex_counts
                .expect("regenerated query should carry per_sex_counts");
            assert_eq!(psc.female + psc.male, q.n_samples as u32);
        }
    }
}
