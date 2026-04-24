//! Loader for the preprocessed `db_pack/` directory.
//!
//! Layout:
//! - `manifest.json` — versioned metadata (reference build, shapes, age stats).
//! - `samples.parquet` — columns: `sample_id` (str), `sex` (u8: 0=F,1=M),
//!   `age` (f32), `population` (str), `pc0`..`pc{n_pcs-1}` (f32 each).
//! - `sites.parquet` — ALL single-allelic db sites. Columns: `chrom` (str),
//!   `pos` (u64), `ref` (str), `alt` (str), `ld_indep` (bool). Row order =
//!   `site_idx` in `0..n_sites`.
//! - `geno_dense.parquet` — site-major dense dosages: one row per site, one
//!   `uint8` column per sample (`sample_0`..`sample_{n_samples-1}`). Used by
//!   the output writer for full-coverage per-site control counts. Loaded
//!   on-demand with column projection so peak memory stays ~`n_sites × n_controls`
//!   instead of `n_sites × n_samples`.
//! - `geno_ld_indep.parquet` — same format as `geno_dense.parquet` but
//!   restricted to the `n_sites_ld_indep` rows where `ld_indep = true`. Used
//!   by the refinement stage with column projection onto the candidate pool
//!   (~`n_ld_indep × n_candidates` bytes loaded at refinement start).
//!
//! PCA is stored as per-PC columns (`pc0`..`pc{n_pcs-1}`) rather than a
//! fixed-size-list. Same storage cost after parquet encoding, simpler to
//! produce/consume on both sides.

use std::fs::File;
use std::path::{Path, PathBuf};

use ndarray::Array2;
use polars::prelude::*;
use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Deserialize, Clone)]
pub struct Manifest {
    pub version: String,
    pub reference_build: String,
    pub n_samples: usize,
    pub n_pcs: usize,
    pub n_sites: usize,
    pub n_sites_ld_indep: usize,
    pub age_mean: f64,
    pub age_sd: f64,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug)]
pub struct DbSamples {
    pub sample_ids: Vec<String>,
    pub sex: Vec<u8>,
    pub age: Vec<f32>,
    pub population: Vec<String>,
    /// Shape: (n_samples, n_pcs).
    pub pca: Array2<f32>,
}

#[derive(Debug)]
pub struct Sites {
    pub chrom: Vec<String>,
    pub pos: Vec<u64>,
    pub ref_allele: Vec<String>,
    pub alt_allele: Vec<String>,
    /// True if this site is in the LD-independent (LD-pruning survivors) subset
    /// used for refinement. The output writer ignores this flag and emits every site.
    pub ld_indep: Vec<bool>,
}

pub const DOSAGE_MISSING: u8 = 255;

pub struct DbPack {
    /// Source directory; needed to lazily project columns out of parquet files
    /// at output time and refinement start.
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub samples: DbSamples,
    pub sites: Sites,
}

pub fn load<P: AsRef<Path>>(dir: P) -> Result<DbPack> {
    let dir = dir.as_ref().to_path_buf();
    let manifest = load_manifest(&dir)?;
    let samples = load_samples(&dir, manifest.n_pcs)?;
    let sites = load_sites(&dir)?;

    if samples.sample_ids.len() != manifest.n_samples {
        return Err(Error::Schema(format!(
            "manifest.n_samples={} but samples.parquet has {} rows",
            manifest.n_samples,
            samples.sample_ids.len()
        )));
    }
    if sites.chrom.len() != manifest.n_sites {
        return Err(Error::Schema(format!(
            "manifest.n_sites={} but sites.parquet has {} rows",
            manifest.n_sites,
            sites.chrom.len()
        )));
    }
    let ld_indep_count = sites.ld_indep.iter().filter(|&&p| p).count();
    if ld_indep_count != manifest.n_sites_ld_indep {
        return Err(Error::Schema(format!(
            "manifest.n_sites_ld_indep={} but sites.ld_indep has {} true entries",
            manifest.n_sites_ld_indep, ld_indep_count
        )));
    }

    Ok(DbPack {
        dir,
        manifest,
        samples,
        sites,
    })
}

fn load_manifest(dir: &Path) -> Result<Manifest> {
    let path = dir.join("manifest.json");
    let file = File::open(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let manifest: Manifest = serde_json::from_reader(file)?;
    Ok(manifest)
}

fn load_samples(dir: &Path, n_pcs: usize) -> Result<DbSamples> {
    let path = dir.join("samples.parquet");
    let file = File::open(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let df = ParquetReader::new(file).finish()?;

    let sample_ids = str_column(&df, "sample_id")?;
    let sex = u8_column(&df, "sex")?;
    let age = f32_column(&df, "age")?;
    let population = str_column(&df, "population")?;

    let n_samples = sample_ids.len();
    let mut pca = Array2::<f32>::zeros((n_samples, n_pcs));
    for j in 0..n_pcs {
        let col_name = format!("pc{j}");
        let col_values = f32_column(&df, &col_name)?;
        if col_values.len() != n_samples {
            return Err(Error::Schema(format!(
                "column {col_name} has {} rows, expected {n_samples}",
                col_values.len()
            )));
        }
        for (i, v) in col_values.into_iter().enumerate() {
            pca[(i, j)] = v;
        }
    }

    Ok(DbSamples {
        sample_ids,
        sex,
        age,
        population,
        pca,
    })
}

fn load_sites(dir: &Path) -> Result<Sites> {
    let path = dir.join("sites.parquet");
    let file = File::open(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let df = ParquetReader::new(file).finish()?;

    Ok(Sites {
        chrom: str_column(&df, "chrom")?,
        pos: u64_column(&df, "pos")?,
        ref_allele: str_column(&df, "ref")?,
        alt_allele: str_column(&df, "alt")?,
        ld_indep: bool_column(&df, "ld_indep")?,
    })
}

/// Project the given sample columns from `geno_dense.parquet`.
///
/// Returned shape: `(n_sites, selected.len())`. Dosages: 0/1/2 for present
/// genotypes, [`DOSAGE_MISSING`] for missing.
pub fn load_geno_dense_cols(dir: &Path, selected: &[usize]) -> Result<Array2<u8>> {
    load_geno_cols(dir, "geno_dense.parquet", selected)
}

/// Project the given sample columns from `geno_ld_indep.parquet`.
///
/// Returned shape: `(n_sites_ld_indep, selected.len())`. Dosages: 0/1/2 for
/// present genotypes, [`DOSAGE_MISSING`] for missing.
pub fn load_geno_ld_indep_cols(dir: &Path, selected: &[usize]) -> Result<Array2<u8>> {
    load_geno_cols(dir, "geno_ld_indep.parquet", selected)
}

fn load_geno_cols(dir: &Path, filename: &str, selected: &[usize]) -> Result<Array2<u8>> {
    let path = dir.join(filename);
    let file = File::open(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let col_names: Vec<String> = selected.iter().map(|i| format!("sample_{i}")).collect();
    let df = ParquetReader::new(file)
        .with_columns(Some(col_names.clone()))
        .finish()?;

    let n_rows = df.height();
    let m = selected.len();
    let mut arr = Array2::<u8>::zeros((n_rows, m));
    for (j, name) in col_names.iter().enumerate() {
        let col = u8_column(&df, name)?;
        if col.len() != n_rows {
            return Err(Error::Schema(format!(
                "{filename} column {name} has {} rows, expected {n_rows}",
                col.len()
            )));
        }
        for (s, v) in col.into_iter().enumerate() {
            arr[[s, j]] = v;
        }
    }
    Ok(arr)
}

// --- small polars helpers -------------------------------------------------

fn str_column(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    df.column(name)?
        .str()?
        .into_iter()
        .map(|o| {
            o.map(str::to_string)
                .ok_or_else(|| Error::Schema(format!("null in column {name}")))
        })
        .collect()
}

fn u8_column(df: &DataFrame, name: &str) -> Result<Vec<u8>> {
    // Polars' default feature set does not include the UInt8 dtype, so we
    // read as Int32 and narrow explicitly. Values outside 0..=255 are an
    // input-integrity error.
    df.column(name)?
        .cast(&DataType::Int32)?
        .i32()?
        .into_iter()
        .map(|o| {
            let v = o.ok_or_else(|| Error::Schema(format!("null in column {name}")))?;
            u8::try_from(v)
                .map_err(|_| Error::Schema(format!("value {v} in {name} does not fit u8")))
        })
        .collect()
}

fn u64_column(df: &DataFrame, name: &str) -> Result<Vec<u64>> {
    df.column(name)?
        .cast(&DataType::UInt64)?
        .u64()?
        .into_iter()
        .map(|o| o.ok_or_else(|| Error::Schema(format!("null in column {name}"))))
        .collect()
}

fn bool_column(df: &DataFrame, name: &str) -> Result<Vec<bool>> {
    df.column(name)?
        .cast(&DataType::Boolean)?
        .bool()?
        .into_iter()
        .map(|o| o.ok_or_else(|| Error::Schema(format!("null in column {name}"))))
        .collect()
}

fn f32_column(df: &DataFrame, name: &str) -> Result<Vec<f32>> {
    df.column(name)?
        .cast(&DataType::Float32)?
        .f32()?
        .into_iter()
        .map(|o| o.ok_or_else(|| Error::Schema(format!("null in column {name}"))))
        .collect()
}

#[cfg(test)]
pub(crate) mod fixture {
    use super::*;

    /// Build a tiny db_pack in `dir` for tests using deterministic synthetic
    /// site positions (chrom "1", pos = 100 + i*10, ref "A", alt "G").
    ///
    /// Sample i has non-ref calls at sites `(i % n_sites, 1)` and
    /// `((i+1) % n_sites, 2)`.
    pub fn build(dir: &Path, n_samples: usize, n_pcs: usize, n_sites: usize) {
        let sites: Vec<(String, u64, String, String)> = (0..n_sites)
            .map(|i| ("1".to_string(), 100 + i as u64 * 10, "A".to_string(), "G".to_string()))
            .collect();
        build_with_sites(dir, n_samples, n_pcs, &sites);
    }

    /// Like [`build`] but uses caller-provided site coordinates so tests can
    /// exercise the join against a real query's allele list. All fixture
    /// sites are flagged `ld_indep = true`.
    pub fn build_with_sites(
        dir: &Path,
        n_samples: usize,
        n_pcs: usize,
        sites: &[(String, u64, String, String)],
    ) {
        let n_sites = sites.len();
        let manifest = serde_json::json!({
            "version": "1.0",
            "reference_build": "GRCh38",
            "n_samples": n_samples,
            "n_pcs": n_pcs,
            "n_sites": n_sites,
            "n_sites_ld_indep": n_sites,
            "age_mean": 50.0_f64,
            "age_sd": 10.0_f64
        });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();

        let sample_ids: Vec<String> = (0..n_samples).map(|i| format!("s{i}")).collect();
        let sample_id_refs: Vec<&str> = sample_ids.iter().map(String::as_str).collect();
        let sex_i32: Vec<i32> = (0..n_samples).map(|i| (i % 2) as i32).collect();
        let age: Vec<f32> = (0..n_samples).map(|i| 40.0 + i as f32).collect();
        let population: Vec<String> = (0..n_samples).map(|_| "MXL".to_string()).collect();
        let population_refs: Vec<&str> = population.iter().map(String::as_str).collect();

        let mut columns: Vec<Column> = vec![
            Column::new("sample_id".into(), sample_id_refs.as_slice()),
            Column::new("sex".into(), sex_i32.as_slice()),
            Column::new("age".into(), age.as_slice()),
            Column::new("population".into(), population_refs.as_slice()),
        ];
        let pc_vecs: Vec<Vec<f32>> = (0..n_pcs)
            .map(|j| {
                (0..n_samples)
                    .map(|i| (i * n_pcs + j) as f32 * 0.01)
                    .collect()
            })
            .collect();
        for (j, col) in pc_vecs.iter().enumerate() {
            columns.push(Column::new(format!("pc{j}").into(), col.as_slice()));
        }
        let mut df = DataFrame::new(n_samples, columns).unwrap();
        let out = File::create(dir.join("samples.parquet")).unwrap();
        ParquetWriter::new(out).finish(&mut df).unwrap();

        let site_chrom_refs: Vec<&str> = sites.iter().map(|(c, _, _, _)| c.as_str()).collect();
        let site_pos_i64: Vec<i64> = sites.iter().map(|(_, p, _, _)| *p as i64).collect();
        let site_ref_refs: Vec<&str> = sites.iter().map(|(_, _, r, _)| r.as_str()).collect();
        let site_alt_refs: Vec<&str> = sites.iter().map(|(_, _, _, a)| a.as_str()).collect();
        let ld_indep: Vec<bool> = vec![true; n_sites];
        let mut sdf = DataFrame::new(
            n_sites,
            vec![
                Column::new("chrom".into(), site_chrom_refs.as_slice()),
                Column::new("pos".into(), site_pos_i64.as_slice()),
                Column::new("ref".into(), site_ref_refs.as_slice()),
                Column::new("alt".into(), site_alt_refs.as_slice()),
                Column::new("ld_indep".into(), ld_indep.as_slice()),
            ],
        )
        .unwrap();
        let out = File::create(dir.join("sites.parquet")).unwrap();
        ParquetWriter::new(out).finish(&mut sdf).unwrap();

        // Both geno_dense.parquet and geno_ld_indep.parquet use the same layout:
        // one row per site, one column per sample (sample_0..sample_{n-1}).
        // Since all fixture sites are ld_indep, both files are identical here.
        // Polars' default features don't include UInt8, so write as Int32 —
        // u8_column casts back on read.
        let geno = build_geno_matrix(n_samples, n_sites);
        write_geno_parquet(dir, "geno_dense.parquet", &geno, n_samples, n_sites);
        write_geno_parquet(dir, "geno_ld_indep.parquet", &geno, n_samples, n_sites);
    }

    fn build_geno_matrix(n_samples: usize, n_sites: usize) -> Vec<Vec<i32>> {
        // Column-indexed: dense_cols[sample_idx][site_idx]
        // Sample i has dosage 1 at site (i % n_sites) and dosage 2 at site
        // ((i+1) % n_sites), 0 elsewhere.
        let mut cols: Vec<Vec<i32>> = vec![vec![0; n_sites]; n_samples];
        for (i, row) in cols.iter_mut().enumerate() {
            let s1 = i % n_sites;
            let s2 = (i + 1) % n_sites;
            row[s1] = 1;
            row[s2] = 2;
        }
        cols
    }

    fn write_geno_parquet(
        dir: &Path,
        filename: &str,
        cols: &[Vec<i32>],
        n_samples: usize,
        n_sites: usize,
    ) {
        let mut columns: Vec<Column> = Vec::with_capacity(n_samples);
        for (i, col) in cols.iter().enumerate() {
            columns.push(Column::new(format!("sample_{i}").into(), col.as_slice()));
        }
        let mut ddf = DataFrame::new(n_sites, columns).unwrap();
        let out = File::create(dir.join(filename)).unwrap();
        ParquetWriter::new(out).finish(&mut ddf).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn loads_fixture() {
        let tmp = tempdir().unwrap();
        fixture::build(tmp.path(), 5, 3, 4);
        let pack = load(tmp.path()).unwrap();

        assert_eq!(pack.samples.sample_ids.len(), 5);
        assert_eq!(pack.samples.pca.shape(), &[5, 3]);
        assert_eq!(pack.manifest.n_sites, 4);
        assert_eq!(pack.manifest.n_sites_ld_indep, 4);
        assert_eq!(pack.sites.chrom.len(), 4);
        assert!(pack.sites.ld_indep.iter().all(|&p| p));

        // pca[i, j] = (i * n_pcs + j) * 0.01
        approx::assert_abs_diff_eq!(pack.samples.pca[(2, 1)], 0.07, epsilon = 1e-6);
    }

    #[test]
    fn geno_dense_cols_roundtrip() {
        let tmp = tempdir().unwrap();
        fixture::build(tmp.path(), 5, 3, 4);
        // Sample 2: dosage 1 at site 2, dosage 2 at site 3
        let mat = load_geno_dense_cols(tmp.path(), &[2]).unwrap();
        assert_eq!(mat.shape(), &[4, 1]);
        assert_eq!(mat[[2, 0]], 1);
        assert_eq!(mat[[3, 0]], 2);
        assert_eq!(mat[[0, 0]], 0);
    }

    #[test]
    fn geno_ld_indep_cols_roundtrip() {
        let tmp = tempdir().unwrap();
        fixture::build(tmp.path(), 5, 3, 4);
        // Same pattern as dense since all fixture sites are ld_indep
        let mat = load_geno_ld_indep_cols(tmp.path(), &[2]).unwrap();
        assert_eq!(mat.shape(), &[4, 1]);
        assert_eq!(mat[[2, 0]], 1);
        assert_eq!(mat[[3, 0]], 2);
    }

    #[test]
    fn rejects_mismatched_manifest() {
        let tmp = tempdir().unwrap();
        fixture::build(tmp.path(), 5, 3, 4);
        let bad = serde_json::json!({
            "version": "1.0", "reference_build": "GRCh38",
            "n_samples": 999, "n_pcs": 3, "n_sites": 4, "n_sites_ld_indep": 4,
            "age_mean": 50.0, "age_sd": 10.0
        });
        std::fs::write(tmp.path().join("manifest.json"), bad.to_string()).unwrap();
        assert!(load(tmp.path()).is_err());
    }
}
