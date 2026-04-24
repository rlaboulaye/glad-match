//! Loader for the preprocessed `db_pack/` directory.
//!
//! Layout:
//! - `manifest.json` — versioned metadata (reference build, shapes, age stats).
//! - `samples.parquet` — columns: `sample_id` (str), `sex` (u8: 0=F,1=M),
//!   `age` (f32), `population` (str), `pc0`..`pc{n_pcs-1}` (f32 each).
//! - `sites.parquet` — ALL single-allelic db sites. Columns: `chrom` (str),
//!   `pos` (u64), `ref` (str), `alt` (str), `in_pruned` (bool). Row order =
//!   `site_idx` in `0..n_sites`.
//! - `geno_dense.parquet` — site-major dense dosages: one row per site, one
//!   `uint8` column per sample (`sample_0`..`sample_{n_samples-1}`). Used by
//!   the output writer for full-coverage per-site control counts. Loaded
//!   on-demand with column projection so peak memory stays ~`n_sites × n_controls`
//!   instead of `n_sites × n_samples`.
//! - `geno_pruned.bin` — sample-major sparse payload restricted to the
//!   pruned-set rows of `sites.parquet`: a sequence of 5-byte records
//!   `(u32 site_idx LE, u8 dosage)`, dosage ∈ {1, 2, 255}. `site_idx` indexes
//!   `sites.parquet` (not a pruned-only sub-index); 255 encodes a missing call.
//! - `geno_pruned.off` — `u64 LE` offsets into `geno_pruned.bin`, length
//!   `n_samples + 1`; sample `i`'s records live in `[offsets[i], offsets[i+1])`.
//!
//! PCA is stored as per-PC columns (`pc0`..`pc{n_pcs-1}`) rather than a
//! fixed-size-list. Same storage cost after parquet encoding, simpler to
//! produce/consume on both sides.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
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
    pub n_sites_pruned: usize,
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
    /// True if this site is in the LD-pruned subset. Refinement only joins
    /// against pruned sites; the output writer ignores this flag and emits
    /// every site.
    pub in_pruned: Vec<bool>,
}

/// Sample-major sparse non-ref genotype store over the pruned site set.
pub struct GenoPruned {
    payload: Mmap,
    offsets: Vec<u64>,
}

pub const DOSAGE_MISSING: u8 = 255;

impl GenoPruned {
    pub fn n_samples(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Iterate `(site_idx, dosage)` for the given sample. Dosage is 1 or 2 for
    /// present non-ref calls, or [`DOSAGE_MISSING`] for missing calls.
    pub fn non_ref_sites(&self, sample_idx: usize) -> GenoIter<'_> {
        let start = self.offsets[sample_idx] as usize;
        let end = self.offsets[sample_idx + 1] as usize;
        GenoIter {
            slice: &self.payload[start..end],
        }
    }
}

pub struct GenoIter<'a> {
    slice: &'a [u8],
}

impl Iterator for GenoIter<'_> {
    type Item = (u32, u8);
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.len() < 5 {
            return None;
        }
        let site = u32::from_le_bytes(self.slice[..4].try_into().ok()?);
        let dosage = self.slice[4];
        self.slice = &self.slice[5..];
        Some((site, dosage))
    }
}

pub struct DbPack {
    /// Source directory; needed to lazily project columns out of
    /// `geno_dense.parquet` at output time.
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub samples: DbSamples,
    pub sites: Sites,
    pub geno: GenoPruned,
}

pub fn load<P: AsRef<Path>>(dir: P) -> Result<DbPack> {
    let dir = dir.as_ref().to_path_buf();
    let manifest = load_manifest(&dir)?;
    let samples = load_samples(&dir, manifest.n_pcs)?;
    let sites = load_sites(&dir)?;
    let geno = load_geno(&dir)?;

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
    let in_pruned_count = sites.in_pruned.iter().filter(|&&p| p).count();
    if in_pruned_count != manifest.n_sites_pruned {
        return Err(Error::Schema(format!(
            "manifest.n_sites_pruned={} but sites.in_pruned has {} true entries",
            manifest.n_sites_pruned, in_pruned_count
        )));
    }
    if geno.n_samples() != manifest.n_samples {
        return Err(Error::Schema(format!(
            "manifest.n_samples={} but geno_pruned.off encodes {} samples",
            manifest.n_samples,
            geno.n_samples()
        )));
    }

    Ok(DbPack {
        dir,
        manifest,
        samples,
        sites,
        geno,
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
        in_pruned: bool_column(&df, "in_pruned")?,
    })
}

/// Read the dense geno table for the given sample subset, projecting only
/// those `sample_{idx}` columns out of parquet (so memory is
/// `n_sites × selected.len()` rather than `n_sites × n_samples`).
///
/// Returned shape: `(n_sites, selected.len())`. Dosages: 0/1/2 for present
/// genotypes, [`DOSAGE_MISSING`] for missing.
pub fn load_geno_dense_cols(dir: &Path, selected: &[usize]) -> Result<Array2<u8>> {
    let path = dir.join("geno_dense.parquet");
    let file = File::open(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let col_names: Vec<String> = selected.iter().map(|i| format!("sample_{i}")).collect();
    let df = ParquetReader::new(file)
        .with_columns(Some(col_names.clone()))
        .finish()?;

    let n_sites = df.height();
    let m = selected.len();
    let mut arr = Array2::<u8>::zeros((n_sites, m));
    for (j, name) in col_names.iter().enumerate() {
        let col = u8_column(&df, name)?;
        if col.len() != n_sites {
            return Err(Error::Schema(format!(
                "geno_dense column {name} has {} rows, expected {n_sites}",
                col.len()
            )));
        }
        for (s, v) in col.into_iter().enumerate() {
            arr[[s, j]] = v;
        }
    }
    Ok(arr)
}

fn load_geno(dir: &Path) -> Result<GenoPruned> {
    let off_path = dir.join("geno_pruned.off");
    let mut off_file = File::open(&off_path).map_err(|source| Error::Io {
        path: off_path.clone(),
        source,
    })?;
    let mut raw = Vec::new();
    off_file
        .read_to_end(&mut raw)
        .map_err(|source| Error::Io {
            path: off_path.clone(),
            source,
        })?;
    if raw.len() % 8 != 0 {
        return Err(Error::Schema(format!(
            "geno_pruned.off size {} is not a multiple of 8",
            raw.len()
        )));
    }
    let offsets: Vec<u64> = raw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if offsets.is_empty() {
        return Err(Error::Schema("empty geno_pruned.off".into()));
    }

    let payload_path = dir.join("geno_pruned.bin");
    let payload_file = File::open(&payload_path).map_err(|source| Error::Io {
        path: payload_path.clone(),
        source,
    })?;
    let payload = unsafe { Mmap::map(&payload_file) }.map_err(|source| Error::Io {
        path: payload_path.clone(),
        source,
    })?;

    let expected = *offsets.last().unwrap() as usize;
    if payload.len() < expected {
        return Err(Error::Schema(format!(
            "geno_pruned.bin size {} is less than last offset {}",
            payload.len(),
            expected
        )));
    }

    Ok(GenoPruned { payload, offsets })
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
    use std::io::Write;

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
    /// sites are flagged `in_pruned = true`.
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
            "n_sites_pruned": n_sites,
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
        let in_pruned: Vec<bool> = vec![true; n_sites];
        let mut sdf = DataFrame::new(
            n_sites,
            vec![
                Column::new("chrom".into(), site_chrom_refs.as_slice()),
                Column::new("pos".into(), site_pos_i64.as_slice()),
                Column::new("ref".into(), site_ref_refs.as_slice()),
                Column::new("alt".into(), site_alt_refs.as_slice()),
                Column::new("in_pruned".into(), in_pruned.as_slice()),
            ],
        )
        .unwrap();
        let out = File::create(dir.join("sites.parquet")).unwrap();
        ParquetWriter::new(out).finish(&mut sdf).unwrap();

        // geno_dense.parquet: one row per site, one column per sample.
        // Match the fixture's sparse pattern: sample i has dosage 1 at site
        // (i % n_sites) and dosage 2 at site ((i+1) % n_sites), 0 elsewhere.
        // Polars' default features don't include UInt8, so write as Int32 —
        // u8_column casts back on read.
        let mut dense_cols: Vec<Vec<i32>> = vec![vec![0; n_sites]; n_samples];
        for (i, row) in dense_cols.iter_mut().enumerate() {
            let s1 = i % n_sites;
            let s2 = (i + 1) % n_sites;
            row[s1] = 1;
            row[s2] = 2;
        }
        let mut dense_columns: Vec<Column> = Vec::with_capacity(n_samples);
        for (i, col) in dense_cols.iter().enumerate() {
            dense_columns.push(Column::new(
                format!("sample_{i}").into(),
                col.as_slice(),
            ));
        }
        let mut ddf = DataFrame::new(n_sites, dense_columns).unwrap();
        let out = File::create(dir.join("geno_dense.parquet")).unwrap();
        ParquetWriter::new(out).finish(&mut ddf).unwrap();

        // geno_pruned.bin: site_idx values index into sites.parquet (== full
        // site indexing since fixture has no non-pruned sites).
        let mut offsets: Vec<u64> = vec![0];
        let mut payload: Vec<u8> = Vec::new();
        for i in 0..n_samples {
            let s1 = (i % n_sites) as u32;
            let s2 = ((i + 1) % n_sites) as u32;
            payload.extend_from_slice(&s1.to_le_bytes());
            payload.push(1);
            payload.extend_from_slice(&s2.to_le_bytes());
            payload.push(2);
            offsets.push(payload.len() as u64);
        }
        std::fs::write(dir.join("geno_pruned.bin"), &payload).unwrap();
        let mut off_file = File::create(dir.join("geno_pruned.off")).unwrap();
        for &o in &offsets {
            off_file.write_all(&o.to_le_bytes()).unwrap();
        }
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
        assert_eq!(pack.sites.chrom.len(), 4);
        assert!(pack.sites.in_pruned.iter().all(|&p| p));
        assert_eq!(pack.geno.n_samples(), 5);

        // pca[i, j] = (i * n_pcs + j) * 0.01
        approx::assert_abs_diff_eq!(pack.samples.pca[(2, 1)], 0.07, epsilon = 1e-6);

        // Sample 2 → [(2, 1), (3, 2)] in full-site indexing (== pruned-only
        // indexing in the fixture since all sites are pruned).
        let sites: Vec<_> = pack.geno.non_ref_sites(2).collect();
        assert_eq!(sites, vec![(2, 1), (3, 2)]);
    }

    #[test]
    fn rejects_mismatched_manifest() {
        let tmp = tempdir().unwrap();
        fixture::build(tmp.path(), 5, 3, 4);
        let bad = serde_json::json!({
            "version": "1.0", "reference_build": "GRCh38",
            "n_samples": 999, "n_pcs": 3, "n_sites": 4, "n_sites_pruned": 4,
            "age_mean": 50.0, "age_sd": 10.0
        });
        std::fs::write(tmp.path().join("manifest.json"), bad.to_string()).unwrap();
        assert!(load(tmp.path()).is_err());
    }
}
