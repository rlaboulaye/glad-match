//! Per-site control genotype-count TSV writer (gzipped).
//!
//! Output columns:
//!   `chrom  pos  ref  alt  AC_ctrl  AN_ctrl  n_00_ctrl  n_01_ctrl  n_11_ctrl  n_miss_ctrl`
//!
//! One row per site in `sites.parquet`, in site_idx order. The output covers
//! ALL db sites (not just the LD-pruned subset — pruning is an internal
//! refinement concept). Control counts are tallied from `geno_dense.parquet`
//! with column projection onto the selected sample set, so peak memory is
//! proportional to `n_sites × n_controls` rather than `n_sites × n_samples`.
//!
//! `AC_ctrl` and `AN_ctrl` are derived from the genotype counts:
//!   `AC = n_01 + 2 * n_11`
//!   `AN = 2 * (n_00 + n_01 + n_11)`  (missing calls excluded from AN)

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::error::{Error, Result};
use crate::io::db_pack::{self, DOSAGE_MISSING, DbPack};

/// Write the control-counts TSV (gzipped) to `path`.
///
/// Loads only the `selected` samples' columns from `geno_dense.parquet`
/// (column projection); does not read the full dense matrix.
pub fn write<P: AsRef<Path>>(path: P, pack: &DbPack, selected: &[usize]) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let gz = GzEncoder::new(BufWriter::new(file), Compression::default());
    let mut w = BufWriter::new(gz);

    writeln!(
        w,
        "chrom\tpos\tref\talt\tAC_ctrl\tAN_ctrl\tn_00_ctrl\tn_01_ctrl\tn_11_ctrl\tn_miss_ctrl"
    )
    .map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let n_sites = pack.sites.chrom.len();

    // Load projected dense dosages: shape (n_sites, selected.len()).
    // Empty selection is valid — every site gets zero counts.
    let dense = if selected.is_empty() {
        ndarray::Array2::<u8>::zeros((n_sites, 0))
    } else {
        db_pack::load_geno_dense_cols(&pack.dir, selected).map_err(|e| {
            Error::Schema(format!("loading geno_dense.parquet: {e}"))
        })?
    };

    for s in 0..n_sites {
        let row = dense.row(s);
        let mut n00 = 0u32;
        let mut n01 = 0u32;
        let mut n11 = 0u32;
        let mut n_miss = 0u32;
        for &d in row.iter() {
            match d {
                0 => n00 += 1,
                1 => n01 += 1,
                2 => n11 += 1,
                DOSAGE_MISSING => n_miss += 1,
                _ => n_miss += 1, // treat unexpected values as missing
            }
        }
        let ac_ctrl = n01 + 2 * n11;
        let an_ctrl = 2 * (n00 + n01 + n11);
        writeln!(
            w,
            "{chrom}\t{pos}\t{r}\t{a}\t{ac}\t{an}\t{n00}\t{n01}\t{n11}\t{nm}",
            chrom = pack.sites.chrom[s],
            pos = pack.sites.pos[s],
            r = pack.sites.ref_allele[s],
            a = pack.sites.alt_allele[s],
            ac = ac_ctrl,
            an = an_ctrl,
            n00 = n00,
            n01 = n01,
            n11 = n11,
            nm = n_miss,
        )
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }

    // Flush BufWriter → GzEncoder → BufWriter → File.
    let inner_gz = w.into_inner().map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other(format!("flushing buf writer: {e:?}")),
    })?;
    inner_gz.finish().map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::db_pack;
    use flate2::read::GzDecoder;
    use std::io::Read;

    fn read_gzip_to_string(path: &Path) -> String {
        let f = File::open(path).unwrap();
        let mut gz = GzDecoder::new(f);
        let mut s = String::new();
        gz.read_to_string(&mut s).unwrap();
        s
    }

    #[test]
    fn writes_header_and_one_row_per_site() {
        let tmp = tempfile::tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), 4, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();

        let selected = vec![0, 2];
        let out_path = tmp.path().join("out.tsv.gz");
        write(&out_path, &pack, &selected).unwrap();

        let s = read_gzip_to_string(&out_path);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(
            lines[0],
            "chrom\tpos\tref\talt\tAC_ctrl\tAN_ctrl\tn_00_ctrl\tn_01_ctrl\tn_11_ctrl\tn_miss_ctrl"
        );
        // 4 sites → 4 data rows.
        assert_eq!(lines.len(), 5);
        // Rows are in site_idx order.
        let pos_vals: Vec<u64> = lines[1..].iter()
            .map(|l| l.split('\t').nth(1).unwrap().parse().unwrap())
            .collect();
        assert_eq!(pos_vals, vec![100, 110, 120, 130]);
    }

    #[test]
    fn empty_selected_yields_zero_counts() {
        let tmp = tempfile::tempdir().unwrap();
        db_pack::fixture::build(tmp.path(), 4, 2, 2);
        let pack = db_pack::load(tmp.path()).unwrap();

        let selected: Vec<usize> = Vec::new();
        let out_path = tmp.path().join("out.tsv.gz");
        write(&out_path, &pack, &selected).unwrap();

        let s = read_gzip_to_string(&out_path);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 sites
        for line in &lines[1..] {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols[4], "0", "AC_ctrl should be 0");
            assert_eq!(cols[5], "0", "AN_ctrl should be 0");
            assert_eq!(cols[6], "0", "n_00 should be 0");
        }
    }

    #[test]
    fn ctrl_counts_match_fixture_dosage_pattern() {
        let tmp = tempfile::tempdir().unwrap();
        // 4 samples, 2 PCs, 4 sites.
        // Fixture: sample i → dosage 1 at (i%4), dosage 2 at ((i+1)%4).
        // Select all 4 samples.
        // Site 0 (pos 100): sample 0 dos=1, sample 3 dos=2 → n01=1, n11=1, n00=2
        //   AC=1+2=3, AN=2*(2+1+1)=8
        // Same pattern applies to every site by symmetry.
        db_pack::fixture::build(tmp.path(), 4, 2, 4);
        let pack = db_pack::load(tmp.path()).unwrap();

        let selected = vec![0, 1, 2, 3];
        let out_path = tmp.path().join("out.tsv.gz");
        write(&out_path, &pack, &selected).unwrap();

        let s = read_gzip_to_string(&out_path);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 5);
        for line in &lines[1..] {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols[4], "3", "AC_ctrl");
            assert_eq!(cols[5], "8", "AN_ctrl");
            assert_eq!(cols[6], "2", "n_00_ctrl");
            assert_eq!(cols[7], "1", "n_01_ctrl");
            assert_eq!(cols[8], "1", "n_11_ctrl");
            assert_eq!(cols[9], "0", "n_miss_ctrl");
        }
    }
}
