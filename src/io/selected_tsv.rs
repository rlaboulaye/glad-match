//! Writer for the optional selected-controls index file.
//!
//! Output: plain (uncompressed) TSV with columns:
//!   `sample_idx  sample_id  sex  age  population`
//!
//! Enabled by `--selected-out <path>`. Intended for internal use only —
//! never returned to external users.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::io::db_pack::DbSamples;

pub fn write<P: AsRef<Path>>(
    path: P,
    selected: &[usize],
    samples: &DbSamples,
) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut w = BufWriter::new(file);

    writeln!(w, "sample_idx\tsample_id\tsex\tage\tpopulation").map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;

    for &idx in selected {
        writeln!(
            w,
            "{idx}\t{sid}\t{sex}\t{age:.1}\t{pop}",
            sid = samples.sample_ids[idx],
            sex = samples.sex[idx],
            age = samples.age[idx],
            pop = samples.population[idx],
        )
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }

    Ok(())
}
