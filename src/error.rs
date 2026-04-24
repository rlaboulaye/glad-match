use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {what}: {source}")]
    Parse {
        what: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("schema mismatch: {0}")]
    Schema(String),

    #[error(transparent)]
    Polars(#[from] polars::error::PolarsError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
