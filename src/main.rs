use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;

use glad_match::filter::FilterSpec;
use glad_match::io::{db_pack, output_tsv, query, selected_tsv, summary_json};
use glad_match::ot::SinkhornParams;
use glad_match::pipeline::{self, MatchParams};
use glad_match::refine::RefineParams;

#[derive(Parser, Debug)]
#[command(
    name = "glad-match",
    version,
    about = "Match genomic controls from the GLAD DB"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the full matching pipeline.
    Run(RunArgs),
}

#[derive(Args, Debug)]
struct RunArgs {
    /// Path to the gzipped JSON query produced by glad-prep.
    #[arg(long)]
    query: PathBuf,
    /// Directory containing the preprocessed db_pack artifacts.
    #[arg(long)]
    db_pack: PathBuf,
    /// Number of controls to select.
    #[arg(long)]
    n_controls: usize,
    /// Output TSV path (gzipped).
    #[arg(long)]
    out: PathBuf,
    /// Sidecar summary JSON path.
    #[arg(long)]
    summary: PathBuf,

    /// Multiplier on n_controls for the candidate pool.
    #[arg(long, default_value_t = 4)]
    pool_factor: usize,
    /// Seed for all randomness.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Sinkhorn entropic regularization. 0.0 → auto: median(C) / 50.
    #[arg(long, default_value_t = 0.0)]
    sinkhorn_eps: f32,
    /// Sinkhorn marginal-KL penalty (mass-deletion strength).
    #[arg(long, default_value_t = 0.1)]
    sinkhorn_rho: f32,
    #[arg(long, default_value_t = 1000)]
    sinkhorn_max_iter: usize,
    #[arg(long, default_value_t = 1e-6)]
    sinkhorn_tol: f32,

    #[arg(long, default_value_t = 10_000)]
    refine_max_iter: usize,
    #[arg(long, default_value_t = 50)]
    refine_batch: usize,
    /// Stop refinement when |log λ| < tol.
    #[arg(long, default_value_t = 0.01)]
    refine_tol: f64,
    /// Stop after this many consecutive non-improving batches.
    #[arg(long, default_value_t = 200)]
    refine_plateau: usize,
    /// Floor weight on proposal sampling to prevent starvation.
    #[arg(long, default_value_t = 0.05)]
    uniform_floor: f64,

    /// Population labels to exclude from candidate selection (comma-separated).
    #[arg(long, value_delimiter = ',')]
    exclude_population: Vec<String>,

    /// Minimum cell count before age-histogram suppression in the summary.
    #[arg(long, default_value_t = 5)]
    k_anon_min: u32,

    /// Write a TSV of selected control sample indices (internal use only).
    /// Columns: sample_idx, sample_id, sex, age, population.
    #[arg(long)]
    selected_out: Option<PathBuf>,

    /// Tracing log level (e.g. trace, debug, info, warn, error).
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&args.log_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    info!("Loading query from {}", args.query.display());
    let q = query::read(&args.query)
        .with_context(|| format!("loading query from {}", args.query.display()))?;
    info!(
        "  query: mode={}, n_samples={}, n_snps_found={}",
        q.distributions.mode.label(),
        q.n_samples,
        q.n_snps_found,
    );

    info!("Loading db_pack from {}", args.db_pack.display());
    let pack = db_pack::load(&args.db_pack)
        .with_context(|| format!("loading db_pack from {}", args.db_pack.display()))?;
    info!(
        "  db_pack: n_samples={}, n_pcs={}, n_ld_indep={}",
        pack.manifest.n_samples, pack.manifest.n_pcs, pack.manifest.n_sites_ld_indep,
    );

    let n_controls = args.n_controls;
    let params = MatchParams {
        n_controls,
        pool_factor: args.pool_factor,
        seed: args.seed,
        sinkhorn: SinkhornParams {
            eps: args.sinkhorn_eps,
            rho: args.sinkhorn_rho,
            max_iter: args.sinkhorn_max_iter,
            tol: args.sinkhorn_tol,
        },
        refine: RefineParams {
            n_controls,
            seed: args.seed,
            max_iter: args.refine_max_iter,
            batch: args.refine_batch,
            tol: args.refine_tol,
            plateau: args.refine_plateau,
            uniform_floor: args.uniform_floor,
        },
        filter: FilterSpec {
            exclude_populations: args.exclude_population,
        },
    };

    let started = Instant::now();
    info!("Running match_controls (n_controls={n_controls})");
    let result = pipeline::match_controls(&q, &pack, &params).context("matching controls")?;
    let elapsed = started.elapsed();
    info!(
        "Done in {:.2}s | λ {:.3} → {:.3} | iters={} | accepted_swaps={}",
        elapsed.as_secs_f32(),
        result.initial_lambda,
        result.final_lambda,
        result.iterations,
        result.accepted_swaps,
    );

    info!("Writing TSV → {}", args.out.display());
    output_tsv::write(&args.out, &pack, &result.selected)
        .with_context(|| format!("writing TSV to {}", args.out.display()))?;

    if let Some(ref sel_path) = args.selected_out {
        info!("Writing selected indices → {}", sel_path.display());
        selected_tsv::write(sel_path, &result.selected, &pack.samples)
            .with_context(|| format!("writing selected indices to {}", sel_path.display()))?;
    }

    info!("Writing summary → {}", args.summary.display());
    let summary = pipeline::build_summary(
        &q,
        &pack,
        &result,
        args.seed,
        pack.samples.sample_ids.len(),
        &summary_json::default_age_bins(),
        args.k_anon_min,
    );
    summary_json::write(&args.summary, &summary)
        .with_context(|| format!("writing summary to {}", args.summary.display()))?;

    Ok(())
}
