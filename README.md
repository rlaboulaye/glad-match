# CLOISTR <br> <sub>Controls via Latent-space Optimization using Inferred Statistical TRansport</sub>

Select ancestry-, age-, and sex-matched genomic controls from the GLAD database for a query cohort, returning only aggregate allele and genotype counts — no individual-level data crosses either direction.

The typical workflow is:

1. Run [glad-prep](https://github.com/rlaboulaye/glad-prep) on your cohort to produce a `.glad.gz` query file (per-site allele counts + a fitted GMM over PCA/age space).
2. Run `glad-match` against the GLAD `db_pack` to select matched controls.
3. Use the output TSV alongside your own cohort's counts for downstream GWAS or other analyses.

## Setup

```bash
cargo build --release
# binary at target/release/glad-match
```

The preprocessing script (`build_db_pack.py`) requires Python with `numpy`, `pyarrow`, `cyvcf2`, and `polars`:

```bash
uv pip install numpy pyarrow cyvcf2 polars
```

## The `db_pack` format

`glad-match` consumes a preprocessed `db_pack/` directory with the following files:

| File | Description |
|---|---|
| `manifest.json` | Shape constants (`n_samples`, `n_pcs`, `n_sites`, `n_sites_ld_indep`) and age normalization parameters (`age_mean`, `age_sd`) |
| `samples.parquet` | Per-sample metadata: `sample_id`, `sex` (0=F, 1=M), `age`, `population`, `pc0`…`pc{n_pcs-1}` |
| `sites.parquet` | All db sites: `chrom`, `pos`, `ref`, `alt`, `ld_indep` |
| `geno_dense.parquet` | Site-major dosage matrix (one row per site, one `uint8` column per sample named `sample_0`…`sample_{n-1}`; dosages 0/1/2/255). Used by the output writer for full-site coverage. |
| `geno_ld_indep.parquet` | Same layout as `geno_dense.parquet`, restricted to the `n_sites_ld_indep` rows where `ld_indep = true`. Used by the refinement stage with column projection onto the candidate pool. |

### Building a `db_pack` from raw data

`build_db_pack.py` converts the standard GLAD DB artifacts into this format. It expects:

- A **bgzipped, tabix-indexed VCF** of all samples (`db.vcf.gz`)
- **PLINK PCA outputs**: `.eigenvec` (PC coordinates)
- **PLINK LD-pruning outputs**: `.bim` and `.prune.in` (the kept-SNP list)
- A **sample metadata Parquet** file with columns `sample_id`, `sex`, `age`, `population`
- A **glad-prep metadata JSON** (`glad_meta.json`) containing `reference_build`, `n_pcs`, `age_mean`, and `age_sd`

```bash
python build_db_pack.py \
  --eigenvec     db_pca.eigenvec \
  --meta-parquet db_meta.parquet \
  --bim          db.bim \
  --prune-in     db.prune.in \
  --vcf          db.vcf.gz \
  --glad-meta    glad_meta.json \
  --out-dir      db_pack/
```

## Running `glad-match`

```bash
glad-match run \
  --query      query.glad.gz \
  --db-pack    db_pack/ \
  --n-controls 500 \
  --out        controls.tsv.gz \
  --summary    summary.json \
  [--selected-out selected.tsv]   # internal use only; see below
```

Run `glad-match run --help` for the full parameter list.

### Outputs

**`controls.tsv.gz`** — gzipped TSV of per-site aggregate control counts, one row per db site:

| Column | Description |
|---|---|
| `chrom`, `pos`, `ref`, `alt` | Site coordinates |
| `AC_ctrl` | Alt allele count across selected controls |
| `AN_ctrl` | Total allele count (excludes missing calls) |
| `n_00_ctrl` | HOM REF genotype count |
| `n_01_ctrl` | HET genotype count |
| `n_11_ctrl` | HOM ALT genotype count |
| `n_miss_ctrl` | Missing-call count |

`AC_ctrl = n_01 + 2 × n_11`; `AN_ctrl = 2 × (n_00 + n_01 + n_11)`.

**`summary.json`** — pipeline diagnostics and aggregate demographics of the selected controls: Sinkhorn convergence, initial/final genomic control λ, per-population counts, and an age histogram (cells with fewer than 5 individuals are suppressed for privacy).

**`--selected-out selected.tsv`** (optional) — plain TSV listing the selected controls' db indices and metadata (`sample_idx`, `sample_id`, `sex`, `age`, `population`). Intended for internal QC only; not returned to external users.

## Key tuning parameters

| Flag | Default | Notes |
|---|---|---|
| `--n-controls` | — | Number of controls to select (required) |
| `--pool-factor` | 4 | Candidate pool size as a multiple of `n-controls` |
| `--seed` | 42 | RNG seed for reproducibility |
| `--refine-tol` | 0.01 | Stop refinement when `|log λ| < tol` |
| `--sinkhorn-eps` | auto | OT regularization ε (0 → `median(cost)/50`) |
| `--sinkhorn-rho` | 0.1 | Marginal-KL penalty; lower = more mass may be discarded |
| `--exclude-population` | — | Comma-separated population labels to exclude |

## Privacy

The user submits only aggregates: per-site alt/total allele counts and a GMM fit over PCA (and optionally age) space. No individual genotypes, phenotype labels, or PCA projections leave the user's environment.

`glad-match` returns only aggregates: per-site genotype counts over the selected control set, and a k-anonymized demographic summary. No individual db sample records are included in any output returned to the user. The optional `--selected-out` file is for internal operator use only.

## Algorithm

See [ALGORITHM.md](ALGORITHM.md) for a detailed description of the optimal transport candidate selection, incremental χ² greedy refinement, and the design choices behind the genomic control λ objective.
