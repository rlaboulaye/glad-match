# Toy example walkthrough

A minimal end-to-end run for quickly testing the full cloistr pipeline. All
outputs use a `toy_` prefix or live in `toy_db_pack/` and `toy_ref_pack/`,
so they sit alongside any real data without conflict. To clean up, delete
files matching `toy_*` in each directory and the two `toy_*_pack/` directories.

The toy simulation uses 10 Mb of sequence, 100 reference individuals per
population, and 2 000 admixed individuals per population — fast enough to
complete in a few minutes on a laptop.

## Prerequisites

Build the binaries and install Python dependencies:

```bash
cargo build --release
uv pip install ".[sim]"
```

Requires `plink` (v1.9) and `bgzip`/`tabix` (htslib) on your PATH.

---

## Step 1 — Simulate

```bash
python sim/simulate.py \
  --config  sim/toy.config.yml \
  --out-dir raw \
  --prefix  toy_
```

Writes to `raw/`:

| File | Contents |
|---|---|
| `toy_db.vcf.gz` | Database genotypes |
| `toy_db_meta.parquet` | sample_id, population, sex, age, case_pheno1, liability_pheno1 |
| `toy_query.vcf.gz` | Query cohort genotypes |
| `toy_query_meta.parquet` | Same columns as toy_db_meta.parquet |
| `toy_query_sample_meta.tsv` | sample_id, sex, age — input for `cloistr-encode --sample-meta` |
| `toy_causal_variants.parquet` | phenotype, pos, ref, alt, effect — true causal SNPs for GWAS benchmarking |

VCF `.tbi` indexes are created automatically alongside each `.vcf.gz`.

## Step 2 — Select query population

The simulated query cohort contains multiple admixed populations. For this
walkthrough we use MXL only:

```bash
python sim/subset_query.py \
  --population MXL \
  --data-dir   raw \
  --prefix     toy_
```

Writes `raw/toy_mxl_query.vcf.gz` (+ `.tbi`), `raw/toy_mxl_query_meta.parquet`,
and `raw/toy_mxl_query_sample_meta.tsv`.

## Step 3 — PCA

```bash
plink --vcf raw/toy_db.vcf.gz \
      --make-bed --double-id \
      --out pca/toy_db

plink --bfile pca/toy_db \
      --indep-pairwise 50 10 0.1 \
      --out pca/toy_db

plink --bfile pca/toy_db \
      --extract pca/toy_db.prune.in \
      --pca 20 var-wts \
      --out pca/toy_db_pca
```

## Step 4 — Build `toy_db_pack`

```bash
python scripts/build_db_pack.py \
  --eigenvec     pca/toy_db_pca.eigenvec \
  --meta-parquet raw/toy_db_meta.parquet \
  --bim          pca/toy_db.bim \
  --prune-in     pca/toy_db.prune.in \
  --vcf          raw/toy_db.vcf.gz \
  --out-dir      toy_db_pack
```

## Step 5 — Build `toy_ref_pack`

```bash
python scripts/build_ref_pack.py \
  --db-pack      toy_db_pack \
  --bim          pca/toy_db.bim \
  --eigenvec-var pca/toy_db_pca.eigenvec.var \
  --eigenval     pca/toy_db_pca.eigenval \
  --out-dir      toy_ref_pack
```

## Step 6 — Encode the query cohort

```bash
cloistr-encode prepare \
  --vcf         raw/toy_mxl_query.vcf.gz \
  --weights     toy_ref_pack/pca_weights.tsv.gz \
  --meta        toy_ref_pack/manifest.json \
  --eigenval    toy_ref_pack/db_pca.eigenval \
  --sample-meta raw/toy_mxl_query_sample_meta.tsv \
  --output      queries/toy.enc.gz
```

## Step 7 — Run matching

```bash
cloistr run \
  --query      queries/toy.enc.gz \
  --db-pack    toy_db_pack \
  --n-controls 500 \
  --out        controls/toy_controls.tsv.gz \
  --summary    controls/toy_summary.json
```

Results are in `controls/toy_controls.tsv.gz` (per-site aggregate genotype
counts) and `controls/toy_summary.json` (λ before/after, population
breakdown, age histogram).

---

## Optional — visualize

Scatter the DB PCA and overlay the query GMM:

```bash
python scripts/plot_gmm_overlay.py \
  --query   queries/toy.enc.gz \
  --db-pack toy_db_pack \
  --out     plots/toy_gmm_overlay.png
```
