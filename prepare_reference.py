#!/usr/bin/env python3
"""
Prepare GLAD reference files for distribution.

Inputs (run from project root):
  db.bim                   - PLINK BIM (all db SNPs after MAF filter)
  db_pca.eigenvec.var      - PCA projection weights (LD-pruned SNPs, no header)
  db_meta.parquet          - DB sample metadata (for age normalization params)
  query_meta.parquet       - Query sample metadata (to generate test TSV)
  --db-pack <dir>          - db_pack/ directory (for per-site allele frequencies)

Outputs:
  glad_pca_weights.tsv.gz  - SNP weights for CLI distribution
  glad_meta.json           - Reference metadata (age scaling, build, etc.)
  query_meta.tsv           - Test TSV in the format the CLI expects from users
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl

N_PCS = 30
PC_COLS = [f"pc{i+1}" for i in range(N_PCS)]


def load_bim(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "snp_id", "cm", "pos", "a1", "a2"],
        schema_overrides={"chrom": pl.String, "snp_id": pl.Int64, "pos": pl.Int64, "a1": pl.String, "a2": pl.String},
    )


def load_eigenvec_var(path: Path) -> pl.DataFrame:
    # Format: CHROM SNP_ID EFFECT_ALLELE OTHER_ALLELE PC1..PC30 (space-separated, no header)
    return pl.read_csv(
        path,
        separator=" ",
        has_header=False,
        new_columns=["chrom", "snp_id", "effect_allele", "other_allele"] + PC_COLS,
        schema_overrides={
            "chrom": pl.String,
            "snp_id": pl.Int64,
            "effect_allele": pl.String,
            "other_allele": pl.String,
            **{c: pl.Float64 for c in PC_COLS},
        },
    )


def compute_allele_freqs(db_pack_path: Path) -> pl.DataFrame:
    """Compute VCF-ALT allele frequency for each LD-independent site via dosage data."""
    sites = pl.read_parquet(db_pack_path / "sites.parquet").filter(pl.col("ld_indep"))
    print(f"  {len(sites):,} LD-independent sites")

    print("  Loading geno_ld_indep.parquet...")
    geno_np = pl.read_parquet(db_pack_path / "geno_ld_indep.parquet").to_numpy()
    if geno_np.shape[0] != len(sites):
        raise ValueError(
            f"geno_ld_indep rows ({geno_np.shape[0]}) != ld_indep sites ({len(sites)})"
        )
    print(f"  {geno_np.shape[0]:,} sites × {geno_np.shape[1]:,} samples")

    dosage_f = geno_np.astype(np.float32)
    dosage_f[geno_np == 255] = np.nan
    alt_freq = np.nanmean(dosage_f, axis=1) / 2.0
    genotypic_std = np.nanstd(dosage_f, axis=1)

    return sites.select(["chrom", "pos", "alt"]).with_columns(
        pl.Series("alt_freq", alt_freq, dtype=pl.Float64),
        pl.Series("genotypic_std", genotypic_std, dtype=pl.Float64),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db-pack", type=Path, required=True,
                   help="Path to db_pack/ directory (source of allele frequencies)")
    args = p.parse_args()

    root = Path(__file__).parent

    print("Loading BIM...")
    bim = load_bim(root / "db.bim")
    print(f"  {len(bim):,} SNPs")

    print("Loading eigenvec.var...")
    evec = load_eigenvec_var(root / "db_pca.eigenvec.var")
    print(f"  {len(evec):,} SNPs, {len(evec.columns) - 4} PCs")

    # Join eigenvec.var onto BIM to resolve genomic positions
    weights = evec.join(bim.select(["snp_id", "pos"]), on="snp_id", how="left")
    n_unmatched = weights["pos"].null_count()
    if n_unmatched > 0:
        print(f"  WARNING: {n_unmatched:,} eigenvec.var SNPs not found in BIM — check snp_id format")
        print(f"  eigenvec.var snp_id sample: {evec['snp_id'].head(5).to_list()}")
        print(f"  BIM snp_id sample:          {bim['snp_id'].head(5).to_list()}")
    else:
        print(f"  All {len(weights):,} SNPs matched to BIM positions")

    # Verify allele consistency (effect/other should be {a1, a2} from BIM)
    check = evec.join(bim.select(["snp_id", "a1", "a2"]), on="snp_id", how="left")
    bad = check.filter(
        ~(
            ((pl.col("effect_allele") == pl.col("a1")) & (pl.col("other_allele") == pl.col("a2")))
            | ((pl.col("effect_allele") == pl.col("a2")) & (pl.col("other_allele") == pl.col("a1")))
        )
    )
    if len(bad) > 0:
        print(f"  WARNING: {len(bad):,} SNPs have allele mismatches between eigenvec.var and BIM")
        print(bad.head(5))
    else:
        print(f"  Allele check passed")

    weights = weights.select(["chrom", "pos", "effect_allele", "other_allele"] + PC_COLS)

    # Compute per-site allele frequencies from geno_ld_indep.parquet
    print("\nComputing allele frequencies from db_pack...")
    freqs = compute_allele_freqs(args.db_pack)

    # Normalize chrom for joining (strip "chr" prefix from both sides)
    weights_j = weights.with_columns(pl.col("chrom").str.strip_prefix("chr").alias("_chrom_norm"))
    freqs_j = freqs.with_columns(pl.col("chrom").str.strip_prefix("chr").alias("_chrom_norm"))

    weights = weights_j.join(
        freqs_j.select(["_chrom_norm", "pos", "alt", "alt_freq", "genotypic_std"]),
        on=["_chrom_norm", "pos"],
        how="left",
    ).drop("_chrom_norm")

    # Flip frequency when the effect allele is REF (not VCF ALT)
    weights = weights.with_columns(
        pl.when(pl.col("effect_allele") == pl.col("alt"))
        .then(pl.col("alt_freq"))
        .otherwise(1.0 - pl.col("alt_freq"))
        .alias("effect_allele_freq")
    ).drop(["alt", "alt_freq"])

    n_missing_freq = weights["effect_allele_freq"].null_count()
    if n_missing_freq > 0:
        print(f"  WARNING: {n_missing_freq:,} SNPs had no frequency match — dropping them")
        weights = weights.filter(pl.col("effect_allele_freq").is_not_null())
    print(f"  Frequencies computed for {len(weights):,} SNPs")

    out_cols = ["chrom", "pos", "effect_allele", "other_allele", "effect_allele_freq", "genotypic_std"] + PC_COLS
    out_path = root / "glad_pca_weights.tsv.gz"
    with gzip.open(out_path, "wb") as f:
        weights.select(out_cols).write_csv(f, separator="\t")
    print(f"\nWrote {out_path.name}: {len(weights):,} SNPs x {N_PCS} PCs ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Age scaling parameters from the db
    print("\nComputing age scaling from db_meta...")
    db_meta = pl.read_parquet(root / "db_meta.parquet")
    age_mean = float(db_meta["age"].mean())
    age_sd = float(db_meta["age"].std())
    print(f"  n={len(db_meta):,}  age_mean={age_mean:.3f}  age_sd={age_sd:.3f}")

    glad_meta = {
        "reference_build": "GRCh38",
        "n_pcs": N_PCS,
        "n_snps": len(weights),
        "age_mean": age_mean,
        "age_sd": age_sd,
    }
    meta_path = root / "glad_meta.json"
    with open(meta_path, "w") as f:
        json.dump(glad_meta, f, indent=2)
    print(f"Wrote {meta_path.name}")

    # Test TSV from query_meta (columns the CLI expects from users)
    print("\nGenerating query_meta.tsv...")
    query_meta = pl.read_parquet(root / "query_meta.parquet")
    tsv_path = root / "query_meta.tsv"
    query_meta.select(["sample_id", "sex", "age"]).write_csv(tsv_path, separator="\t")
    print(f"Wrote {tsv_path.name}: {len(query_meta):,} samples")
    print(f"  sex distribution: {query_meta['sex'].value_counts().sort('sex').to_dict(as_series=False)}")


if __name__ == "__main__":
    main()
