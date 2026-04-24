#!/usr/bin/env python3
"""
build_db_pack.py — convert raw glad artifacts into a glad-match `db_pack/`.

Convenience-only: the real glad-match crate assumes its input is already a
preprocessed db_pack. In production the preprocessing happens elsewhere.

Outputs (in --out-dir):
    manifest.json          — version, reference_build, n_samples, n_pcs,
                             n_sites, n_sites_pruned, age_mean, age_sd,
                             created_at
    samples.parquet        — sample_id, sex, age, population,
                             pc0..pc{n_pcs-1}
    sites.parquet          — ALL single-allelic db sites:
                             chrom, pos, ref, alt, in_pruned
                             (row order = site_idx, 0..n_sites)
    geno_dense.parquet     — site-major dense dosages. One row per site,
                              one column per sample (named `sample_0` ..
                              `sample_{n_samples-1}`), value type uint8;
                              dosage encoding
                              {0=HOM_REF, 1=HET, 2=HOM_ALT, 255=MISSING}.
                              Zstd-compressed.
    geno_pruned.bin        — sample-major sparse records for the pruned
                             subset, used by refinement's incremental χ²:
                             5-byte records (u32 site_idx LE, u8 dosage).
                             site_idx refers to rows of sites.parquet
                             (not a pruned-only sub-index).
    geno_pruned.off        — u64 LE offsets into geno_pruned.bin,
                             length n_samples + 1.

Conventions:
  - A variant is included iff it is single-allelic in the VCF (one ALT).
  - Dosage orientation follows the VCF's REF/ALT. For prune-set sites whose
    bim alleles disagree with the VCF, `in_pruned = True` is still set, but
    glad-prep's query side labels those sites with the bim's alleles, so the
    refinement inner-join will drop them automatically. This keeps the VCF
    as the single source of truth for db dosages.
  - cyvcf2 gt_types: 0 = HOM_REF, 1 = HET, 2 = UNKNOWN (missing), 3 = HOM_ALT.
  - Samples are reindexed to eigenvec order (canonical).

Dependencies (install into .venv with uv):
    uv pip install numpy pyarrow cyvcf2 polars

Usage:
    python build_db_pack.py \\
        --eigenvec db_pca.eigenvec \\
        --meta-parquet db_meta.parquet \\
        --bim db.bim \\
        --prune-in db.prune.in \\
        --vcf db.vcf.gz \\
        --glad-meta /path/to/glad_meta.json \\
        --out-dir db_pack
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print(
        "error: pyarrow is not installed. Install with `uv pip install pyarrow`.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from cyvcf2 import VCF
except ImportError:
    print(
        "error: cyvcf2 is not installed. Install with `uv pip install cyvcf2` "
        "(may require a working htslib).",
        file=sys.stderr,
    )
    sys.exit(1)


DOSAGE_MISSING = 255


def read_eigenvec(path: Path) -> tuple[list[str], np.ndarray]:
    """PLINK .eigenvec: FID IID PC1 PC2 ... per line, whitespace-separated."""
    sample_ids: list[str] = []
    pc_rows: list[list[float]] = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sample_ids.append(parts[1])
            pc_rows.append([float(x) for x in parts[2:]])
    return sample_ids, np.asarray(pc_rows, dtype=np.float32)


def read_bim_prune_positions(
    bim_path: Path, prune_path: Path
) -> set[tuple[str, int]]:
    """Parse bim + prune.in and return the set of (chrom, bp) for pruned sites."""
    prune_snpids: set[str] = set()
    with open(prune_path) as f:
        for line in f:
            s = line.strip()
            if s:
                prune_snpids.add(s)

    positions: set[tuple[str, int]] = set()
    n_missing = 0
    with open(bim_path) as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                raise ValueError(
                    f"{bim_path}: line {i+1} has {len(parts)} cols, expected 6"
                )
            chrom, snpid, _cm, bp, _a1, _a2 = parts[:6]
            if snpid in prune_snpids:
                positions.add((chrom, int(bp)))
                prune_snpids.discard(snpid)
    if prune_snpids:
        n_missing = len(prune_snpids)
        print(
            f"warning: {n_missing} prune snpids not found in bim",
            file=sys.stderr,
        )
    return positions


def reorder_meta(meta_df: pl.DataFrame, sample_ids: list[str]) -> pl.DataFrame:
    order_df = pl.DataFrame(
        {"sample_id": sample_ids, "_order": list(range(len(sample_ids)))}
    )
    joined = order_df.join(meta_df, on="sample_id", how="left")
    null_count = joined["sex"].null_count()
    if null_count > 0:
        missing = (
            joined.filter(pl.col("sex").is_null())["sample_id"].head(5).to_list()
        )
        raise ValueError(
            f"meta missing for {null_count} samples (e.g. {missing})"
        )
    return joined.sort("_order").drop("_order")


def flush_dense_chunk(
    writer: "pq.ParquetWriter",
    buf: np.ndarray,
    fill: int,
    names: list[str],
) -> None:
    """Write `fill` rows of `buf` (row-major, one row per site) as a parquet row group.

    Transposes once into column-major so each per-sample slice is contiguous
    and `pa.array` can wrap it zero-copy.
    """
    if fill == 0:
        return
    col_major = np.ascontiguousarray(buf[:fill].T)
    arrays = [pa.array(col_major[j], type=pa.uint8()) for j in range(col_major.shape[0])]
    batch = pa.RecordBatch.from_arrays(arrays, names=names)
    writer.write_batch(batch)


def write_samples_parquet(
    out_path: Path,
    sample_ids: list[str],
    meta_df: pl.DataFrame,
    pca: np.ndarray,
) -> None:
    n_pcs = pca.shape[1]
    cols: dict = {
        "sample_id": sample_ids,
        "sex": meta_df["sex"].cast(pl.Int32).to_list(),
        "age": meta_df["age"].cast(pl.Float32).to_list(),
        "population": meta_df["population"].to_list(),
    }
    for j in range(n_pcs):
        cols[f"pc{j}"] = pca[:, j].astype(np.float32).tolist()
    pl.DataFrame(cols).write_parquet(out_path, compression="zstd")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eigenvec", type=Path, required=True)
    p.add_argument("--meta-parquet", type=Path, required=True)
    p.add_argument("--bim", type=Path, required=True)
    p.add_argument("--prune-in", type=Path, required=True)
    p.add_argument("--vcf", type=Path, required=True)
    p.add_argument("--glad-meta", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    glad_meta = json.loads(args.glad_meta.read_text())
    print(
        f"glad_meta: build={glad_meta['reference_build']}, "
        f"n_pcs={glad_meta['n_pcs']}, age_mean={glad_meta['age_mean']:.3f}, "
        f"age_sd={glad_meta['age_sd']:.3f}"
    )

    print(f"loading {args.eigenvec} ...")
    sample_ids, pca = read_eigenvec(args.eigenvec)
    n_samples, n_pcs = pca.shape
    print(f"  {n_samples} samples × {n_pcs} PCs")

    print(f"loading {args.meta_parquet} ...")
    meta_df = pl.read_parquet(args.meta_parquet)
    meta_df = reorder_meta(meta_df, sample_ids)

    print("writing samples.parquet ...")
    write_samples_parquet(
        args.out_dir / "samples.parquet", sample_ids, meta_df, pca
    )

    print(f"loading prune set from {args.bim} + {args.prune_in} ...")
    prune_positions = read_bim_prune_positions(args.bim, args.prune_in)
    print(f"  {len(prune_positions)} pruned (chrom, pos) keys")

    print(f"opening VCF {args.vcf} ...")
    vcf = VCF(str(args.vcf))
    vcf_samples = list(vcf.samples)
    print(f"  VCF has {len(vcf_samples)} samples")

    vcf_index_of = {sid: i for i, sid in enumerate(vcf_samples)}
    perm = np.empty(n_samples, dtype=np.int64)
    missing_in_vcf: list[str] = []
    for j, sid in enumerate(sample_ids):
        idx = vcf_index_of.get(sid)
        if idx is None:
            missing_in_vcf.append(sid)
        else:
            perm[j] = idx
    if missing_in_vcf:
        raise SystemExit(
            f"VCF missing {len(missing_in_vcf)} eigenvec samples: "
            f"{missing_in_vcf[:5]}..."
        )

    gt_to_dosage = np.array(
        [0, 1, DOSAGE_MISSING, 2], dtype=np.uint8
    )

    # Dense genotype output: one row per site, one u8 column per sample.
    # Stream row groups to avoid holding all n_sites × n_samples bytes in RAM.
    SITES_PER_CHUNK = 4096
    sample_names = [f"sample_{j}" for j in range(n_samples)]
    dense_schema = pa.schema([pa.field(name, pa.uint8()) for name in sample_names])
    dense_path = args.out_dir / "geno_dense.parquet"
    dense_writer = pq.ParquetWriter(dense_path, dense_schema, compression="zstd")
    dense_buf = np.empty((SITES_PER_CHUNK, n_samples), dtype=np.uint8)
    dense_fill = 0

    site_chroms: list[str] = []
    site_positions: list[int] = []
    site_refs: list[str] = []
    site_alts: list[str] = []
    site_in_pruned: list[bool] = []

    per_sample_buf: list[bytearray] = [bytearray() for _ in range(n_samples)]

    n_seen = 0
    n_multi_skipped = 0
    n_pruned_emitted = 0
    site_idx = 0
    t0 = time.time()
    progress_every = 50_000

    for variant in vcf:
        n_seen += 1
        if n_seen % progress_every == 0:
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1e-6)
            print(
                f"  {n_seen:>9d} variants scanned "
                f"({elapsed:.0f}s, {rate:.0f}/s; emitted={site_idx})"
            )

        alts = variant.ALT
        if len(alts) != 1:
            n_multi_skipped += 1
            continue

        chrom = variant.CHROM
        pos = variant.POS
        ref = variant.REF
        alt = alts[0]
        in_pruned = (chrom, pos) in prune_positions

        # Compute per-sample dosage in eigenvec order.
        gt = np.asarray(variant.gt_types, dtype=np.int32)
        # Clip any out-of-range values defensively (shouldn't occur).
        gt = np.where((gt >= 0) & (gt <= 3), gt, 2)
        gt_e = gt[perm]
        dosage = gt_to_dosage[gt_e]

        site_chroms.append(chrom)
        site_positions.append(pos)
        site_refs.append(ref)
        site_alts.append(alt)
        site_in_pruned.append(in_pruned)

        dense_buf[dense_fill] = dosage
        dense_fill += 1
        if dense_fill == SITES_PER_CHUNK:
            flush_dense_chunk(dense_writer, dense_buf, dense_fill, sample_names)
            dense_fill = 0

        if in_pruned:
            site_bytes = struct.pack("<I", site_idx)
            # Non-hom-ref samples only (dosage != 0).
            nonref_j = np.where(dosage != 0)[0]
            for j in nonref_j:
                buf = per_sample_buf[int(j)]
                buf.extend(site_bytes)
                buf.append(int(dosage[j]))
            n_pruned_emitted += 1

        site_idx += 1

    elapsed = time.time() - t0
    print(
        f"  scanned {n_seen} variants in {elapsed:.0f}s; "
        f"emitted {site_idx} sites (skipped {n_multi_skipped} multi-allelic); "
        f"pruned subset = {n_pruned_emitted}"
    )
    if site_idx == 0:
        raise SystemExit("no sites emitted; aborting")

    print("finalizing geno_dense.parquet ...")
    flush_dense_chunk(dense_writer, dense_buf, dense_fill, sample_names)
    dense_writer.close()
    dense_size = dense_path.stat().st_size
    print(f"  geno_dense.parquet: {dense_size / 1e6:.1f} MB")

    print("writing sites.parquet ...")
    pl.DataFrame(
        {
            "chrom": site_chroms,
            "pos": site_positions,
            "ref": site_refs,
            "alt": site_alts,
            "in_pruned": site_in_pruned,
        }
    ).write_parquet(args.out_dir / "sites.parquet", compression="zstd")

    print("writing geno_pruned.bin + .off ...")
    bin_path = args.out_dir / "geno_pruned.bin"
    off_path = args.out_dir / "geno_pruned.off"
    offsets: list[int] = [0]
    with open(bin_path, "wb") as bf:
        for buf in per_sample_buf:
            if buf:
                bf.write(buf)
            offsets.append(bf.tell())
    with open(off_path, "wb") as of:
        for o in offsets:
            of.write(struct.pack("<Q", o))
    bin_size = bin_path.stat().st_size
    print(
        f"  geno_pruned.bin: {bin_size / 1e6:.1f} MB ({bin_size // 5} records); "
        f".off: {len(offsets)} u64 offsets"
    )

    print("writing manifest.json ...")
    manifest = {
        "version": "1.0",
        "reference_build": glad_meta["reference_build"],
        "n_samples": n_samples,
        "n_pcs": n_pcs,
        "n_sites": site_idx,
        "n_sites_pruned": n_pruned_emitted,
        "age_mean": float(glad_meta["age_mean"]),
        "age_sd": float(glad_meta["age_sd"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = time.time() - t0
    print(f"done in {total:.0f}s. db_pack written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
