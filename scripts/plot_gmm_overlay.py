#!/usr/bin/env python3
"""
Overlay GMM components from a query.glad.gz onto the GLAD DB PCA scatter.

Usage:
    python plot_gmm_overlay.py --query query.glad.gz --db-pack db_pack/ [--out plot.png]
"""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Ellipse


def draw_ellipse(ax, mean_2d, cov_2x2, n_std, **kwargs):
    vals, vecs = np.linalg.eigh(cov_2x2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0.0))
    ax.add_patch(Ellipse(xy=mean_2d, width=w, height=h, angle=angle, **kwargs))


def plot_gmm_on_ax(ax, gmm, db_df, pc_x, pc_y, title):
    """Plot DB scatter then overlay GMM ellipses. Returns (pop_handles, comp_handles)."""
    populations = db_df["population"].unique().sort().to_list()
    pop_colors = plt.cm.tab20(np.linspace(0, 1, len(populations)))

    # Scatter first so matplotlib autoscales to the data.
    pop_handles = []
    for pop, color in zip(populations, pop_colors):
        sub = db_df.filter(pl.col("population") == pop)
        h = ax.scatter(
            sub[f"pc{pc_x}"].to_numpy(),
            sub[f"pc{pc_y}"].to_numpy(),
            s=2, alpha=0.3, color=color, rasterized=True,
        )
        pop_handles.append((h, pop))

    # Snapshot scatter-based limits before ellipses can expand them.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    n_components = gmm["n_components"]
    component_colors = plt.cm.Set1(np.linspace(0, 0.9, n_components))

    comp_handles = []
    for c in range(n_components):
        mean_2d = np.array([gmm["means"][c][pc_x], gmm["means"][c][pc_y]])
        cov_full = np.array(gmm["covariances"][c])
        cov_2d = cov_full[np.ix_([pc_x, pc_y], [pc_x, pc_y])]
        r, g, b = component_colors[c][:3]
        draw_ellipse(
            ax, mean_2d, cov_2d, n_std=2.0,
            facecolor=(r, g, b, 0.08),
            edgecolor=(r, g, b, 0.8),
            linewidth=1.5, linestyle="--", zorder=1,
        )
        draw_ellipse(
            ax, mean_2d, cov_2d, n_std=1.0,
            facecolor=(r, g, b, 0.18),
            edgecolor=(r, g, b, 0.9),
            linewidth=1.5, linestyle="-", zorder=1,
        )
        (h,) = ax.plot(*mean_2d, "+", color=(r, g, b), markersize=10, zorder=3)
        comp_handles.append((h, f"component {c + 1} (w={gmm['weights'][c]:.2f})"))

    # Restore the scatter-based limits.
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(f"PC {pc_x + 1}")
    ax.set_ylabel(f"PC {pc_y + 1}")
    ax.set_title(title)

    return pop_handles, comp_handles


def main():
    parser = argparse.ArgumentParser(
        description="Overlay query GMM from a .glad.gz onto the GLAD DB PCA scatter."
    )
    parser.add_argument("--query", required=True, help="Path to query.glad.gz")
    parser.add_argument("--db-pack", required=True, help="Path to db_pack/ directory")
    parser.add_argument("--pc-x", type=int, default=0, help="PC index for x-axis, 0-based (default: 0)")
    parser.add_argument("--pc-y", type=int, default=1, help="PC index for y-axis, 0-based (default: 1)")
    parser.add_argument("--out", default=None, help="Output image path (default: display)")
    args = parser.parse_args()

    with gzip.open(args.query, "rt") as f:
        query = json.load(f)

    db_df = pl.read_parquet(Path(args.db_pack) / "samples.parquet")
    dist = query["distributions"]
    mode = dist["mode"]
    n_dims = dist["n_dims"]
    pc_x, pc_y = args.pc_x, args.pc_y

    n_pcs = n_dims - 1 if mode in ("sex_and_age", "age_only") else n_dims
    if pc_x >= n_pcs or pc_y >= n_pcs:
        raise ValueError(
            f"--pc-x and --pc-y must be < {n_pcs} for mode '{mode}' (n_dims={n_dims})"
        )

    sex_stratified = mode in ("sex_only", "sex_and_age")

    if sex_stratified:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        all_pop_handles = None
        for ax, sex_val, gmm_key, label in [
            (axes[0], 0, "female", "Female"),
            (axes[1], 1, "male", "Male"),
        ]:
            sex_df = db_df.filter(pl.col("sex") == sex_val)
            pop_handles, comp_handles = plot_gmm_on_ax(
                ax, dist[gmm_key], sex_df, pc_x, pc_y, label
            )
            if all_pop_handles is None:
                all_pop_handles = pop_handles
            ax.legend(
                [h for h, _ in comp_handles],
                [l for _, l in comp_handles],
                title="GMM", fontsize=8, loc="best",
            )

        fig.legend(
            [h for h, _ in all_pop_handles],
            [l for _, l in all_pop_handles],
            title="Population", loc="lower center",
            ncol=min(len(all_pop_handles), 8),
            bbox_to_anchor=(0.5, -0.04),
            markerscale=4, fontsize=8,
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        pop_handles, comp_handles = plot_gmm_on_ax(
            ax, dist["all"], db_df, pc_x, pc_y,
            f"DB PCA — Query GMM overlay (mode={mode})",
        )
        all_handles = pop_handles + comp_handles
        ax.legend(
            [h for h, _ in all_handles],
            [l for _, l in all_handles],
            title="Population / GMM", markerscale=3, fontsize=8, loc="best",
        )

    n_samples = query["n_samples"]
    per_sex = query.get("per_sex_counts")
    sex_str = (f" (F={per_sex['female']}, M={per_sex['male']})" if per_sex else "")
    fig.suptitle(
        f"Query GMM overlay — PC {pc_x + 1} vs PC {pc_y + 1} | "
        f"n_query={n_samples}{sex_str} | mode={mode} | 1σ/2σ ellipses",
        fontsize=11,
    )
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
