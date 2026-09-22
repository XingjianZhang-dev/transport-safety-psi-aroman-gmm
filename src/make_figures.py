"""Reproduce Figs. 3-9 and 11 of the article.

Usage:  python3 src/make_figures.py [calibrated|reconstructed]
Outputs: results/<dataset>/figures/*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mcdm
import reference_values as R
from run_ranking import load

ROOT = Path(__file__).resolve().parent.parent
YEAR_COLOUR = {2010: "tab:blue", 2015: "tab:orange", 2019: "tab:green", 2023: "tab:red"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9})


def radar_axes(ax, labels):
    ang = np.arange(len(labels)) * 2 * np.pi / len(labels)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels)
    return ang


def fig7(ds, out):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15.6), subplot_kw=dict(polar=True))
    for ax, k in zip(axs.ravel(), R.INDICATORS):
        ang = radar_axes(ax, R.C)
        for y in R.YEARS:
            v = load(ds, y)[k].values
            ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], "-*", color=YEAR_COLOUR[y], lw=1, ms=5, label=str(y))
            ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=YEAR_COLOUR[y], alpha=0.1)
        ax.set_title(k, fontsize=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8)
    fig.suptitle("Dynamic changes of safety performance indicators", fontsize=15)
    fig.tight_layout(); fig.savefig(out / "fig7_dynamic_changes_of_spis.png"); plt.close(fig)


def fig8(ds, out):
    V = pd.DataFrame(mcdm.vector_norm(load(ds, 2023).values, R.COST), index=R.C, columns=R.INDICATORS)
    ax = V.plot(kind="bar", stacked=True, figsize=(12, 7), width=0.5)
    ax.set_ylabel("Normalized Value"); ax.set_xlabel("Country"); ax.set_title("Year 2023")
    ax.legend(title="Criteria", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(out / "fig8_deconstruction_of_composite_index.png"); plt.close()
    return V


def fig9(ds, out):
    D = pd.DataFrame(mcdm.vector_norm(load(ds, 2023).values, R.COST)
                     - mcdm.vector_norm(load(ds, 2010).values, R.COST), index=R.C, columns=R.INDICATORS)
    fig, ax = plt.subplots(figsize=(12, 7))
    pos, neg = np.zeros(19), np.zeros(19)
    for k in R.INDICATORS:
        v = D[k].values
        b = ax.bar(R.C, np.where(v > 0, v, 0), bottom=pos, width=0.5, label=k)
        ax.bar(R.C, np.where(v < 0, v, 0), bottom=neg, width=0.5, color=b.patches[0].get_facecolor())
        pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.axhline(0, color="k", lw=1); ax.set_ylabel("Normalized Value"); ax.set_xlabel("Country")
    ax.set_title("Year 2010 - 2023"); ax.legend(title="Criteria", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(out / "fig9_decomposition_of_score_changes.png"); plt.close(fig)


def rank_lines(ds, methods, title, fname, out):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for ax, y in zip(axs.ravel(), R.YEARS):
        X = load(ds, y).values
        ranks = {m: mcdm.ranks_from_scores(mcdm.run_method(X, R.COST, m)[0]) for m in methods}
        order = np.argsort(ranks[methods[0]])
        for m in methods:
            ax.plot(np.array(R.C)[order], ranks[m][order], "-o", ms=3, lw=1, label=m)
        ax.set_title(f"{title} ({y})"); ax.set_ylabel("Ranking"); ax.set_xlabel("Country")
        ax.set_yticks(range(1, 20)); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / fname); plt.close(fig)


def fig3(ds, out):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for ax, y in zip(axs.ravel(), R.YEARS):
        e = pd.read_csv(ROOT / f"results/{ds}/tsne_embedding_{y}.csv", index_col=0)
        sc = ax.scatter(e["Component 1"], e["Component 2"], c=e["Group"], cmap="viridis", s=20)
        for c, row in e.iterrows():
            ax.annotate(c, (row["Component 1"], row["Component 2"]), fontsize=8,
                        xytext=(2, 2), textcoords="offset points")
        ax.set_title(f"GMM Clustering with t-SNE (Year {y})")
        ax.set_xlabel("t-SNE Component 1"); ax.set_ylabel("t-SNE Component 2")
        plt.colorbar(sc, ax=ax, label="Group")
    fig.tight_layout(); fig.savefig(out / "fig3_gmm_clustering_with_tsne.png"); plt.close(fig)


def fig11(ds, V, out):
    groups = pd.read_csv(ROOT / f"results/{ds}/table2_groups.csv", index_col=0)["2023"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))
    for g, ax in zip([1, 2, 3, 4], axs.ravel()):
        ang = radar_axes(ax, R.INDICATORS)
        for c in groups.index[groups == g]:
            v = V.loc[c].values
            ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], "-o", ms=3, label=c)
            ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], alpha=0.08)
        ax.set_title(f"Group {g}")
        ax.legend(bbox_to_anchor=(1.2, 1.1), fontsize=8)
    fig.suptitle("Benchmarking of SPIs within each group (2023)", fontsize=14)
    fig.tight_layout(); fig.savefig(out / "fig11_benchmarking_within_groups.png"); plt.close(fig)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "calibrated"
    out = ROOT / "results" / ds / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig7(ds, out)
    V = fig8(ds, out)
    fig9(ds, out)
    rank_lines(ds, ["Vector", "MinMax", "Max"], "Initial Sensitivity", "fig4_ranking_normalisation.png", out)
    rank_lines(ds, ["PSI", "CRITIC", "Entropy"], "Medial Stability", "fig5_ranking_weighting.png", out)
    rank_lines(ds, ["AROMAN", "COPRAS", "PROMETHEE"], "Lateral Reliability", "fig6_ranking_aggregation.png", out)
    fig3(ds, out)
    fig11(ds, V, out)
    print("figures written to", out)
