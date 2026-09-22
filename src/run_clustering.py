"""Grouping of the G20 countries: Tables 2, 9-12 and the clustering-quality metrics.

Pipeline (Section 4.3): scaled safety performance indicators -> t-SNE (2 components,
perplexity 5) -> Gaussian mixture model with four components; clusters are numbered by
ascending mean of A31 (road fatalities per 100,000 inhabitants), so Group 1 is the
best-performing cluster.  Benchmarks: k-means and fuzzy c-means on the same features.

Usage:  python3 src/run_clustering.py [calibrated|reconstructed]

Outputs (results/<dataset>/):
  table2_groups.csv, table9_grouping_normalisation.csv, table10_grouping_vmeasure.csv,
  table11_grouping_methods.csv, table12_grouping_vmeasure.csv,
  clustering_quality.csv, tsne_embedding_{year}.csv
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer

import reference_values as R
from run_ranking import load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SCALERS = {"VE": Normalizer, "MM": MinMaxScaler, "MA": MaxAbsScaler}
SEED = 42


def order_by_fatalities(labels, a31):
    means = pd.Series(a31).groupby(labels).mean().sort_values()
    mp = {old: i + 1 for i, old in enumerate(means.index)}
    return np.array([mp[l] for l in labels])


def gmm(data, seed=SEED):
    data = np.asarray(data, dtype=np.float64)
    for reg in [1e-6, 1e-4, 1e-2, 1e-1, 1.0]:
        try:
            return GaussianMixture(n_components=4, random_state=seed, reg_covar=reg).fit(data)
        except ValueError:
            continue
    raise RuntimeError("GMM did not converge")


def gmm_tsne(X, scaler="VE", seed=SEED):
    Z = SCALERS[scaler]().fit_transform(X.values)
    emb = TSNE(n_components=2, perplexity=5, random_state=seed).fit_transform(Z).astype(np.float64)
    return order_by_fatalities(gmm(emb, seed).predict(emb), X["A31"].values), emb, Z


def benchmarks(X, seed=SEED):
    Z = Normalizer().fit_transform(X.values)
    km = KMeans(n_clusters=4, random_state=seed, n_init=10).fit_predict(Z)
    _, u, *_ = fuzz.cluster.cmeans(Z.T, c=4, m=2, error=0.005, maxiter=1000, seed=seed)
    return (order_by_fatalities(km, X["A31"].values),
            order_by_fatalities(np.argmax(u, 0), X["A31"].values))


def main(dataset="calibrated", n_runs=50):
    out = ROOT / "results" / dataset
    out.mkdir(parents=True, exist_ok=True)
    G, quality = {}, []
    for y in R.YEARS:
        X = load(dataset, y)
        for sc in SCALERS:
            G[(y, sc)], emb, Z = gmm_tsne(X, sc)
            if sc == "VE":
                pd.DataFrame(emb, index=R.C, columns=["Component 1", "Component 2"]).assign(
                    Group=G[(y, "VE")]).to_csv(out / f"tsne_embedding_{y}.csv")
        G[(y, "K")], G[(y, "C")] = benchmarks(X)
        G[(y, "Gt")] = G[(y, "VE")]
        # clustering quality over repeated runs
        for s in range(n_runs):
            e = TSNE(n_components=2, perplexity=5, random_state=s).fit_transform(Z).astype(float)
            lab_g, lab_t = gmm(Z, s).predict(Z), gmm(e, s).predict(e)
            quality.append(dict(year=y, run=s,
                                silhouette_gmm=silhouette_score(Z, lab_g),
                                silhouette_tsne_gmm=silhouette_score(e, lab_t),
                                davies_bouldin_gmm=davies_bouldin_score(Z, lab_g),
                                davies_bouldin_tsne_gmm=davies_bouldin_score(e, lab_t)))
    pd.DataFrame(G, index=R.C)[[(y, "VE") for y in R.YEARS]].set_axis(
        R.YEARS, axis=1).rename_axis("Code").to_csv(out / "table2_groups.csv")
    pd.DataFrame({(y, s): G[(y, s)] for y in R.YEARS for s in ("VE", "MM", "MA")},
                 index=R.C).to_csv(out / "table9_grouping_normalisation.csv")
    pd.DataFrame({(y, s): G[(y, s)] for y in R.YEARS for s in ("Gt", "C", "K")},
                 index=R.C).to_csv(out / "table11_grouping_methods.csv")
    for name, keys in [("table10_grouping_vmeasure", ("VE", "MM", "MA")),
                       ("table12_grouping_vmeasure", ("Gt", "C", "K"))]:
        pd.DataFrame([dict(Year=y, Pair=f"{a} vs {b}",
                           V_measure=round(float(v_measure_score(G[(y, a)], G[(y, b)])), 3))
                      for y in R.YEARS for a, b in combinations(keys, 2)]).to_csv(out / f"{name}.csv", index=False)
    q = pd.DataFrame(quality)
    q.to_csv(out / "clustering_quality_runs.csv", index=False)
    summary = q.groupby("year").mean(numeric_only=True).drop(columns="run").round(3)
    summary.to_csv(out / "clustering_quality.csv")
    print(f"[{dataset}] Table 2 (groups)")
    print(pd.read_csv(out / "table2_groups.csv", index_col=0).to_string())
    print(f"\n[{dataset}] clustering quality (mean over {n_runs} runs)")
    print(summary.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "calibrated")
