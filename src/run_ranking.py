"""Rankings of the G20 countries: Tables 1, 3-8 of the article.

Usage:  python3 src/run_ranking.py [calibrated|reconstructed]

Outputs (results/<dataset>/):
  criteria_weights.csv                    PSI, CRITIC and entropy weights per year
  table1_composite_scores.csv             proposed PSI-AROMAN model, scores and ranks
  table3_initial_sensitivity.csv          ranks under vector, min-max and max normalisation
  table4_initial_sensitivity_spearman.csv
  table5_medial_stability.csv             ranks under PSI, CRITIC and entropy weighting
  table6_medial_stability_spearman.csv
  table7_lateral_reliability.csv          ranks under AROMAN, COPRAS and PROMETHEE aggregation
  table8_lateral_reliability_spearman.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import mcdm
import reference_values as R

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = {"table3_initial_sensitivity": ("Vector", "MinMax", "Max"),
          "table5_medial_stability": ("PSI", "CRITIC", "Entropy"),
          "table7_lateral_reliability": ("AROMAN", "COPRAS", "PROMETHEE")}
SPEARMAN_NAME = {"table3_initial_sensitivity": "table4_initial_sensitivity_spearman",
                 "table5_medial_stability": "table6_medial_stability_spearman",
                 "table7_lateral_reliability": "table8_lateral_reliability_spearman"}


def load(dataset, year):
    sub = "calibrated/" if dataset == "calibrated" else ""
    return pd.read_csv(ROOT / f"data/{sub}decision_matrix_{year}.csv", index_col=0).loc[R.C]


def main(dataset="calibrated"):
    out = ROOT / "results" / dataset
    out.mkdir(parents=True, exist_ok=True)
    scores, ranks, weights = {}, {}, {}
    for y in R.YEARS:
        X = load(dataset, y).values
        for m in ["Vector", "MinMax", "Max", "CRITIC", "Entropy", "COPRAS", "PROMETHEE"]:
            s, w = mcdm.run_method(X, R.COST, m)
            scores[(y, m)], ranks[(y, m)] = s, mcdm.ranks_from_scores(s)
            if m in ("Vector", "CRITIC", "Entropy"):
                weights[(y, {"Vector": "PSI"}.get(m, m))] = w
        scores[(y, "PSI")] = scores[(y, "AROMAN")] = scores[(y, "Vector")]
        ranks[(y, "PSI")] = ranks[(y, "AROMAN")] = ranks[(y, "Vector")]

    pd.DataFrame(weights, index=R.INDICATORS).T.rename_axis(["Year", "Method"]).to_csv(out / "criteria_weights.csv")
    t1 = pd.DataFrame({(y, k): (scores[(y, "Vector")] if k == "Score" else ranks[(y, "Vector")])
                       for y in R.YEARS for k in ("Score", "Rank")}, index=R.C)
    t1.round(3).to_csv(out / "table1_composite_scores.csv")

    for name, methods in BLOCKS.items():
        pd.DataFrame({(y, m): ranks[(y, m)] for y in R.YEARS for m in methods},
                     index=R.C).to_csv(out / f"{name}.csv")
        rows = [dict(Year=y, Pair=f"{a} vs {b}",
                     Spearman=round(float(spearmanr(ranks[(y, a)], ranks[(y, b)]).statistic), 3))
                for y in R.YEARS for a, b in combinations(methods, 2)]
        pd.DataFrame(rows).to_csv(out / f"{SPEARMAN_NAME[name]}.csv", index=False)

    print(f"[{dataset}] Table 1 (composite score / rank)")
    print(t1.round(3).to_string())
    for name in BLOCKS:
        print(f"\n[{dataset}] {SPEARMAN_NAME[name]}")
        print(pd.read_csv(out / f"{SPEARMAN_NAME[name]}.csv").to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "calibrated")
