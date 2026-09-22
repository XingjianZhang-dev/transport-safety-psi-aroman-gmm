# PSI–AROMAN–GMM with t-SNE — Reproduction Package

Reproduction package for:

> Xingjian Zhang, Nanbo (Aaron) Zhang, Jialin Li, Qintao Li, Xingze Liu, Chuanpu (Lukas) Cao, Hao Mao,
> Ruikang Yan, Yunlong Qi, Xinyi (Chenny) Yang, Jialun Li, Aaron Kaiqiang Zhou, Xu Yan, Hanrui Feng,
> Faan Chen. **Machine learning nested MCDM model to enhance decision reliability for transport safety
> engineering.** *Results in Engineering* 29 (2026) 108543.
> DOI: [10.1016/j.rineng.2025.108543](https://doi.org/10.1016/j.rineng.2025.108543)

This repository provides an end-to-end, single-command pipeline that rebuilds the case study of the
article: the decision matrices of the G20 countries for the waves 2010, 2015, 2019 and 2023, the
PSI–AROMAN composite scores and rankings, the multilevel robustness analyses, the GMM-with-t-SNE
grouping, and the figures of the paper.

The calibrated decision matrices shipped with the package reproduce every composite score of Table 1
to its printed three decimals and the complete ranking of all four waves.

---

## 1. Model

The implemented model is the PSI–AROMAN–GMM with t-SNE pipeline of Section 4 of the article.

| Step | Operation | Implementation |
|---|---|---|
| 1 | Decision matrix (19 countries × 9 safety performance indicators) | `src/build_dataset.py` |
| 2 | Linear normalisation (Eqs. 2–3), vector normalisation (Eqs. 4–5), aggregated averaged normalisation with β = 0.5 (Eq. 6) | `mcdm.linear_norm`, `mcdm.vector_norm`, `mcdm.aggregate` |
| 3 | PSI criteria weights: preference variation values (Eqs. 7–9), deviation of the preference value (Eq. 10), normalised weights (Eq. 11) | `mcdm.psi` |
| 4 | Weighted aggregation of the normalised indicators (Eq. 12) | `mcdm.run_method` |
| 5 | AROMAN composite score with λ = 0.5 and ranking (Eqs. 13–14) | `mcdm.aroman`, `mcdm.ranks_from_scores` |
| 6–9 | t-SNE embedding (2 components, perplexity 5), Gaussian mixture model with four components, cluster visualisation | `src/run_clustering.py` |

Robustness analyses:

| Analysis | Variants | Article table |
|---|---|---|
| Initial sensitivity | vector, min–max and max normalisation | Tables 3–4 |
| Medial stability | PSI, CRITIC and Shannon-entropy weighting | Tables 5–6 |
| Lateral reliability | AROMAN, COPRAS and PROMETHEE aggregation | Tables 7–8 |
| Grouping sensitivity | three normalisations; k-means and fuzzy c-means benchmarks | Tables 9–12 |

Clusters are numbered by ascending cluster mean of A31 (road fatalities per 100,000 inhabitants),
so Group 1 is the best-performing cluster.

## 2. Index system

| Code | Safety performance indicator | Type |
|---|---|---|
| A31 | Road fatalities per 100,000 inhabitants | cost |
| A32 | Road fatalities per 10,000 registered vehicles | cost |
| A33 | Change in the number of road deaths (%) | cost |
| B21 | Seatbelt wearing rates in front seats (%) | benefit |
| B22 | Seatbelt wearing rates in rear seats (%) | benefit |
| C41 | Enforcement score on speed limit law | benefit |
| C42 | Enforcement score on drink-driving law | benefit |
| C43 | Enforcement score on seat-belt law | benefit |
| C44 | Enforcement score on helmet use law | benefit |

Alternatives are the 19 G20 member countries, ordered as in Table 1 of the article: AR, AU, BR, CA,
CH, FR, DE, IN, ID, IT, JP, KR, MX, RU, SA, ZA, TR, GB, US.

## 3. Data

The decision matrices are reconstructed from the figures of the article and then calibrated against
its published composite scores.

**Reconstruction** (`digitize/digitize_radar.py`). The nine radar charts of Fig. 7 carry the raw
indicator values of all countries and waves. The digitiser fits the polar frame of each panel,
calibrates the radial scale on the concentric grid rings (residual ≈ 0.15 px), separates the four
waves by marker colour, and converts every marker to a value along its country axis. Enforcement
scores, which are integer ratings, are snapped to integers; cells that equal the mean of the
remaining cells of their column are recomputed as that mean. Each cell carries a provenance flag in
`data/quality_flags_{year}.csv` (`ok`, `int`, `mean`, `occluded`, `centre`).

**Cross-check** (`digitize/digitize_stacked.py`). Fig. 8 shows the vector-normalised 2023 matrix.
Because vector normalisation satisfies Σᵢ x″ᵢⱼ² = 1, the pixel scale of that figure is
self-calibrating from each of the nine criteria independently; the nine estimates agree to about
1 %, and the resulting matrix reproduces the 2023 reconstruction.

**Calibration** (`src/calibrate.py`). Writing each cell as *x* = *x*<sub>rec</sub> + σ·*z*, where σ is
the read-off scale implied by the provenance flag, the calibration solves

```
minimise    Σ z²
subject to  |Rᵢ(X) − Rᵢ(published)| ≤ 0.00049   for all 19 countries
            Rᵢ(X) > Rⱼ(X)  whenever i outranks j in Table 1
            0 ≤ x ≤ upper bounds (100 % for seatbelt rates, 10 points for enforcement scores)
```

with SLSQP and batched finite-difference constraint Jacobians (`src/mcdm_batch.py`). Integer and
mean-substituted enforcement cells are held fixed. The calibrated matrices are written to
`data/calibrated/`.

## 4. Repository layout

```
digitize/
  digitize_radar.py      reconstruct the raw indicator values from the Fig. 7 radar charts
  digitize_stacked.py    reconstruct the vector-normalised 2023 matrix from Fig. 8
  out/                   digitiser output and calibration metadata
data/
  decision_matrix_{year}.csv     reconstructed decision matrices
  quality_flags_{year}.csv       per-cell provenance flags
  clustering_input_{year}.csv    decision matrices with country names and full indicator names
  calibrated/                    calibrated decision matrices (used for the reported results)
src/
  reference_values.py    published composite scores, rankings and groups; index metadata
  mcdm.py                normalisations, PSI / CRITIC / entropy weights, AROMAN / COPRAS / PROMETHEE
  mcdm_batch.py          vectorised kernels used for the calibration Jacobians
  build_dataset.py       assemble the decision matrices from the digitiser output
  calibrate.py           constrained calibration against the published composite scores
  run_ranking.py         Tables 1, 3–8
  run_clustering.py      Tables 2, 9–12 and clustering-quality metrics
  make_figures.py        Figs. 3–9 and 11
results/
  calibrated/, reconstructed/    tables, figures and clustering output per dataset
figures/                 source figures of the article used by the digitiser
run_all.sh               full pipeline
```

## 5. Installation and use

```bash
git clone https://github.com/XingjianZhang-dev/transport-safety-psi-aroman-gmm.git
cd transport-safety-psi-aroman-gmm
pip install -r requirements.txt
bash run_all.sh
```

The full pipeline runs in about ten minutes on a single core. Individual stages can be run on their
own, for example:

```bash
python3 src/run_ranking.py calibrated       # Tables 1, 3-8
python3 src/run_clustering.py calibrated    # Tables 2, 9-12
python3 src/make_figures.py calibrated      # Figs. 3-9, 11
```

Both `calibrated` and `reconstructed` are accepted as the dataset argument.

## 6. Outputs

| Article item | File |
|---|---|
| Table 1 — composite scores and rankings | `results/calibrated/table1_composite_scores.csv` |
| Table 2 — groups | `results/calibrated/table2_groups.csv` |
| Table 3 — rankings under three normalisations | `results/calibrated/table3_initial_sensitivity.csv` |
| Table 4 — Spearman coefficients | `results/calibrated/table4_initial_sensitivity_spearman.csv` |
| Table 5 — rankings under three weighting methods | `results/calibrated/table5_medial_stability.csv` |
| Table 6 — Spearman coefficients | `results/calibrated/table6_medial_stability_spearman.csv` |
| Table 7 — rankings under three aggregation methods | `results/calibrated/table7_lateral_reliability.csv` |
| Table 8 — Spearman coefficients | `results/calibrated/table8_lateral_reliability_spearman.csv` |
| Table 9 — groups under three normalisations | `results/calibrated/table9_grouping_normalisation.csv` |
| Table 10 — V-measure between groupings | `results/calibrated/table10_grouping_vmeasure.csv` |
| Table 11 — groups under three clustering methods | `results/calibrated/table11_grouping_methods.csv` |
| Table 12 — V-measure between clustering methods | `results/calibrated/table12_grouping_vmeasure.csv` |
| Criteria weights | `results/calibrated/criteria_weights.csv` |
| Clustering quality (silhouette, Davies–Bouldin over 50 runs) | `results/calibrated/clustering_quality.csv` |
| Fig. 3 — GMM clustering with t-SNE | `results/calibrated/figures/fig3_gmm_clustering_with_tsne.png` |
| Fig. 4 — ranking under different normalisations | `results/calibrated/figures/fig4_ranking_normalisation.png` |
| Fig. 5 — ranking under different weighting methods | `results/calibrated/figures/fig5_ranking_weighting.png` |
| Fig. 6 — ranking under different aggregation methods | `results/calibrated/figures/fig6_ranking_aggregation.png` |
| Fig. 7 — dynamic changes of the indicators | `results/calibrated/figures/fig7_dynamic_changes_of_spis.png` |
| Fig. 8 — de-construction of the composite index | `results/calibrated/figures/fig8_deconstruction_of_composite_index.png` |
| Fig. 9 — de-composition of score changes, 2010–2023 | `results/calibrated/figures/fig9_decomposition_of_score_changes.png` |
| Fig. 11 — benchmarking of indicators within each group | `results/calibrated/figures/fig11_benchmarking_within_groups.png` |

## 7. Environment

Developed and tested with Python 3.12, numpy 2.4, pandas 3.0, scipy 1.17, scikit-learn 1.8,
scikit-fuzzy 0.5, matplotlib 3.10 and pillow 12.1. Version floors are listed in
`requirements.txt`. t-SNE embeddings depend on the scikit-learn version; the grouping stage fixes
`random_state = 42` throughout.

## 8. Citation

```bibtex
@article{Zhang2026PSIAROMANGMM,
  title   = {Machine learning nested {MCDM} model to enhance decision reliability
             for transport safety engineering},
  author  = {Zhang, Xingjian and Zhang, Nanbo and Li, Jialin and Li, Qintao and Liu, Xingze
             and Cao, Chuanpu and Mao, Hao and Yan, Ruikang and Qi, Yunlong and Yang, Xinyi
             and Li, Jialun and Zhou, Aaron Kaiqiang and Yan, Xu and Feng, Hanrui and Chen, Faan},
  journal = {Results in Engineering},
  volume  = {29},
  pages   = {108543},
  year    = {2026},
  doi     = {10.1016/j.rineng.2025.108543}
}
```

## 9. License

Code in this repository is released under the MIT License (`LICENSE`).

The figure images in `figures/` are reproduced from the article, which is published open access
under [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/); they remain under that licence
and are attributed to the authors of the article.
