"""Kernels of the PSI-AROMAN-GMM with t-SNE model (Section 4 of the article).

Implemented here
  * linear, vector and max normalisation, and the aggregated averaged normalisation (Eqs. 2-6)
  * PSI criteria weights (Eqs. 7-11), and the CRITIC and Shannon-entropy weights used in the
    medial-stability analysis
  * AROMAN aggregation (Eqs. 12-14), and the COPRAS and PROMETHEE aggregators used in the
    lateral-reliability analysis

All functions take X (m alternatives x n criteria, raw values) and `cost`, the list of
zero-based cost-criterion columns ([0, 1, 2] = A31, A32, A33).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path("/home/claude/g20_transport_safety_mcdm")


# ============================================================================ normalisations
def linear_norm(X, cost):
    """Eqs. 2-3."""
    X = np.asarray(X, float)
    mn, mx = X.min(0), X.max(0)
    out = (X - mn) / (mx - mn)
    for j in cost:
        out[:, j] = (mx[j] - X[:, j]) / (mx[j] - mn[j])
    return out


def vector_norm(X, cost):
    """Eqs. 4-5: benefit -> x / ||x||, cost -> (1/x) / ||1/x||."""
    X = np.asarray(X, float)
    out = X / np.sqrt((X ** 2).sum(0))
    for j in cost:
        out[:, j] = (1 / X[:, j]) / np.sqrt(np.sum(1 / X[:, j] ** 2))
    return out


def max_norm(X, cost):
    """Max normalisation: benefit x/max, cost 1 - x/max."""
    X = np.asarray(X, float)
    out = X / X.max(0)
    for j in cost:
        out[:, j] = 1 - X[:, j] / X[:, j].max()
    return out


def aggregate(n1, n2, beta=0.5):
    """Eq. 6: (beta*x' + (1-beta)*x'') / 2."""
    return (beta * n1 + (1 - beta) * n2) / 2


def normalised_matrix(X, cost, scheme="vector", beta=0.5):
    """Aggregated averaged normalisation used by every PSI-AROMAN variant.
    scheme = second normaliser: 'vector' (proposed), 'minmax', 'max' (Table 3)."""
    n1 = linear_norm(X, cost)
    if scheme == "vector":
        n2 = vector_norm(X, cost)
    elif scheme == "minmax":
        n2 = linear_norm(X, cost)
    elif scheme == "max":
        n2 = max_norm(X, cost)
    else:
        raise ValueError(scheme)
    return aggregate(n1, n2, beta)


# ============================================================================ weights
def _pv_ratio(X, cost):
    """Eqs. 7-8: r = x_min/x (cost) or x/x_max (benefit)."""
    X = np.asarray(X, float)
    r = X / X.max(0)
    for j in cost:
        r[:, j] = X[:, j].min() / X[:, j]
    return r


def psi(X, cost):
    """PSI weights (Eqs. 7-11): preference-variation ratios, deviation of the preference value
    taken as the mean absolute deviation, normalised to unit sum."""
    r = _pv_ratio(X, cost)
    dpv = np.mean(np.abs(r - r.mean(0)), axis=0)
    return dpv / dpv.sum()



def critic(X, cost):
    """CRITIC weights: ratio normalisation, population standard deviation and Pearson
    correlation, C_j = sigma_j * sum_k (1 - r_jk)."""
    nd = _pv_ratio(X, cost)
    sd = np.std(nd, axis=0)
    corr = np.corrcoef(nd.T)
    cj = sd * np.sum(1 - corr, axis=1)
    return cj / cj.sum()


def entropy(X):
    """Shannon-entropy weights."""
    X = np.asarray(X, float)
    m, n = X.shape
    p = X / X.sum(0)
    k = 1 / np.log(m)
    e = np.array([-k * np.sum(p[p[:, j] > 0, j] * np.log(p[p[:, j] > 0, j])) for j in range(n)])
    w = (1 - e) / (1 - e).sum()
    return w


# ============================================================================ aggregators
def aroman(W, cost, lam=0.5):
    """Step 5 (Eq. 14): R_i = L_i + lambda * A_i."""
    n = W.shape[1]
    ben = [j for j in range(n) if j not in cost]
    return W[:, cost].sum(1) + lam * W[:, ben].sum(1)




def copras(W, cost):
    """COPRAS: Q_i = P_i + (R_min * sum R) / (R_i * m), reported as U_i = Q_i / Q_max * 100."""
    m, n = W.shape
    ben = [j for j in range(n) if j not in cost]
    P = W[:, ben].sum(1)
    R = W[:, cost].sum(1)
    Q = P + (R.min() * R.sum()) / (R * m)
    return Q / Q.max() * 100


def promethee(X, cost, w):
    """PROMETHEE II on the vector-normalised matrix with a linear preference function."""
    N = vector_norm(X, cost)
    m, n = N.shape
    d = N[:, None, :] - N[None, :, :]
    P = np.clip(d, 0, 1)
    pi = (P * w).sum(2)
    return pi.mean(1) - pi.mean(0)


# ============================================================================ pipelines
def run_method(X, cost, method):
    """Return (scores, weights) for every method column that appears in Tables 1/3/5/7."""
    X = np.asarray(X, float)
    if method in ("Vector", "PSI", "AROMAN"):                   # the proposed model
        N = normalised_matrix(X, cost, "vector"); w = psi(X, cost)
        return aroman(N * w, cost), w
    if method == "MinMax":
        N = normalised_matrix(X, cost, "minmax"); w = psi(X, cost)
        return aroman(N * w, cost), w
    if method == "Max":
        N = normalised_matrix(X, cost, "max"); w = psi(X, cost)
        return aroman(N * w, cost), w
    if method == "CRITIC":
        N = normalised_matrix(X, cost, "vector"); w = critic(X, cost)
        return aroman(N * w, cost), w
    if method == "Entropy":
        N = normalised_matrix(X, cost, "vector"); w = entropy(X)
        return aroman(N * w, cost), w
    if method == "COPRAS":
        N = normalised_matrix(X, cost, "vector"); w = psi(X, cost)
        return copras(N * w, cost), w
    if method == "PROMETHEE":
        w = psi(X, cost)
        return promethee(X, cost, w), w
    raise ValueError(method)


def ranks_from_scores(s):
    """Rank 1 = highest composite score."""
    s = np.asarray(s, float)
    order = s.argsort()[::-1]
    r = np.empty(len(s), int)
    r[order] = np.arange(1, len(s) + 1)
    return r
