"""Batched versions of the v0 kernels: X has shape (B, m, n).  Used by fit_data.py to evaluate
all finite-difference perturbations of the Jacobian in one numpy call."""
from __future__ import annotations

import numpy as np

COST = np.array([0, 1, 2])


def _flip(benefit, cost_val):
    out = benefit.copy()
    out[..., COST] = cost_val[..., COST]
    return out


def linear(X):
    mn, mx = X.min(1, keepdims=True), X.max(1, keepdims=True)
    return _flip((X - mn) / (mx - mn), (mx - X) / (mx - mn))


def vector(X):
    b = X / np.sqrt((X ** 2).sum(1, keepdims=True))
    inv = 1 / X
    c = inv / np.sqrt((inv ** 2).sum(1, keepdims=True))
    return _flip(b, c)


def maxn(X):
    b = X / X.max(1, keepdims=True)
    return _flip(b, 1 - b)


def ratio(X):
    return _flip(X / X.max(1, keepdims=True), X.min(1, keepdims=True) / X)


def psi(X):
    r = ratio(X)
    d = np.abs(r - r.mean(1, keepdims=True)).mean(1)
    return d / d.sum(1, keepdims=True)


def critic(X):
    nd = ratio(X)
    sd = nd.std(1)
    c = nd - nd.mean(1, keepdims=True)
    cov = np.einsum("bmi,bmj->bij", c, c) / X.shape[1]
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    cj = sd * (1 - corr).sum(2)
    return cj / cj.sum(1, keepdims=True)


def entropy(X):
    p = X / X.sum(1, keepdims=True)
    e = -(p * np.log(p)).sum(1) / np.log(X.shape[1])
    return (1 - e) / (1 - e).sum(1, keepdims=True)


def aroman(W):
    return W[..., :3].sum(2) + 0.5 * W[..., 3:].sum(2)


def scores(X, method="v0"):
    lin = linear(X)
    if method in ("v0", "CRITIC", "Entropy"):
        N = (0.5 * lin + 0.5 * vector(X)) / 2
    elif method == "MinMax":
        N = (0.5 * lin + 0.5 * lin) / 2
    elif method == "Max":
        N = (0.5 * lin + 0.5 * maxn(X)) / 2
    else:
        raise ValueError(method)
    w = {"CRITIC": critic, "Entropy": entropy}.get(method, psi)(X)
    return aroman(N * w[:, None, :])
