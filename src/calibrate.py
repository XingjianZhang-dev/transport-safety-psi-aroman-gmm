"""Calibrate the reconstructed decision matrices against the published composite scores.

The reconstruction returns each cell with a read-off scale sigma_ij that follows from how the
cell was recovered from the radar chart (see data/quality_flags_*.csv).  Writing
x_ij = x_ij^rec + sigma_ij * z_ij, the calibration solves

    minimise   sum_ij z_ij^2
    subject to |R_i(X) - R_i^published| <= 0.00049        (printed to three decimals)
               R_i(X) > R_j(X) whenever country i outranks country j in Table 1
               0 <= x <= upper bounds (100 % for seatbelt rates, 10 points for enforcement);
               enforcement cells recovered as integers or as column means are held fixed.

SLSQP is used; the constraint Jacobian is obtained from the batched kernels in mcdm_batch.py.
Output: data/calibrated/decision_matrix_{year}.csv and results/calibration_report.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import mcdm_batch as MB
import reference_values as R

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.00049
UPPER = np.array([np.inf, np.inf, np.inf, 100, 100, 10, 10, 10, 10], float)


def read_off_sigma(X, F, px):
    S = np.zeros_like(X)
    for j in range(9):
        for i in range(19):
            f = F.iloc[i, j]
            S[i, j] = {"ok": 1.0 * px[j], "occluded": 2.5 * px[j]}.get(f, 0.02)
            if f == "centre":
                S[i, j] = max(0.5 * X[i, j], 2 * px[j])
    return S


def calibrate(year, px, maxiter=1000):
    X = pd.read_csv(ROOT / f"data/decision_matrix_{year}.csv", index_col=0).loc[R.C]
    F = pd.read_csv(ROOT / f"data/quality_flags_{year}.csv", index_col=0).loc[R.C]
    Xd = X.values.astype(float)
    S = read_off_sigma(Xd, F, px)
    xd, sf, n = Xd.ravel(), S.ravel(), Xd.size
    s_ref = R.T1_SCORE[year].values
    better, worse = np.argsort(R.T1_RANK[year].values)[:-1], np.argsort(R.T1_RANK[year].values)[1:]

    def batch(Z):
        s = MB.scores((xd + sf * Z).reshape(-1, 19, 9), "v0")
        return np.concatenate([s - (s_ref - TOL), (s_ref + TOL) - s,
                               (s[:, better] - s[:, worse]) / np.abs(s).mean(1, keepdims=True) - 1e-6], axis=1)

    def c_fun(z):
        return batch(z[None])[0]

    def c_jac(z, h=1e-7):
        Z = np.tile(z, (n + 1, 1))
        Z[1:][np.diag_indices(n)] += h
        C = batch(Z)
        return ((C[1:] - C[0]) / h).T

    lb = np.full_like(Xd, 1e-2); lb[:, 5:] = 0.5
    ub = np.tile(UPPER, (19, 1)); ub[:, :3] = 5 * Xd[:, :3]
    fixed = np.isin(F.values, ["int", "mean"])
    lb, ub = np.where(fixed, Xd - 0.05, lb), np.where(fixed, Xd + 0.05, ub)
    res = minimize(lambda z: 0.5 * z @ z, np.zeros(n), jac=lambda z: z, method="SLSQP",
                   bounds=list(zip((lb.ravel() - xd) / sf, (ub.ravel() - xd) / sf)),
                   constraints=[dict(type="ineq", fun=c_fun, jac=c_jac)],
                   options=dict(maxiter=maxiter, ftol=1e-12))
    Xc = (xd + sf * res.x).reshape(19, 9)
    return Xc, res


if __name__ == "__main__":
    import json
    meta = json.loads((ROOT / "digitize/out/fig7_meta.json").read_text())
    px = np.array([1 / meta[k]["calib_a"] for k in R.INDICATORS])
    out = ROOT / "data/calibrated"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in R.YEARS:
        Xc, res = calibrate(y, px)
        pd.DataFrame(Xc, index=R.C, columns=R.INDICATORS).rename_axis("Code").to_csv(
            out / f"decision_matrix_{y}.csv", float_format="%.6f")
        s = MB.scores(Xc[None], "v0")[0]
        rows.append(dict(year=y, iterations=res.nit,
                         max_abs_score_error=float(np.abs(s - R.T1_SCORE[y].values).max()),
                         scores_matching_3dp=int((np.round(s, 3) == R.T1_SCORE[y].values).sum())))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/calibration_report.csv", index=False)
