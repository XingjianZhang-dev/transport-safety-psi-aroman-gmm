"""Digitise Fig. 8 ("De-construction of composite index", Year 2023).

Hypothesis tested here: every segment is the VECTOR-normalised value x''_ij (Eqs. 4-5) of the
2023 decision matrix.  Under that hypothesis each criterion column satisfies
sum_i x''_ij^2 = 1, so the pixel scale can be self-calibrated independently from each of
the nine colours; nine agreeing scales = hypothesis confirmed.
Because the whole v0 pipeline is invariant to per-column rescaling, x'' alone is enough to
recompute the paper's 2023 ranking (benefit: x ~ x'', cost: x ~ 1/x'').
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

C = ["AR", "AU", "BR", "CA", "CH", "FR", "DE", "IN", "ID", "IT",
     "JP", "KR", "MX", "RU", "SA", "ZA", "TR", "GB", "US"]
LEG = ["A11", "A21", "A31", "B12", "B13", "C41", "C42", "C43", "C44"]
# legend label -> paper indicator (Fig. 1 naming), from the benchmarking text of Sec. 6.4
MAP = {"A11": "A31", "A21": "A32", "A31": "A33", "B12": "B21", "B13": "B22",
       "C41": "C41", "C42": "C42", "C43": "C43", "C44": "C44"}


def legend_colours(img):
    """Sample the 9 legend patches (right-hand box)."""
    H, W, _ = img.shape
    reg = img[:, int(W * 0.91):int(W * 0.96)].astype(int)
    sat = reg.max(2) - reg.min(2)
    grey = (np.abs(reg[..., 0] - reg[..., 1]) < 8) & (np.abs(reg[..., 1] - reg[..., 2]) < 8) \
        & (reg[..., 0] > 100) & (reg[..., 0] < 160)
    rowmask = ((sat > 60) | grey).sum(1) > 15
    lab, n = ndi.label(rowmask)
    cols = []
    for i in range(1, n + 1):
        rows = np.nonzero(lab == i)[0]
        if len(rows) < 8:
            continue
        band = reg[rows[len(rows) // 2]]
        s = band.max(1) - band.min(1)
        g = (np.abs(band[:, 0] - band[:, 1]) < 8) & (band[:, 0] > 100) & (band[:, 0] < 160)
        pix = band[(s > 60) | g]
        cols.append(np.median(pix, axis=0))
    return np.array(cols[:9])


def digitise(path: Path):
    img = np.array(Image.open(path).convert("RGB")).astype(int)
    H, W, _ = img.shape
    pal = legend_colours(img)
    assert len(pal) == 9, f"found {len(pal)} legend colours"
    # classify every pixel of the plotting area to the nearest palette colour
    plot = img[:, :int(W * 0.905)]
    d = np.sqrt(((plot[:, :, None, :] - pal[None, None]) ** 2).sum(-1))
    cls = d.argmin(-1)
    ok = d.min(-1) < 35
    colmask = ok.sum(0) > 40            # a bar column holds >40 classified pixels
    # bars = runs of columns containing classified pixels
    lab, n = ndi.label(ndi.binary_opening(colmask, iterations=2))
    bars = [np.nonzero(lab == i)[0] for i in range(1, n + 1)]
    bars = [b for b in bars if len(b) > 20]
    assert len(bars) == 19, f"found {len(bars)} bars"
    heights = np.zeros((19, 9))
    base = []
    for bi, b in enumerate(bars):
        xs = b[len(b) // 4: 3 * len(b) // 4]          # central half of the bar
        for k in range(9):
            heights[bi, k] = np.median(((cls[:, xs] == k) & ok[:, xs]).sum(0))
        rows = np.nonzero(ok[:, xs].any(1))[0]
        base.append(rows.max())
    # self-calibration: sum_i x''^2 = 1 per column  ->  px per unit = ||h_col||
    scales = np.sqrt((heights ** 2).sum(0))
    scale = float(np.median(scales))
    vals = heights / scale
    return dict(palette=pal.tolist(), bar_x=[float(b.mean()) for b in bars],
                baseline_px=float(np.median(base)), scales_px_per_unit=scales.tolist(),
                scale=scale, x_vector=vals.tolist(), totals=vals.sum(1).tolist())


if __name__ == "__main__":
    res = digitise(Path(sys.argv[1]))
    out = Path(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    sc = np.array(res["scales_px_per_unit"])
    print("per-criterion self-calibrated scale (px/unit):", np.round(sc, 1),
          f" CV={sc.std() / sc.mean():.3%}")
    v = np.array(res["x_vector"])
    import pandas as pd
    print(pd.DataFrame(v, index=C, columns=[f"{k}({MAP[k]})" for k in LEG]).round(3).to_string())
