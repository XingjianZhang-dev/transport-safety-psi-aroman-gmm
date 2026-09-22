"""Digitize the nine radar charts of Fig. 7 (raw SPI values, 19 countries x 4 years).

Pipeline per sub-plot
  1. locate the outer (black) polar frame -> centre (cx, cy) and frame radius
  2. locate the grey concentric grid rings -> linear pixel->value calibration
  3. isolate each year's star markers by tab10 colour, erode away the thin lines
  4. assign each marker blob to the nearest of the 19 country axes, convert to value
Occluded markers (a later year drawn exactly on top) are flagged and filled with the
value of the occluding marker, since identical position means identical value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

COUNTRIES = ["AR", "AU", "BR", "CA", "CH", "FR", "DE", "IN", "ID", "IT",
             "JP", "KR", "MX", "RU", "SA", "ZA", "TR", "GB", "US"]
YEARS = [2010, 2015, 2019, 2023]
# tab10 colours as they appear in the PDF raster (after ICC conversion), sampled from
# the most saturated legend-marker pixels
COLORS = {2010: (43, 114, 164), 2015: (231, 132, 34),
          2019: (60, 152, 54), 2023: (189, 46, 46)}
# tick values printed on each sub-plot (read from the figure)
TICKS = {
    "A31": [5, 10, 15, 20, 25, 30],
    "A32": [2.5, 5, 7.5, 10, 12.5, 15, 17.5],
    "A33": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
    "B21": [20, 40, 60, 80, 100],
    "B22": [20, 40, 60, 80, 100],
    "C41": [2, 4, 6, 8],
    "C42": [2, 4, 6, 8],
    "C43": [2, 4, 6, 8, 10],
    "C44": [2, 4, 6, 8, 10],
}
PANELS = ["A31", "A32", "A33", "B21", "B22", "C41", "C42", "C43", "C44"]
N_AX = len(COUNTRIES)
AX_ANG = np.arange(N_AX) * 2 * np.pi / N_AX  # AR at 0 rad, counter-clockwise


def tiles(img: np.ndarray):
    H, W, _ = img.shape
    xs = [0, W // 3, 2 * W // 3, W]
    ys = [0, H // 3, 2 * H // 3, H]
    for r in range(3):
        for c in range(3):
            yield PANELS[3 * r + c], (xs[c], ys[r]), img[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]


def find_frame(tile: np.ndarray):
    """Coarse-to-fine search for the circle that best explains the dark pixels."""
    dark = np.all(tile < 90, axis=2)
    yy, xx = np.nonzero(dark)
    h, w = dark.shape
    best = None
    cands = [(cx, cy) for cx in np.arange(w * 0.35, w * 0.60, 4)
             for cy in np.arange(h * 0.40, h * 0.65, 4)]
    def score(cx, cy):
        d = np.hypot(xx - cx, yy - cy)
        hist, edges = np.histogram(d, bins=np.arange(270, 300, 1.0))
        k = hist.argmax()
        return hist[k], edges[k] + 0.5
    for cx, cy in cands:
        s, r = score(cx, cy)
        if best is None or s > best[0]:
            best = (s, cx, cy, r)
    _, cx, cy, r = best
    for step in (2, 1, 0.5, 0.25):
        improved = True
        while improved:
            improved = False
            for dx in (-step, 0, step):
                for dy in (-step, 0, step):
                    s, rr = score(cx + dx, cy + dy)
                    if s > best[0]:
                        best = (s, cx + dx, cy + dy, rr)
                        cx, cy, r = cx + dx, cy + dy, rr
                        improved = True
    # refine radius as mean distance of dark pixels within +-3 px of the peak
    d = np.hypot(xx - cx, yy - cy)
    sel = np.abs(d - r) < 3
    return cx, cy, float(d[sel].mean())


def find_rings(tile, cx, cy, R):
    """Radial darkness profile averaged over angles that avoid the axis spokes."""
    g = tile.astype(float).mean(axis=2)
    angs = np.concatenate([AX_ANG + (k / 8) * 2 * np.pi / N_AX for k in range(2, 7)])
    rs = np.arange(3, R - 2, 0.5)
    prof = np.zeros_like(rs)
    for a in angs:
        x = cx + rs * np.cos(a)
        y = cy - rs * np.sin(a)
        v = ndi.map_coordinates(g, [y, x], order=1)
        base = ndi.median_filter(v, size=15)
        prof += np.clip(base - v, 0, None)
    prof /= len(angs)
    # peaks
    thr = 4 * np.median(prof) + 1
    pk = [i for i in range(3, len(rs) - 3)
          if prof[i] == prof[i - 3:i + 4].max() and prof[i] > thr]
    # merge peaks closer than 6 px
    peaks = []
    for i in pk:
        if peaks and rs[i] - peaks[-1][0] < 6:
            if prof[i] > peaks[-1][1]:
                peaks[-1] = (rs[i], prof[i])
        else:
            peaks.append((rs[i], prof[i]))
    return [p for p, _ in peaks], rs, prof


def calibrate(rs, prof, R, ticks):
    """Constrained calibration r_px = a * value (matplotlib polar axes start at r = 0).

    Search the scale a that maximises the mean ring-darkness profile at the radii of the
    printed tick values (only ticks strictly inside the frame are scored, because the
    outermost tick can coincide with the frame).  Then refine each ring to its local
    profile maximum (+-4 px) and re-fit a by least squares through the origin.
    """
    t = np.array(ticks, dtype=float)
    best = None
    for a in np.arange(R / t.max() * 0.80, (R + 2) / t.max(), 0.002):
        rr = a * t
        inside = rr < R - 3
        if inside.sum() < 2:
            continue
        sc = np.interp(rr[inside], rs, prof).mean()
        if best is None or sc > best[0]:
            best = (sc, a)
    a0 = best[1]
    refined = []
    for v in t:
        r0 = a0 * v
        if r0 >= R - 3:
            continue
        w = (rs > r0 - 4) & (rs < r0 + 4)
        refined.append((v, rs[w][np.argmax(prof[w])]))
    v, r = np.array(refined).T
    a = float((v * r).sum() / (v * v).sum())
    rms = float(np.sqrt(np.mean((r - a * v) ** 2)))
    return a, 0.0, rms, r


def marker_blobs(tile, color, tol=48, erode=1):
    diff = tile.astype(int) - np.array(color)[None, None, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    mask = dist < tol
    st = ndi.generate_binary_structure(2, 1)
    core = ndi.binary_erosion(mask, structure=st, iterations=erode)
    lab, n = ndi.label(core)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 2:
            continue
        # centroid of the *un-eroded* blob around this core (more accurate)
        grown = ndi.binary_dilation(lab == i, structure=st, iterations=erode + 1) & mask
        gy, gx = np.nonzero(grown)
        out.append((gx.mean(), gy.mean(), len(ys)))
    return out


def centre_estimate(tile, cx, cy, k, a, perp_tol=1.5, marker_r=2.5):
    """Crude estimate for a spoke whose markers are buried in the central cluster:
    outermost saturated series-coloured pixel lying on the spoke, minus a marker radius."""
    h, w, _ = tile.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - cx, cy - yy
    proj = dx * np.cos(AX_ANG[k]) + dy * np.sin(AX_ANG[k])
    perp = np.abs(-dx * np.sin(AX_ANG[k]) + dy * np.cos(AX_ANG[k]))
    near = (perp <= perp_tol) & (proj > 1) & (proj < 30)
    col = np.zeros((h, w), bool)
    for c in COLORS.values():
        col |= np.sqrt(((tile.astype(int) - np.array(c)) ** 2).sum(2)) < 60
    sel = near & col
    if not sel.any():
        return 0.0
    return max(float(proj[sel].max()) - marker_r, 1.0) / a


def digitize(fig_path: Path, out_dir: Path):
    img = np.array(Image.open(fig_path).convert("RGB"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result, meta = {}, {}
    # pass 1: frame of every panel in absolute coordinates
    frames = {}
    for name, (ox, oy), tile in tiles(img):
        cx, cy, R = find_frame(tile)
        frames[name] = (cx + ox, cy + oy, R)
    # all panels share one layout: use per-column x, per-row y and global R medians
    cols = [[PANELS[3 * r + c] for r in range(3)] for c in range(3)]
    rows = [[PANELS[3 * r + c] for c in range(3)] for r in range(3)]
    X = {p: np.median([frames[q][0] for q in col]) for col in cols for p in col}
    Y = {p: np.median([frames[q][1] for q in row]) for row in rows for p in row}
    Rg = float(np.median([f[2] for f in frames.values()]))
    for name, (ox, oy), tile in tiles(img):
        cx, cy, R = X[name] - ox, Y[name] - oy, Rg
        rings, rs, prof = find_rings(tile, cx, cy, R)
        a, b, rms, used = calibrate(rs, prof, R, TICKS[name])
        vals = {y: [np.nan] * N_AX for y in YEARS}
        found = {y: [False] * N_AX for y in YEARS}
        pix = {y: [None] * N_AX for y in YEARS}
        for y in YEARS:
            for bx, by, area in marker_blobs(tile, COLORS[y]):
                dx, dy = bx - cx, cy - by
                r = np.hypot(dx, dy)
                if r > R + 4:
                    continue  # legend stars etc.
                ang = np.arctan2(dy, dx) % (2 * np.pi)
                k = int(np.round(ang / (2 * np.pi / N_AX))) % N_AX
                dang = np.degrees(np.abs((ang - AX_ANG[k] + np.pi) % (2 * np.pi) - np.pi))
                perp = abs(-dx * np.sin(AX_ANG[k]) + dy * np.cos(AX_ANG[k]))
                if (r > 25 and dang > 6) or (r <= 25 and perp > 2.0):
                    continue
                # project onto the axis direction (radius along the spoke)
                rproj = dx * np.cos(AX_ANG[k]) + dy * np.sin(AX_ANG[k])
                if found[y][k]:
                    # keep the blob closest to the spoke
                    prev = pix[y][k]
                    if prev[2] <= dang:
                        continue
                vals[y][k] = (rproj - b) / a
                found[y][k] = True
                pix[y][k] = (bx + ox, by + oy, dang, area)
        # occlusion fill: a missing year sits under the marker drawn on top of it.
        # draw order is 2010 < 2015 < 2019 < 2023, so look at later years first.
        occl = {y: [False] * N_AX for y in YEARS}
        for k in range(N_AX):
            for i, y in enumerate(YEARS):
                if not found[y][k]:
                    later = [z for z in YEARS[i + 1:] if found[z][k]]
                    earlier = [z for z in YEARS[:i] if found[z][k]]
                    src = later[0] if later else (earlier[-1] if earlier else None)
                    if src is not None:
                        vals[y][k] = vals[src][k]
                    occl[y][k] = True
        centre_fill = []
        for k in range(N_AX):
            if not any(found[y][k] for y in YEARS):
                v = centre_estimate(tile, cx, cy, k, a)
                for y in YEARS:
                    vals[y][k] = v
                centre_fill.append(COUNTRIES[k])
        result[name] = {str(y): vals[y] for y in YEARS}
        meta[name] = dict(center=(cx + ox, cy + oy), frame_R=R, rings=[float(x) for x in rings],
                          calib_a=a, calib_b=b, calib_rms=rms, used_rings=[float(x) for x in used],
                          frame_value=(R - b) / a,
                          centre_filled=centre_fill,
                          occluded={str(y): [COUNTRIES[k] for k in range(N_AX) if occl[y][k]] for y in YEARS},
                          markers={str(y): pix[y] for y in YEARS})
    (out_dir / "fig7_digitized.json").write_text(json.dumps(result, indent=1))
    (out_dir / "fig7_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return result, meta


if __name__ == "__main__":
    fig = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fig7.png")
    res, meta = digitize(fig, Path(sys.argv[2]) if len(sys.argv) > 2 else Path("."))
    for k, m in meta.items():
        print(f"{k}: centre=({m['center'][0]:.1f},{m['center'][1]:.1f}) R={m['frame_R']:.1f} "
              f"frame={m['frame_value']:.3f} a={m['calib_a']:.3f} b={m['calib_b']:.2f} "
              f"rms={m['calib_rms']:.2f}px rings={np.round(m['rings'],1).tolist()}")
        print("   occluded:", {y: v for y, v in m['occluded'].items() if v})
