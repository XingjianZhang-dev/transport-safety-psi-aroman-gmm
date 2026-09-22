"""Build the four yearly decision matrices (19 countries x 9 SPIs) from the Fig. 7 digitization.

Post-processing
  * C41-C44 (WHO enforcement ratings, integer 0-10): values within 0.12 of an integer are
    snapped to it.  The remaining cells are candidate mean-substituted cells: we recompute
    them as the mean of the integer cells of the same column/year and accept the
    recomputed value if it agrees with the digitized one within 0.12.
  * every cell carries a quality flag:
        ok        marker found, value read directly
        int       C-column value snapped to an integer
        mean      C-column value recomputed as column mean (mean substitution confirmed)
        occluded  marker hidden under a later/earlier year's marker; value copied from it
        centre    A32 marker buried in the central cluster; crude estimate (low precision)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
C = ["AR", "AU", "BR", "CA", "CH", "FR", "DE", "IN", "ID", "IT",
     "JP", "KR", "MX", "RU", "SA", "ZA", "TR", "GB", "US"]
NAMES = ["Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany", "India",
         "Indonesia", "Italy", "Japan", "South Korea", "Mexico", "Russia", "Saudi Arabia",
         "South Africa", "Turkey", "United Kingdom", "United States"]
IND = ["A31", "A32", "A33", "B21", "B22", "C41", "C42", "C43", "C44"]
LONG = ["Road_fatalities_per_100_000_inhabitants",
        "Road_fatalities_per_10_000_registered_vehicles",
        "Change_in_number_of_road_deaths",
        "Seatbelt_wearing_rates_front_seats",
        "Seatbelt_wearing_rates_rear_seats",
        "Enforcement_score_speed_limit_law",
        "Enforcement_score_drink_driving_law",
        "Enforcement_score_seat_belt_law",
        "Enforcement_score_helmet_use_law"]
YEARS = [2010, 2015, 2019, 2023]


def build(dig_json: Path, meta_json: Path, out_dir: Path):
    dig = json.loads(dig_json.read_text())
    meta = json.loads(meta_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {y: pd.DataFrame(index=C, columns=IND, dtype=float) for y in YEARS}
    flag = {y: pd.DataFrame("ok", index=C, columns=IND) for y in YEARS}
    for ind in IND:
        for y in YEARS:
            data[y][ind] = np.array(dig[ind][str(y)], dtype=float)
            for c in meta[ind]["occluded"].get(str(y), []):
                flag[y].loc[c, ind] = "occluded"
            for c in meta[ind]["centre_filled"]:
                flag[y].loc[c, ind] = "centre"
    log = []
    for ind in ["C41", "C42", "C43", "C44"]:
        for y in YEARS:
            v = data[y][ind]
            is_int = (v - v.round()).abs() <= 0.12
            ints = v[is_int].round()
            mean_obs = ints.mean()
            for c in C:
                if is_int[c]:
                    data[y].loc[c, ind] = round(v[c])
                    if flag[y].loc[c, ind] == "ok":
                        flag[y].loc[c, ind] = "int"
                else:
                    ok = abs(v[c] - mean_obs) <= 0.12
                    log.append(dict(indicator=ind, year=y, country=c, digitized=round(v[c], 3),
                                    mean_of_integer_cells=round(mean_obs, 3),
                                    n_integer_cells=int(is_int.sum()), confirmed=bool(ok)))
                    if ok:
                        data[y].loc[c, ind] = mean_obs
                        flag[y].loc[c, ind] = "mean"
    log = pd.DataFrame(log)
    for y in YEARS:
        df = data[y].copy()
        df.index.name = "Code"
        df.to_csv(out_dir / f"decision_matrix_{y}.csv", float_format="%.4f")
        flag[y].to_csv(out_dir / f"quality_flags_{y}.csv")
        # v0-style clustering input (Country, Code, long column names)
        cl = df.copy()
        cl.columns = LONG
        cl.insert(0, "Country", NAMES)
        cl.reset_index().to_csv(out_dir / f"clustering_input_{y}.csv", index=False, float_format="%.4f")
    log.to_csv(out_dir / "mean_substitution_check.csv", index=False)
    summary = pd.concat({y: flag[y].apply(pd.Series.value_counts).fillna(0).astype(int) for y in YEARS})
    summary.to_csv(out_dir / "quality_flag_counts.csv")
    return data, flag, log


if __name__ == "__main__":
    data, flag, log = build(ROOT / "digitize/out/fig7_digitized.json",
                            ROOT / "digitize/out/fig7_meta.json", ROOT / "data")
    pd.set_option("display.width", 220)
    print(log.to_string(index=False))
    print("\nconfirmed mean-substitution cells:", int(log.confirmed.sum()), "/", len(log))
    for y in YEARS:
        print(f"\n{y}  flags:", flag[y].stack().value_counts().to_dict())
        print(data[y].round(2).T.to_string())
