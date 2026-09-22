#!/usr/bin/env bash
# Full pipeline: figure digitisation -> decision matrices -> calibration -> tables -> figures.
set -euo pipefail
cd "$(dirname "$0")"
python3 digitize/digitize_radar.py figures/fig7_radar_spis.png digitize/out
python3 digitize/digitize_stacked.py figures/fig8_composite_deconstruction.png digitize/out/fig8_digitized.json
cd src
python3 build_dataset.py
python3 calibrate.py
for d in reconstructed calibrated; do
  python3 run_ranking.py "$d"
done
python3 run_clustering.py calibrated
python3 make_figures.py calibrated
