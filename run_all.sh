#!/usr/bin/env bash
# Reproduces every number, table and figure in the report (CPU is fine).
# Prerequisite: data/raw_4families.csv.gz (see scripts/download_and_filter.py)
set -euo pipefail
cd "$(dirname "$0")"
python -m src.data                               # clean, balance (1000/family), split
python -m src.check_task_labels --score || true  # precision of task labels (audit file)
python -m src.run_experiments --stage rq1_rq2
python -m src.era_analysis                       # RQ2 (iii): temporal drift in prompts
python -m src.run_experiments --stage rq2_confounded
python -m src.run_experiments --stage rq3
python -m src.run_experiments --stage rq4
python -m src.stylometry                         # RQ4 hand-crafted features
python -m src.interpret                          # RQ4 LR weights + CNN saliency
python -m src.analyze                            # tables, figures, results/SUMMARY.md
