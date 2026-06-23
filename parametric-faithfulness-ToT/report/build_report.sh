#!/usr/bin/env bash
set -euo pipefail

PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"
"$PYTHON" -m py_compile analysis_pipeline.py
"$PYTHON" analysis_pipeline.py --output_dir report > report/analysis_generation.log

cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/tot_report_pdflatex_1.log
pdflatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/tot_report_pdflatex_2.log
