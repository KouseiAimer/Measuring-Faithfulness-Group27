#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$REPORT_DIR/.." && pwd)"

cd "$ROOT"
python -m py_compile report/analysis_pipeline.py
python report/analysis_pipeline.py --output_dir report > report/analysis_generation.log

cd "$REPORT_DIR"
lualatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/main_cn_report_lualatex_1.log
lualatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/main_cn_report_lualatex_2.log
rm -f report.aux report.log report.out report.toc
rm -rf "$REPORT_DIR/__pycache__"
