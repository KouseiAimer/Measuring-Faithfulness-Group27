#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${REPORT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python}"

cd "${REPORT_DIR}"
"${PYTHON_BIN}" analysis_pipeline.py --artifacts "${ROOT_DIR}/artifacts"
lualatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/parametric_faithfulness_sft_report_latex_1.log
lualatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/parametric_faithfulness_sft_report_latex_2.log
rm -f report.aux report.log report.out

printf 'Generated report: %s\n' "${REPORT_DIR}/report.pdf"
