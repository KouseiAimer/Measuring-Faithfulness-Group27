#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

pdflatex -interaction=nonstopmode -halt-on-error tot_cot_faithfulness_ppt.tex >/tmp/tot_cot_ppt_pdflatex_1.log
pdflatex -interaction=nonstopmode -halt-on-error tot_cot_faithfulness_ppt.tex >/tmp/tot_cot_ppt_pdflatex_2.log
