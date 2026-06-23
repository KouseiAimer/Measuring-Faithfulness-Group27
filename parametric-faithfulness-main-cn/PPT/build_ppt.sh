#!/usr/bin/env bash
set -euo pipefail

PPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PPT_DIR"

pdflatex -interaction=nonstopmode -halt-on-error qwen_ceval_fur_ppt.tex >/tmp/qwen_ceval_fur_ppt_pdflatex_1.log
pdflatex -interaction=nonstopmode -halt-on-error qwen_ceval_fur_ppt.tex >/tmp/qwen_ceval_fur_ppt_pdflatex_2.log

rm -f qwen_ceval_fur_ppt.aux qwen_ceval_fur_ppt.log qwen_ceval_fur_ppt.nav \
      qwen_ceval_fur_ppt.out qwen_ceval_fur_ppt.snm qwen_ceval_fur_ppt.toc
