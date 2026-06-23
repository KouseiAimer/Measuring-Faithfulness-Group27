#!/usr/bin/env bash
set -euo pipefail

PPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PPT_DIR}"

pdflatex -interaction=nonstopmode -halt-on-error deepseek_sft_fur_ppt.tex >/tmp/deepseek_sft_fur_ppt_latex_1.log
pdflatex -interaction=nonstopmode -halt-on-error deepseek_sft_fur_ppt.tex >/tmp/deepseek_sft_fur_ppt_latex_2.log
rm -f deepseek_sft_fur_ppt.aux deepseek_sft_fur_ppt.log deepseek_sft_fur_ppt.nav \
      deepseek_sft_fur_ppt.out deepseek_sft_fur_ppt.snm deepseek_sft_fur_ppt.toc

printf 'Generated slides: %s\n' "${PPT_DIR}/deepseek_sft_fur_ppt.pdf"
