# CoT vs ToT Parametric Faithfulness Report

## Validation

All four primary result files passed the final integrity audit:

| Dataset | Reasoning | Target coverage | Valid step records | Malformed / duplicate / incomplete epochs |
| --- | --- | ---: | ---: | ---: |
| OpenBookQA | CoT | 50 / 50 | 286 | 0 / 0 / 0 |
| OpenBookQA | ToT-selected | 50 / 50 | 297 | 0 / 0 / 0 |
| ARC-Challenge | CoT | 100 / 100 | 752 | 0 / 0 / 0 |
| ARC-Challenge | ToT-selected | 100 / 100 | 672 | 0 / 0 / 0 |

No exception, malformed JSON, CUDA OOM, duplicate step, unexpected target ID, or
missing evaluation epoch was found in the primary results.

## Configuration

- Model: `Llama-3.2-3B-Instruct`.
- Datasets: OpenBookQA (`50` target questions, `20` retain questions) and
  ARC-Challenge (`100` target questions, `20` retain questions).
- Unlearning: NPO+KL, learning rate `3e-05`, `5` epochs, POS-filtered content
  targets, optimizing `mlp.down_proj.weight` (FF2).
- CoT: one greedy reasoning path.
- ToT-selected: validation-selected multi-path approach. The winning procedure
  was `sample_select` (five candidate reasoning paths) on both datasets.

## Metrics

- `Eff`: reduction in length-normalized probability of the unlearned reasoning
  step, averaged over post-unlearning epochs and valid steps.
- `Spec`: retained-question direct-answer agreement after unlearning.
- `FF-HARD`: question-level answer flip after unlearning at least one step,
  reported following the paper protocol only where the direct and reasoning
  predictions initially agree.
- `Max FF-SOFT`: mean, over eligible questions, of the largest removed
  probability mass from the initial direct-answer prediction.
- `FF-HARD (all)`: supplemental unfiltered question-level answer flip rate.
- `Post-CoT Agree`: diagnostic agreement of newly generated reasoning and direct
  prediction after unlearning; it is not the paper's `Gen` metric.

The paper's `Gen` column is zero-shot MMLU accuracy after unlearning. MMLU
supplement runs were not executed for these completed primary experiments, so
`Gen (MMLU)` is intentionally reported as unavailable rather than inferred.

## Results

The generated tables are available in `tables/`, with machine-readable values
in `data/`. The central results are:

| Dataset | Reasoning | Eff | Spec | FF-HARD | Max FF-SOFT | FF-HARD (all) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenBookQA | CoT | 81.2 | 95.4 | 72.5 | 50.2 | 74.0 |
| OpenBookQA | ToT-selected | 81.8 | 95.1 | 71.8 | 55.3 | 78.0 |
| ARC-Challenge | CoT | 75.8 | 95.6 | 68.8 | 51.3 | 72.0 |
| ARC-Challenge | ToT-selected | 79.6 | 95.4 | 72.2 | 58.5 | 73.0 |

On ARC-Challenge, ToT-selected yields higher efficacy (`+3.8` points), higher
paper-protocol FF-HARD (`+3.4`), and higher Max FF-SOFT (`+7.3`), with a small
specificity reduction (`-0.2`). On OpenBookQA, ToT-selected yields a small
efficacy increase (`+0.6`) and a larger Max FF-SOFT increase (`+5.2`), while
paper-protocol FF-HARD is essentially unchanged (`-0.7`).

Because the paper-protocol filter may admit different questions for CoT and
ToT, a stricter paired comparison uses only questions eligible in both arms:

| Dataset | Common eligible N | CoT FF-HARD | ToT FF-HARD | ToT - CoT |
| --- | ---: | ---: | ---: | ---: |
| OpenBookQA | 37 | 70.3 | 70.3 | 0.0 |
| ARC-Challenge | 62 | 64.5 | 67.7 | +3.2 |

Thus, ToT supplies stronger parametric faithfulness evidence on ARC-Challenge,
while on OpenBookQA it primarily increases soft influence and unfiltered flip
rate without improving paired hard flips.

## ToT Selection

`sample_select` was selected over `beam_prune` on validation for both datasets:

| Dataset | sample-select accuracy | beam-prune accuracy | Selected |
| --- | ---: | ---: | --- |
| OpenBookQA | 80.0 | 70.0 | sample-select |
| ARC-Challenge | 80.0 | 76.7 | sample-select |

## Figures

- `figures/eff_spec_faithfulness.pdf`: efficacy-specificity tradeoff, with
  marker size encoding FF-HARD.
- `figures/faithfulness_comparison.pdf`: CoT versus ToT FF-HARD comparison.
- `figures/unlearning_trajectories.pdf`: efficacy and specificity across
  unlearning epochs.
- `figures/tot_mode_selection.pdf`: validation accuracy and path diversity for
  the candidate ToT procedures.

## Limitations

The experiment compares a greedy CoT path against validation-selected
best-of-five ToT paths; it does not establish that every deeper search tree
would behave similarly. The MMLU general-capability control was not run, so the
report cannot claim post-unlearning preservation of broad capabilities beyond
the measured same-domain specificity. A future supplemental run can populate
`Gen (MMLU)` without changing the completed main comparison.

