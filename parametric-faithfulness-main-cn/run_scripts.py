import os, sys

models = {
    'Qwen/Qwen3-8B': True,
    'Qwen/Qwen3-3B': True,
    'Qwen/Qwen3-1.7B': False,
}

datasets = ['ceval']

lrs = [5e-5, 3e-5, 5e-6]

model_to_short = {
  'Qwen/Qwen3-8B': 'Qwen3-8B',
  'Qwen/Qwen3-3B': 'Qwen3-3B',
  'Qwen/Qwen3-1.7B': 'Qwen3-1.7B',
}

ablate_ff2_small = 'ablate_ff2.job'
ablate_ff2_large = 'ablate_ff2_large.job'

small_script = 'ul_step_pos_ff2.job'
big_script = 'ul_step_pos_ff2_L2.job'

method = 'npo_KL'

# sbatch --job-name=Qwen3-8B-ceval-stepwise-lr1e-05 ul_step_pos_ff2_L2.job Qwen/Qwen3-8B ceval 1e-5 npo_KL
script_template = "sbatch --job-name={}-{}-stepwise-lr{} {} {} {} {} {}"

for dataset in datasets:
  for model, big in models.items():
    print("-"*30)
    print(f"{dataset} => {model}")
    print("-"*30)
    for lr in [1e-5]: # adjust per model if needed
      mod = model_to_short[model]
      script = big_script if big else small_script
      print(script_template.format(
        mod, dataset, lr, script, model, dataset, lr, method
      ))
