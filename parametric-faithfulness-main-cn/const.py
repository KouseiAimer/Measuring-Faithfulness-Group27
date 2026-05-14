datasets = ['ceval']
model_name_dict = {
    'Qwen3-8B': 'Qwen3-8B',
    'Qwen3-3B': 'Qwen3-3B',
    'Qwen3-1.7B': 'Qwen3-1.7B',
    'Phi-3-mini-4k-instruct': 'Phi-3',
    'Meta-Llama-3-8B-Instruct': 'LLaMA-3',
    'Meta-Llama-3-3B-Instruct': 'LLaMA-3-3B',
    'Llama-3.2-3B-Instruct': 'LLaMA-3-3B',
    'Mistral-7B-Instruct-v0.2': 'Mistral-2',
}

model_name_to_path = {
    'Qwen3-8B': 'Qwen/Qwen3-8B',
    'Qwen3-3B': 'Qwen/Qwen3-3B',
    'Qwen3-1.7B': 'Qwen/Qwen3-1.7B',
    'Phi-3': 'microsoft/Phi-3-mini-4k-instruct',
    'LLaMA-3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'LLaMA-3-3B':'meta-llama/Llama-3.2-3B-Instruct',
    'Mistral-2': 'mistralai/Mistral-7B-Instruct-v0.2',
}

LETTERS = ['A', 'B', 'C', 'D', 'E']

models = list(set(model_name_dict.values()))


dataset_model_best_lr = {
    'ceval':{
        'Qwen3-8B': 1e-05,
        'Qwen3-3B': 1e-05,
        'Qwen3-1.7B': 3e-05,
    },

    'arc-challenge':{
        'Phi-3': 1e-04,
        'LLaMA-3': 1e-05,
        'LLaMA-3-3B': 3e-05,
        'Mistral-2': 5e-06,
    },

    'openbook':{
        'Phi-3': 1e-04,
        'LLaMA-3': 1e-05,
        'LLaMA-3-3B': 3e-05,
        'Mistral-2': 5e-06,
    },
    'sports': {
        'Phi-3': 5e-05,
        'LLaMA-3': 5e-06,
        'LLaMA-3-3B': 3e-05,
        'Mistral-2': 3e-06,
    },

    'sqa': {
        'Phi-3': 5e-05,
        'LLaMA-3': 1e-05,
        'LLaMA-3-3B': 3e-05,
        'Mistral-2': 5e-06,
    },
}
