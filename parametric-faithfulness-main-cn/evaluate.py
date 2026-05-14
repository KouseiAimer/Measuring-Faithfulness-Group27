import re
import random
import gc
import nltk

import torch
import numpy as np

from tqdm import tqdm
from pprint import pprint

from models import load_model_and_tokenizer
from dataload import DATASETS

BOWMAN_HUMAN_ANSWER_PREFIX = "Human: Given all of the above, what's the single, most likely answer?"
BOWMAN_ASSISTANT_ANSWER_PREFIX = "Assistant: The single, most likely answer is ("

ANSWER_LETTERS = ["A", "B", "C", "D", "E"] # No MCQA dataset considered has more than 5

def decode_new_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def answer_letter_token_ids(tokenizer, answer_letters):
    ids = []
    for letter in answer_letters:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Could not encode answer letter {letter!r}")
        ids.append(encoded[0])
    return ids

def segment_generated_cot(cot_text):
    cot_text = cot_text.strip()
    if not cot_text:
        return []

    has_cjk = re.search(r"[\u4e00-\u9fff]", cot_text) is not None
    if has_cjk:
        pieces = []
        for block in re.split(r"\n+", cot_text):
            block = block.strip()
            if not block:
                continue
            pieces.extend(
                part.strip()
                for part in re.split(r"(?<=[\u3002\uff01\uff1f!?\uff1b;])\s*", block)
                if part.strip()
            )
        return pieces or [cot_text]

    try:
        pieces = [part.strip() for part in nltk.sent_tokenize(cot_text) if part.strip()]
    except LookupError:
        pieces = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cot_text) if part.strip()]
    return pieces or [cot_text]

def answer_probabilities(model, tokenizer, dh, instance):
    device = model.device
    with torch.no_grad():
        n_options = len(dh.get_answer_letters(instance))
        answer_letters = ANSWER_LETTERS[:n_options]
        answer_indices = answer_letter_token_ids(tokenizer, answer_letters)

        prompt = dh.make_bowman_demonstration(instance)
        answer_inputs = tokenizer.encode(prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)
        
        answer_output = model.generate(input_ids=answer_inputs, max_new_tokens=10,
                                        output_scores=True,
                                        temperature=0., do_sample=False,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id)
    
        # 2.1 obtain letter completion probabilities
        first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
        letter_probs = first_token_probs[answer_indices]
        predicted_letter_index = torch.argmax(letter_probs).item()
        letter_probs = letter_probs.detach().cpu().float().numpy()
    
        # 2.2 take only newly generated output
        answer_output = answer_output[0][0]
        answer_new_output = answer_output[answer_inputs.shape[-1]:]
        answer_new_output_text = decode_new_tokens(tokenizer, answer_new_output)
        
        return answer_new_output_text, letter_probs, predicted_letter_index

def complete(model, tokenizer, prompt, max_new_tokens=300, temperature=0., do_sample=False, split_newline=True):
  do_sample = temperature > 0. # overwrite
  with torch.no_grad():
    device = model.device
    
    inputs = tokenizer.encode(prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)

    outputs = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                                    output_scores=True,
                                    temperature=temperature, do_sample=do_sample,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id)

    # 2 take only newly generated output
    output = outputs[0][0]
    new_output = output[inputs.shape[-1]:]
    new_output_text = decode_new_tokens(tokenizer, new_output)
    if split_newline:
      new_output_text = new_output_text.strip().split("\n\n")[0] 
    
    return new_output_text

def letter_completion(model, tokenizer, prompt, N):
  with torch.no_grad():
    device = model.device
    answer_letters = ANSWER_LETTERS[:N]
    answer_indices = answer_letter_token_ids(tokenizer, answer_letters)


    # Step 5: make answer prompt
    answer_inputs = tokenizer.encode(prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)
    answer_output = model.generate(input_ids = answer_inputs, max_new_tokens=20,
                                    output_scores=True, return_dict_in_generate=True,
                                      pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2.1 obtain letter completion probabilities
    first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
    letter_probs = first_token_probs[answer_indices]
    predicted_letter_index = torch.argmax(letter_probs).item()
    letter_probs = letter_probs.detach().cpu().float().numpy()

    # 2.2 take only newly generated output
    answer_output = answer_output[0][0]
    answer_new_output = answer_output[answer_inputs.shape[-1]:]
    answer_new_output_text = decode_new_tokens(tokenizer, answer_new_output)
    
    return letter_probs, predicted_letter_index

def generate_dataset_cots(model_id, tokenizer, dataset_id, temperature, sentencize=True, max_instances=250, seed=1001):
    print(f"Generating new CoTs for {model_id}, {dataset_id}, sentencize={sentencize}")
    model, _ = load_model_and_tokenizer(model_id)
    DH = DATASETS[dataset_id]
    _, valid, test = DH.get_dataset_splits()
    if dataset_id == 'sqa': test = valid # SQA test doesn't have answers

    instance_info = []
    
    indices = list(range(len(test)))
    random.Random(seed).shuffle(indices)
    if max_instances is not None:
        indices = indices[:max_instances]

    for idx, instance_idx in tqdm(enumerate(indices), total=len(indices)):
        instance = test[instance_idx]
        _, nocot_probs, _ = answer_probabilities(
            model, tokenizer, DH, instance)

        cot_prompt = DH.make_cot_prompt(instance)
        cot = complete(model, tokenizer, cot_prompt, temperature=temperature)

        cot_probs, _  = generation_fixed_cot(model, tokenizer, DH, instance, cot)
        segmented_cot = None
        if sentencize:
            segmented_cot = segment_generated_cot(cot)
        inst_details = {
            'id': instance[DH.id_key],
            'question': DH.get_question(instance),
            'correct_letter': DH.correct_answer_letter(instance),
            'cot_prompt': DH.make_cot_prompt(instance),
            'cot': cot,
            'options': DH.get_answer_choices(instance),
            'nocot_probs': nocot_probs.tolist(),
            'cot_probs': cot_probs.tolist(),
            'segmented_cot': segmented_cot, 
            'subject': instance.get('subject'),
            'raw_instance': instance,
        }
        if idx == 1:
          pprint(inst_details)
        instance_info.append(inst_details)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return instance_info

def generation_fixed_cot(model, tokenizer, dh, instance, cot_text):
  with torch.no_grad():
    device = model.device
    n_options = len(dh.get_answer_letters(instance))
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = answer_letter_token_ids(tokenizer, answer_letters)
    cot_prompt = dh.make_cot_prompt(instance)

    cot_text = cot_text.strip().split("\n\n")[0] # Only up to double newline

    # Step 5: make answer prompt
    answer_prompt = dh.make_answer_prompt(cot_prompt + cot_text)
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)
    answer_output = model.generate(input_ids = answer_inputs, max_new_tokens=20,
                                    output_scores=True, return_dict_in_generate=True,
                                      pad_token_id=tokenizer.pad_token_id)

    # 2.1 obtain letter completion probabilities
    first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
    letter_probs = first_token_probs[answer_indices]
    predicted_letter_index = torch.argmax(letter_probs).item()
    letter_probs = letter_probs.detach().cpu().float().numpy()

    # 2.2 take only newly generated output
    answer_output = answer_output[0][0]
    answer_new_output = answer_output[answer_inputs.shape[-1]:]
    answer_new_output_text = decode_new_tokens(tokenizer, answer_new_output)
    
    return letter_probs, predicted_letter_index

def generate(model, tokenizer, instance):
  with torch.no_grad():
    device = model.device
    # Step 1: make answer prompt
    n_options = len(instance['cot_probs'])
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = answer_letter_token_ids(tokenizer, answer_letters)

    DELIM = "\n\n"
    answer_prompt = DELIM.join([instance['question'], instance['options']])
    answer_prompt = answer_prompt + f"{DELIM}Answer: ("
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)

    answer_output = model.generate(input_ids=answer_inputs, max_new_tokens=10,
                                    output_scores=True,
                                    temperature=0., do_sample=False,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2.1 obtain letter completion probabilities
    first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
    letter_probs = first_token_probs[answer_indices]
    predicted_letter_index = torch.argmax(letter_probs).item()
    letter_probs = letter_probs.detach().cpu().float().numpy()

    # 2.2 take only newly generated output
    answer_output = answer_output[0][0]
    answer_new_output = answer_output[answer_inputs.shape[-1]:]
    answer_new_output_text = decode_new_tokens(tokenizer, answer_new_output)
    
    return answer_new_output_text, letter_probs, predicted_letter_index

def get_cot_prompt(instance):
  LTSBS = "Assistant: Let's think step by step:\n"
  DELIM = "\n\n"
  answer_prompt = DELIM.join([instance['question'], instance['options'], LTSBS])
  return answer_prompt

def generate_cot(model, tokenizer, instance, max_new_tokens=300, temperature=0., do_sample=False):
  with torch.no_grad():
    # "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n"
    LTSBS = "Assistant: Let's think step by step:\n"

    device = model.device

    # 1. Make CoT prompt
    DELIM = "\n\n"
    answer_prompt = DELIM.join([instance['question'], instance['options'], LTSBS])
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)

    cot_output = model.generate(input_ids=answer_inputs, max_new_tokens=max_new_tokens,
                                    output_scores=True,
                                    temperature=temperature, do_sample=do_sample,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2 take only newly generated output
    cot_output = cot_output[0][0]
    cot_new_output = cot_output[answer_inputs.shape[-1]:]
    cot_new_output_text = decode_new_tokens(tokenizer, cot_new_output)
    cot_new_output_text = cot_new_output_text.strip().split("\n\n")[0] 
    
    return cot_new_output_text, answer_prompt

def cot_generate(model, tokenizer, instance, max_new_tokens=300, temperature=0., do_sample=False):
  with torch.no_grad():
    device = model.device

    n_options = len(instance['cot_probs'])
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = answer_letter_token_ids(tokenizer, answer_letters)

    cot_text, cot_prompt = generate_cot(model, tokenizer, instance,
                          max_new_tokens=max_new_tokens, temperature=temperature,do_sample=do_sample)

    prefix = cot_prompt + cot_text
    # 1. Make CoT prompt
    DELIM = "\n"
    answer_prompt = DELIM.join([prefix, BOWMAN_HUMAN_ANSWER_PREFIX, BOWMAN_ASSISTANT_ANSWER_PREFIX])
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)
    # 2. Generate answer
    answer_output = model.generate(input_ids=answer_inputs,
                                    max_new_tokens=10,
                                    output_scores=True,
                                    temperature=temperature, do_sample=do_sample,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2.1 obtain letter completion probabilities
    first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
    letter_probs = first_token_probs[answer_indices]
    predicted_letter_index = torch.argmax(letter_probs).item()
    letter_probs = letter_probs.detach().cpu().float().numpy()

    # 2.2 take only newly generated output
    answer_output = answer_output[0][0]
    answer_new_output = answer_output[answer_inputs.shape[-1]:]
    answer_new_output_text = decode_new_tokens(tokenizer, answer_new_output)
    
    return answer_new_output_text, letter_probs, predicted_letter_index,cot_text, cot_prompt

def completion_probabilities(model, tokenizer, prefix, targets):
    device = model.device
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device) # [1, T]
    prefix_length = prefix_ids.size(-1)

    n_sequences = len(targets)
    n_prefix_ids = prefix_ids.repeat(n_sequences, 1)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # Convert targets to tensors, concat to inputs
    target_ids = tokenizer(targets, padding=True, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # Count lengths of individual target sequences for scaling
    non_pad = target_ids != pad_token_id
    lengths = torch.count_nonzero(non_pad, dim=-1)

    # Stack inputs
    input = torch.hstack([
       n_prefix_ids,target_ids[:,:-1] # Exclude last target token from input
                          ])

    # Fwd pass
    outputs = model.forward(input, return_dict=True) # logits = B, T, V
    relevant_logits = outputs['logits'][:,prefix_length-1:]
    # Any benefits from logsoftmax if we want the actual probability in the end?
    # Yes, numeric stability
    token_probs = torch.log_softmax(relevant_logits, dim=-1)

    # Pad tokens should contribute log-probability 0.
    token_probs[:,:,pad_token_id] = 0.
    target_probs = token_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    target_probs = target_probs.squeeze(1)
    seq_probs = torch.sum(target_probs, dim=1)  # prod if not in logspace
    length_penalty = model.generation_config.length_penalty
    seq_probs /= lengths**length_penalty # if not logspace seq_probs /= lengths

    return seq_probs
