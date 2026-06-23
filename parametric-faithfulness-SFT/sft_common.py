"""Shared utilities for the independent DeepSeek-revision SFT experiment."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import nltk
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = ROOT / "local_models" / "Llama-3.2-3B-Instruct"
DEFAULT_DATASET = ROOT / "local_datasets" / "openbookqa"
ARTIFACTS = ROOT / "artifacts"

ANSWER_LETTERS = ["A", "B", "C", "D", "E"]
ANSWER_PREFIX = "Human: Given all of the above, what's the single, most likely answer?"
ANSWER_COMPLETION_PREFIX = "Assistant: The single, most likely answer is ("


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_env(path: Path | None = None) -> None:
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def openbook_dataset(path: Path | str = DEFAULT_DATASET):
    return load_from_disk(str(path))


def answer_letters(instance: dict) -> list[str]:
    return [str(letter) for letter in instance["choices"]["label"]]


def answer_choices(instance: dict) -> list[str]:
    return [
        f"({letter}): {text}"
        for letter, text in zip(instance["choices"]["label"], instance["choices"]["text"])
    ]


def cot_prompt(instance: dict) -> str:
    choices = "\n".join(answer_choices(instance))
    return (
        f"Human: Question: {instance['question_stem']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Assistant: Let's think step by step:\n"
    )


def answer_prompt(instance: dict, rationale: str | None = None) -> str:
    choices = "\n".join(answer_choices(instance))
    if rationale is None:
        return (
            f"Human: Question: {instance['question_stem']}\n\n"
            f"Choices:\n{choices}\n\n"
            f"{ANSWER_PREFIX}\n{ANSWER_COMPLETION_PREFIX}"
        )
    return (
        f"{cot_prompt(instance)}{rationale.strip()}\n\n"
        f"{ANSWER_PREFIX}\n{ANSWER_COMPLETION_PREFIX}"
    )


def configure_tokenizer(model_path: Path | str):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.clean_up_tokenization_spaces = False
    return tokenizer


def load_causal_model(model_path: Path | str, train: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.train() if train else model.eval()
    return model


def model_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def generate_rationale(
    model,
    tokenizer,
    instance: dict,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    prompt = cot_prompt(instance)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model_device(model))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs.update({"temperature": temperature, "top_p": 0.9})
    outputs = model.generate(**inputs, **kwargs)
    new_ids = outputs[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text.split("\n\n")[0].strip()


@torch.no_grad()
def generate_rationales(
    model,
    tokenizer,
    instances: list[dict],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> list[str]:
    prompts = [cot_prompt(instance) for instance in instances]
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(model_device(model))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs.update({"temperature": temperature, "top_p": 0.9})
    outputs = model.generate(**inputs, **kwargs)
    prefix_length = inputs["input_ids"].shape[-1]
    texts = tokenizer.batch_decode(outputs[:, prefix_length:], skip_special_tokens=True)
    return [text.strip().split("\n\n")[0].strip() for text in texts]


@torch.no_grad()
def option_probabilities(model, tokenizer, instance: dict, rationale: str | None = None) -> list[float]:
    prompt = answer_prompt(instance, rationale=rationale)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model_device(model))
    logits = model(**inputs).logits[0, -1]
    labels = answer_letters(instance)
    token_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
    values = torch.softmax(logits[token_ids], dim=-1).float().cpu().tolist()
    return values


@torch.no_grad()
def option_probabilities_batch(
    model, tokenizer, instances: list[dict], rationales: list[str | None]
) -> list[list[float]]:
    prompts = [
        answer_prompt(instance, rationale=rationale)
        for instance, rationale in zip(instances, rationales)
    ]
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(model_device(model))
    logits = model(**inputs).logits[:, -1, :]
    outputs = []
    for index, instance in enumerate(instances):
        token_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in answer_letters(instance)]
        outputs.append(torch.softmax(logits[index, token_ids], dim=-1).float().cpu().tolist())
    return outputs


def predicted_letter(instance: dict, probabilities: list[float]) -> str:
    return answer_letters(instance)[int(np.argmax(probabilities))]


def normalize_answer(value: object) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else ""


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def explicit_answer_leak(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?[A-E]\)?\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def sentence_tokenize(text: str) -> list[str]:
    try:
        sentences = nltk.sent_tokenize(text.strip())
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]
