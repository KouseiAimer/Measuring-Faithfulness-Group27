"""Ask DeepSeek to revise student drafts and build a filtered SFT dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from sft_common import (
    ARTIFACTS,
    append_jsonl,
    cot_prompt,
    explicit_answer_leak,
    load_env,
    normalize_answer,
    read_jsonl,
    word_count,
    write_jsonl,
)


SYSTEM_PROMPT = """You are a careful science multiple-choice reasoning teacher.
You will receive a question, answer options, and a draft rationale from a smaller student model.
Solve the problem independently and correct factual or logical errors in the draft.
Return valid JSON with exactly two keys: "rationale" and "answer".
The rationale must contain 2 to 4 short sentences and at most 80 words.
The rationale must explain the scientific reasoning, but must not mention an answer letter,
an option letter, or phrases such as "the answer is".
The answer field must be a single option letter."""


def teacher_prompt(row: dict) -> str:
    options = "\n".join(row["options"])
    return (
        f"Question:\n{row['question']}\n\n"
        f"Options:\n{options}\n\n"
        f"Student draft rationale:\n{row['student_draft']}\n\n"
        "Revise the rationale and determine the answer. Do not assume the draft is correct."
    )


def extract_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def sentence_count(text: str) -> int:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([chunk for chunk in chunks if chunk.strip()])


def request_revision(
    session: requests.Session,
    api_key: str,
    row: dict,
    model: str,
    api_url: str,
    timeout: float,
    max_tokens: int,
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": teacher_prompt(row)},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = session.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"].get("content") or ""
    usage = body.get("usage", {})
    return content, usage


def filter_revision(row: dict, content: str, min_words: int, max_words: int) -> dict:
    try:
        parsed = extract_json(content)
    except (json.JSONDecodeError, TypeError):
        return {"accepted": False, "reject_reason": "invalid_json", "teacher_content": content}
    rationale = str(parsed.get("rationale", "")).strip()
    teacher_answer = normalize_answer(parsed.get("answer"))
    n_words = word_count(rationale)
    n_sentences = sentence_count(rationale)
    reason = ""
    if teacher_answer != str(row["correct_letter"]).upper():
        reason = "wrong_teacher_answer"
    elif not rationale:
        reason = "empty_rationale"
    elif n_words < min_words or n_words > max_words:
        reason = "rationale_length"
    elif n_sentences < 2 or n_sentences > 4:
        reason = "sentence_count"
    elif explicit_answer_leak(rationale):
        reason = "answer_leak"
    return {
        "accepted": not reason,
        "reject_reason": reason,
        "teacher_answer": teacher_answer,
        "teacher_rationale": rationale,
        "teacher_word_count": n_words,
        "teacher_sentence_count": n_sentences,
        "teacher_content": content,
    }


def rebuild_sft(revisions_path: Path, sft_path: Path) -> int:
    accepted = []
    for row in read_jsonl(revisions_path):
        if not row.get("accepted"):
            continue
        accepted.append(
            {
                "id": row["id"],
                "prompt": cot_prompt(row["raw_instance"]),
                "completion": row["teacher_rationale"],
                "question": row["question"],
                "options": row["options"],
                "correct_letter": row["correct_letter"],
                "student_draft": row["student_draft"],
                "student_cot_answer": row["student_cot_answer"],
                "teacher_answer": row["teacher_answer"],
                "raw_instance": row["raw_instance"],
            }
        )
    write_jsonl(sft_path, accepted)
    return len(accepted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, default=ARTIFACTS / "data" / "train_student_drafts.jsonl")
    parser.add_argument("--revisions", type=Path, default=ARTIFACTS / "data" / "teacher_revisions.jsonl")
    parser.add_argument("--sft-output", type=Path, default=ARTIFACTS / "data" / "sft_train.jsonl")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--api-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--limit", type=int, default=0, help="Maximum new API requests; 0 means all drafts.")
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Completion budget including deepseek-v4-pro thinking tokens.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Maximum concurrent DeepSeek requests.")
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set in the environment or local .env file.")
    drafts = read_jsonl(args.drafts)
    if not drafts:
        raise SystemExit(f"No student drafts found: {args.drafts}")
    existing = read_jsonl(args.revisions)
    done = {
        str(row["id"])
        for row in existing
        if not (args.retry_errors and row.get("reject_reason") == "api_error")
    }
    pending = [row for row in drafts if str(row["id"]) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"Drafts: {len(drafts)}, already processed: {len(done)}, API requests this run: {len(pending)}")

    def revise_one(row: dict) -> dict:
        content = ""
        usage = {}
        error = ""
        session = requests.Session()
        for attempt in range(args.retries + 1):
            try:
                content, usage = request_revision(
                    session, api_key, row, args.model, args.api_url, args.timeout, args.max_tokens
                )
                error = ""
                break
            except (requests.RequestException, KeyError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(2**attempt)
        result = (
            {"accepted": False, "reject_reason": "api_error", "api_error": error}
            if error
            else filter_revision(row, content, args.min_words, args.max_words)
        )
        return {**row, **result, "teacher_model": args.model, "api_usage": usage}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(revise_one, row): row["id"] for row in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek revisions"):
            append_jsonl(args.revisions, future.result())

    accepted = rebuild_sft(args.revisions, args.sft_output)
    print(f"Accepted SFT records: {accepted}; wrote {args.sft_output}.")


if __name__ == "__main__":
    main()
