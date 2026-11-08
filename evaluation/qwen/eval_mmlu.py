import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

CONFIG_DIR = Path(__file__).resolve().parent / "config"
RESULT_DIR = Path("./results")
PROMPT_TEMPLATE = """Answer the following multiple-choice question. Give concise reasoning and finish with exactly: The answer is (X), where X is A, B, C, or D.

Question: {question}

{options}
"""


def load_model_config(model_name, config_dir):
    for path in config_dir.glob("*.yaml"):
        try:
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg and cfg.get("name") == model_name:
                print(f"Using config: {path}")
                return cfg
        except Exception as exc:
            print(f"[WARNING] Failed to parse config {path}: {exc}")
    print(f"[WARNING] No config found for {model_name}; using defaults.")
    return {"name": model_name, "max_tokens": 4096}


def build_generation_kwargs(cfg):
    kwargs = {key: cfg[key] for key in ('temperature', 'top_p') if key in cfg}
    if cfg.get('do_sample') is False:
        kwargs.setdefault('temperature', 0.0)
    extra_body = {key: cfg[key] for key in ('top_k', 'repetition_penalty', 'min_p') if key in cfg}
    if extra_body:
        kwargs['extra_body'] = extra_body
    kwargs['max_tokens'] = cfg.get('max_tokens', 4096)
    kwargs.setdefault('extra_body', {}).setdefault('chat_template_kwargs', {})['enable_thinking'] = False
    return kwargs


def strip_thinking(text):
    text = text or ""
    if "<think>" in text and "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def extract_answer_letter(text):
    content = strip_thinking(text)
    patterns = (
        r"the answer is\s*\(([ABCD])\)",
        r"(?:final answer|answer)\s*(?:is|:)\s*\(?([ABCD])\)?",
        r"\(([ABCD])\)",
        r"^\s*([ABCD])(?:\s|[.)]|$)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    return ""


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = {"question", "subject", "choices", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    samples = []
    for sample_id, row in df.iterrows():
        choices = list(row["choices"])
        answer_idx = int(row["answer"])
        if len(choices) != 4 or answer_idx not in range(4):
            raise ValueError(f"Invalid choices/answer at row {sample_id}: choices={choices}, answer={answer_idx}")
        samples.append({
            "sample_id": int(sample_id),
            "question": str(row["question"]),
            "subject": str(row["subject"]),
            "choices": [str(x) for x in choices],
            "correct_letter": "ABCD"[answer_idx],
        })
    return samples


def evaluate_one(client, model_name, sample, generation_kwargs):
    options = "\n".join(f"{letter}) {choice}" for letter, choice in zip("ABCD", sample["choices"]))
    prompt = PROMPT_TEMPLATE.format(question=sample["question"], options=options)
    base = {
        "sample_id": sample["sample_id"],
        "subject": sample["subject"],
        "question": sample["question"],
        "choices": sample["choices"],
        "correct_letter": sample["correct_letter"],
    }
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **generation_kwargs,
        )
        raw = response.choices[0].message.content or ""
        predicted = extract_answer_letter(raw)
        return {
            **base,
            "model_raw": raw,
            "predicted_letter": predicted,
            "extraction_failed": not bool(predicted),
            "is_correct": predicted == sample["correct_letter"],
            "label": int(predicted == sample["correct_letter"]),
        }
    except Exception as exc:
        return {
            **base,
            "model_raw": "",
            "predicted_letter": "",
            "extraction_failed": True,
            "is_correct": False,
            "label": 0,
            "error": f"Generation error: {exc}",
        }


def summarize(results):
    total = len(results)
    correct = sum(item["label"] for item in results)
    errors = sum("error" in item for item in results)
    extraction_failures = sum(item["extraction_failed"] for item in results)
    grouped = defaultdict(list)
    for item in results:
        grouped[item["subject"]].append(item)
    by_subject = {}
    for subject in sorted(grouped):
        rows = grouped[subject]
        subject_correct = sum(item["label"] for item in rows)
        by_subject[subject] = {
            "total": len(rows),
            "correct": subject_correct,
            "accuracy": subject_correct / len(rows),
        }
    return {
        "dataset_type": "mmlu",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "errors": errors,
        "extraction_failures": extraction_failures,
        "by_subject": by_subject,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--test-file", default=str(Path(__file__).resolve().parents[2] / "eval_data/mmlu.parquet"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    cfg = load_model_config(args.model_name, Path(args.config_dir))
    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{args.port}/v1")
    samples = load_dataset(args.test_file)
    print(f"[INFO] Loaded MMLU samples: {len(samples)}")
    kwargs = build_generation_kwargs(cfg)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_one, client, cfg["name"], sample, kwargs): sample for sample in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc="MMLU"):
            results.append(future.result())
    results.sort(key=lambda item: item["sample_id"])
    output = {"summary": summarize(results), "details": results}
    out = Path(args.output_file) if args.output_file else Path(args.output_dir) / f"{args.model_name}_mmlu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = output["summary"]
    print(f"[DONE] MMLU accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total']})")
    print(f"[DONE] Results saved to: {out}")


if __name__ == "__main__":
    main()
