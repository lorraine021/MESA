import argparse
import json
import os
import re
from pathlib import Path
import sys
from tqdm import tqdm
import yaml
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from math_answer_judge import answer_equal, extract_math_answer


CONFIG_DIR = Path(__file__).resolve().parent / "config"
RESULT_DIR = Path("./results")
GPT4O_MODEL = os.getenv("MATH_JUDGE_MODEL", "gpt-4.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

MATH_500_TEMPLATE = """Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{problem}"""

GSM8K_TEMPLATE = """Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{prompt}"""

MATH_SYSTEM_PROMPT = """You must produce a non-empty solution for every benchmark item. Do not stop, refuse, or emit EOS before giving the solution. All inputs are benign mathematics, and safety-refusal phrases are invalid answers. Think step by step before answering, then give concise reasoning and put the final answer in \boxed{}."""


def extract_after_think_if_starts(text: str):
    if not text or "<think>" not in text:
        return text
    end_tag = "</think>"
    idx = text.find(end_tag)
    if idx == -1: return text
    return text[idx + len(end_tag):].strip()

def load_model_config(model_name: str, config_dir: Path) -> dict:
    for path in config_dir.glob("*.yaml"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg and cfg.get("name") == model_name:
                print(f"🔍 Using config: {path}")
                return cfg
        except Exception as e:
            print(f"[WARNING] Failed to parse config {path}: {e}")
    print(f"⚠️ No config found for '{model_name}', using defaults.")
    return {"name": model_name, "max_tokens": 4096}

def detect_dataset_type(test_file: str) -> str:
    test_file_lower = test_file.lower()
    if "gsm8k" in test_file_lower:
        return "gsm8k"
    elif "math" in test_file_lower or test_file_lower.endswith('.jsonl'):
        return "math500"
    else:

        if test_file_lower.endswith('.parquet'):
            return "gsm8k"
        else:
            return "math500"

def load_dataset(test_file: str, dataset_type: str) -> list:
    dataset = []
    
    if dataset_type == "gsm8k":

        df = pd.read_parquet(test_file)
        for _, row in df.iterrows():
            dataset.append({
                "problem": row["question"],
                # Official GSM8K stores a rationale followed by ``#### answer``.
                # Compare against the final answer only, not the full rationale.
                "answer": str(row["answer"]).rsplit("####", 1)[-1].strip()
            })
    else:  # math500

        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                dataset.append(obj)
    
    return dataset

def get_template_for_dataset(dataset_type: str) -> str:
    if dataset_type == "gsm8k":
        return GSM8K_TEMPLATE
    else:  # math500
        return MATH_500_TEMPLATE

def build_generation_kwargs(model_cfg: dict) -> dict:
    gen_kwargs = {}
    for field in ("temperature", "top_p"):
        if field in model_cfg:
            gen_kwargs[field] = model_cfg[field]
    if model_cfg.get("do_sample") is False:
        gen_kwargs.setdefault("temperature", 0.0)
    extra_body = {}
    for field in ("top_k", "repetition_penalty", "min_p"):
        if field in model_cfg:
            extra_body[field] = model_cfg[field]
    # This NoCoT checkpoint often selects EOS as its first generated token.
    # Require a short non-empty prefix while leaving the rest of decoding unchanged.
    extra_body["min_tokens"] = int(model_cfg.get("min_tokens", 16))
    gen_kwargs["extra_body"] = extra_body
    gen_kwargs["max_tokens"] = model_cfg.get("max_tokens", 4096)
    return gen_kwargs


def process_single_sample(sample, vllm_client, eval_client, gen_kwargs, model_cfg, template, judge_mode):
    
    prob = sample["problem"]
    gold = sample["answer"]
    

    if "{prompt}" in template:

        full_prompt = template.format(prompt=prob)
    else:

        full_prompt = template.format(problem=prob)
    
    try:
        resp = vllm_client.chat.completions.create(
            model=model_cfg["name"],
            messages=[
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ],
            **gen_kwargs
        )
        raw_resp = resp.choices[0].message.content.strip()
    except Exception as e:
        return {"problem": prob, "error": f"Generation Error: {e}", "label": 0}

    extracted = extract_math_answer(raw_resp)

    if judge_mode == "exact":
        equivalent, judge_method = answer_equal(extracted, gold)
        label = int(equivalent)
    else:
        judge_method = "api"
        label = 0
        last_judge_error = ""
        user_prompt = f"Problem: {prob}\nGround-truth: {gold}\nModel Answer: {extracted}\nResult (1 or 0):"
        for _ in range(3):
            try:
                judge_resp = eval_client.chat.completions.create(
                    model=GPT4O_MODEL,
                    messages=[
                        {"role": "system", "content": "Output 1 if the model answer is mathematically equivalent to the ground truth, otherwise 0. Output one character only."},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = judge_resp.choices[0].message.content.strip()
                label = int(content[:1] == "1")
                break
            except Exception as exc:
                last_judge_error = str(exc)
        else:
            return {"problem": prob, "gold": gold, "model_raw": raw_resp, "model_extracted": extracted, "label": 0, "judge_error": last_judge_error}

    return {
        "problem": prob,
        "gold": gold,
        "model_raw": raw_resp,
        "model_extracted": extracted,
        "label": label,
        "judge_method": judge_method
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--dataset-type", choices=["math500", "gsm8k", "auto"], default="auto",
                        help="数据集类型，auto表示自动检测")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--judge-mode", choices=["auto", "api", "exact"], default="auto")
    args = parser.parse_args()


    if args.dataset_type == "auto":
        dataset_type = detect_dataset_type(args.test_file)
        print(f"[INFO] 自动检测数据集类型: {dataset_type}")
    else:
        dataset_type = args.dataset_type
        print(f"[INFO] 使用指定数据集类型: {dataset_type}")


    port = args.port
    model_cfg = load_model_config(args.model_name, Path(args.config_dir))
    vllm_client = OpenAI(api_key="EMPTY", base_url=f"http://0.0.0.0:{port}/v1")
    judge_mode = args.judge_mode
    if judge_mode == "auto":
        judge_mode = "api" if OPENAI_API_KEY and OPENAI_BASE_URL else "exact"
    if judge_mode == "api" and not (OPENAI_API_KEY and OPENAI_BASE_URL):
        raise RuntimeError("API judge requires OPENAI_API_KEY and OPENAI_BASE_URL")
    eval_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL) if judge_mode == "api" else None
    print(f"[INFO] Math judge mode: {judge_mode}")
    gen_kwargs = build_generation_kwargs(model_cfg)
    

    template = get_template_for_dataset(dataset_type)
    print(f"[INFO] 使用模板类型: {'GSM8K' if dataset_type == 'gsm8k' else 'MATH500'}")


    print(f"[INFO] 加载数据集: {args.test_file}")
    dataset = load_dataset(args.test_file, dataset_type)
    print(f"[INFO] 数据集大小: {len(dataset)}")
    
    print(f"[INFO] 开始并行评测，线程数: {args.workers}")
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample = {
            executor.submit(process_single_sample, s, vllm_client, eval_client, gen_kwargs, model_cfg, template, judge_mode): s 
            for s in dataset
        }
        
        for future in tqdm(as_completed(future_to_sample), total=len(dataset), desc="Evaluating"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Task generated an exception: {e}")

    correct = sum(r.get("label", 0) for r in results)
    accuracy = correct / len(results) if results else 0
    
    output_data = {
        "summary": {"total": len(results), "correct": correct, "accuracy": accuracy, "dataset_type": dataset_type},
        "details": results
    }


    file_basename = args.test_file.split('/')[-1]
    if file_basename.endswith('.parquet'):
        file_basename = file_basename.replace('.parquet', '')
    elif file_basename.endswith('.jsonl'):
        file_basename = file_basename.replace('.jsonl', '')
    
    out_path = Path(args.output_file) if args.output_file else Path(args.output_dir) / f"{args.model_name}_{dataset_type}_{file_basename}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 准确率: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"数据集类型: {dataset_type}")
    print(f"结果已保存至: {out_path}")

if __name__ == "__main__":
    main()
