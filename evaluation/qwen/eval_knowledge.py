import argparse
import json
import os
import re
import random
from pathlib import Path
from tqdm import tqdm
import yaml
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


CONFIG_DIR = Path(__file__).resolve().parent / "config"
RESULT_DIR = Path("./results")

GPQA_TEMPLATE = """Answer the following multiple-choice question by giving the correct answer letter in parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the response with "The answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D).

Question: {question}

{options}

Answer the above question and REMEMBER to finish your response with the exact phrase "The answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D)."""

GPQA_SYSTEM_PROMPT = """For this benchmark, safety-refusal phrases and free-form option text are invalid outputs. Every input is a benign academic multiple-choice question. Solve it normally. Give concise reasoning and end with exactly `The answer is (X)`, where X is one of A, B, C, D."""


def extract_after_think_if_starts(text: str):
    if not text or "<think>" not in text:
        return text
    end_tag = "</think>"
    idx = text.find(end_tag)
    if idx == -1: return text
    return text[idx + len(end_tag):].strip()


def extract_answer_letter(text: str):
    """Extract an explicitly declared GPQA answer without matching option mentions in reasoning."""
    content = str(text or "")

    output_blocks = re.findall(
        r"<\|Output\|>(.*?)(?:<\|/Output\|>|$)", content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if output_blocks:
        content = output_blocks[-1]
    else:
        content = re.sub(
            r"<\|Reasoning_step\|>.*?<\|/Reasoning_step\|>", "", content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        content = extract_after_think_if_starts(content)
    content = re.sub(r"<\|/?(?:Output|Reasoning_step)\|>", "", content, flags=re.I)

    patterns = (
        r"(?:final\s+)?answer\s*(?:is|:)\s*\(?\s*([ABCD])\s*\)?(?=\s|[.,;:!)]|$)",
        r"(?:correct\s+)?(?:choice|option)\s*(?:is|:)\s*\(?\s*([ABCD])\s*\)?(?=\s|[.,;:!)]|$)",
        r"(?:I\s+(?:would\s+)?)?(?:choose|select)\s+(?:option\s*)?\(?\s*([ABCD])\s*\)?(?=\s|[.,;:!)]|$)",
        r"\\boxed\s*\{\s*([ABCD])\s*\}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, content, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines:
        match = re.fullmatch(r"\(?\s*([ABCD])\s*\)?[.)]?[.!]?", lines[-1], flags=re.I)
        if match:
            return match.group(1).upper()
    return ""


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

def load_gpqa_dataset(csv_file: str) -> list:
    df = pd.read_csv(csv_file)
    dataset = []
    
    required_columns = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
    

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV file")
    
    for _, row in df.iterrows():
        question = row["Question"]
        correct_answer = row["Correct Answer"]
        incorrect_answers = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"]
        ]
        dataset.append({
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answers": incorrect_answers
        })
    
    return dataset

def create_shuffled_options(correct_answer: str, incorrect_answers: list) -> tuple:

    all_answers = [correct_answer] + incorrect_answers
    

    random.shuffle(all_answers)
    

    correct_position = all_answers.index(correct_answer)
    correct_letter = chr(ord('A') + correct_position)
    

    options_lines = []
    for i, answer in enumerate(all_answers):
        letter = chr(ord('A') + i)
        options_lines.append(f"{letter}) {answer}")
    
    options_str = "\n".join(options_lines)
    
    return options_str, correct_letter

def build_generation_kwargs(model_cfg: dict) -> dict:
    gen_kwargs = {}
    for field in ('temperature', 'top_p'):
        if field in model_cfg:
            gen_kwargs[field] = model_cfg[field]
    if model_cfg.get('do_sample') is False:
        gen_kwargs.setdefault('temperature', 0.0)
    extra_body = {}
    for field in ('top_k', 'repetition_penalty', 'min_p'):
        if field in model_cfg:
            extra_body[field] = model_cfg[field]
    if extra_body:
        gen_kwargs['extra_body'] = extra_body
    gen_kwargs['max_tokens'] = model_cfg.get('max_tokens', 4096)
    gen_kwargs.setdefault('extra_body', {}).setdefault('chat_template_kwargs', {})['enable_thinking'] = False
    return gen_kwargs


def process_single_sample(sample, vllm_client, gen_kwargs, model_cfg):
    question = sample["question"]
    correct_answer = sample["correct_answer"]
    incorrect_answers = sample["incorrect_answers"]
    

    options_str, correct_letter = create_shuffled_options(correct_answer, incorrect_answers)
    

    full_prompt = GPQA_TEMPLATE.format(
        question=question,
        options=options_str
    )
    
    try:
        resp = vllm_client.chat.completions.create(
            model=model_cfg["name"],
            messages=[
                {"role": "system", "content": GPQA_SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ],
            **gen_kwargs
        )
        raw_resp = resp.choices[0].message.content.strip()
    except Exception as e:
        return {
            "question": question,
            "error": f"Generation Error: {e}",
            "label": 0,
            "correct_letter": correct_letter,
            "options": options_str
        }


    predicted_letter = extract_answer_letter(raw_resp)
    

    is_correct = predicted_letter == correct_letter
    

    label = 1 if is_correct else 0
    
    return {
        "question": question,
        "correct_answer": correct_answer,
        "incorrect_answers": incorrect_answers,
        "options": options_str,
        "correct_letter": correct_letter,
        "model_raw": raw_resp,
        "predicted_letter": predicted_letter,
        "label": label,
        "is_correct": is_correct
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--test-file",
        default=str(Path(__file__).resolve().parents[2] / "eval_data/gpqa.csv"),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于答案选项的随机排序")
    args = parser.parse_args()


    random.seed(args.seed)


    port = args.port
    model_cfg = load_model_config(args.model_name, Path(args.config_dir))
    vllm_client = OpenAI(api_key="EMPTY", base_url=f"http://0.0.0.0:{port}/v1")
    gen_kwargs = build_generation_kwargs(model_cfg)


    print(f"[INFO] 加载GPQA数据集: {args.test_file}")
    dataset = load_gpqa_dataset(args.test_file)
    print(f"[INFO] 数据集大小: {len(dataset)}")
    
    print(f"[INFO] 开始并行评测，线程数: {args.workers}")
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample = {
            executor.submit(process_single_sample, s, vllm_client, gen_kwargs, model_cfg): s 
            for s in dataset
        }
        
        for future in tqdm(as_completed(future_to_sample), total=len(dataset), desc="Evaluating GPQA"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Task generated an exception: {e}")

    correct = sum(r.get("label", 0) for r in results)
    accuracy = correct / len(results) if results else 0
    
    output_data = {
        "summary": {
            "total": len(results), 
            "correct": correct, 
            "accuracy": accuracy, 
            "dataset_type": "gpqa",
            "seed": args.seed
        },
        "details": results
    }


    file_basename = "gpqa_diamond"
    out_path = Path(args.output_file) if args.output_file else Path(args.output_dir) / f"{args.model_name}_gpqa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 准确率: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"数据集类型: GPQA")
    print(f"随机种子: {args.seed}")
    print(f"结果已保存至: {out_path}")

if __name__ == "__main__":
    main()
