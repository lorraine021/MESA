import argparse
import json
import os
import re
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import yaml
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def check_correctness(sample, completion, timeout=5):
    full_code = f"{sample['prompt']}\n{completion}\n{sample['test']}\ncheck({sample['entry_point']})"

    def unsafe_execute():
        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            return "passed"
        except Exception as e:
            return f"failed: {e}"

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    def worker(d):
        d['res'] = unsafe_execute()

    p = multiprocessing.Process(target=worker, args=(result_dict,))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        return 0, "timed out"

    res = result_dict.get('res', "failed")
    if res == "passed":
        return 1, "passed"
    else:
        return 0, res




def build_mbpp_prompt(sample):
    text = sample['text']
    test_list = sample['test_list']
    

    tests = '\n'.join(test_list)
    
    prompt = f"""Write executable Python code that solves this task: {text}

The code must pass these tests:
{tests}

Return only executable Python code. Begin immediately with [BEGIN], followed by the required function definition, and finish with [DONE]."""
    return prompt

def check_correctness_mbpp(sample, completion, timeout=5):

    setup_code = sample.get('test_setup_code', '') or ''
    test_list = sample['test_list']
    

    full_code = f"{setup_code}\n{completion}\n" + "\n".join(test_list)

    def unsafe_execute():
        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            return "passed"
        except Exception as e:
            return f"failed: {e}"

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    def worker(d):
        d['res'] = unsafe_execute()

    p = multiprocessing.Process(target=worker, args=(result_dict,))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        return 0, "timed out"

    res = result_dict.get('res', "failed")
    if res == "passed":
        return 1, "passed"
    else:
        return 0, res



def extract_code(text: str, dataset_type: str = "humaneval"):
    text = text.strip()


    if dataset_type == "mbpp":
        # Models may place only one marker inside a Markdown code fence. Remove
        # both markers globally before parsing so they can never be executed as
        # Python identifiers (for example, ``[DONE]`` -> NameError).
        text = re.sub(r"\[\s*(?:BEGIN|DONE)\s*\]", "", text, flags=re.IGNORECASE).strip()


        match = re.search(r"\[BEGIN\]\s*(.*?)\s*\[DONE\]", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        

        match_open = re.search(r"\[BEGIN\]\s*(.*)", text, re.DOTALL)
        if match_open:
            candidate = match_open.group(1).strip()
            if "[DONE]" not in candidate:
                return candidate



    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()


    match_open = re.search(r"```python\s*(.*)", text, re.DOTALL)
    if match_open:
        candidate = match_open.group(1).strip()
        if "```" not in candidate:
            return candidate


    match_simple = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match_simple:
        return match_simple.group(1).strip()


    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def "):
            start_idx = i
            break
    if start_idx is not None:
        code_lines = lines[start_idx:]
        cleaned = []
        for line in code_lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append(line)
                continue

            if not line.startswith((' ', '\t')) and not stripped.startswith(('def ', 'class ', 'import ', 'from ', '@')):
                break
            cleaned.append(line)
        return "\n".join(cleaned).strip()


    return text




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
    return {"name": model_name, "max_tokens": 1024}

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
    extra_body['min_tokens'] = int(model_cfg.get('min_tokens', 16))
    gen_kwargs['extra_body'] = extra_body
    gen_kwargs['max_tokens'] = model_cfg.get('max_tokens', 1024)
    gen_kwargs.setdefault('extra_body', {}).setdefault('chat_template_kwargs', {})['enable_thinking'] = False
    return gen_kwargs



def process_sample(sample, vllm_client, gen_kwargs, model_cfg, dataset_type="humaneval", max_retries=5):

    if dataset_type == "mbpp":
        prompt = build_mbpp_prompt(sample)
        task_id = sample["task_id"]
        system_content = (
            "You must produce a non-empty executable Python solution for every task. "
            "Do not stop, refuse, or emit EOS before giving the solution. "
            "First solve the task internally, then output only code. "
            "Begin immediately with [BEGIN] followed by the function definition and finish with [DONE]."
        )
    else:  # humaneval
        prompt = sample["prompt"]
        task_id = sample["task_id"]
        system_content = "You are a code completion assistant. Only output the completed Python function body. Do not include any explanations, markdown, or extra text."


    raw_output = ""
    code = ""
    last_error = ""


    for attempt in range(max_retries):
        try:
            resp = vllm_client.chat.completions.create(
                model=model_cfg["name"],
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                **gen_kwargs
            )
            

            if resp.choices and resp.choices[0].message.content:
                raw_output = resp.choices[0].message.content
            else:
                raw_output = ""


            if not raw_output or not raw_output.strip():
                print(f"[WARN] Task {task_id} output is empty. Retrying ({attempt + 1}/{max_retries})...")
                last_error = "Empty Output"
                time.sleep(1)
                continue
            

            code = extract_code(raw_output, dataset_type)
            

            if not code.strip():
                print(f"[WARN] Task {task_id} extracted code is empty (raw length: {len(raw_output)}). Retrying ({attempt + 1}/{max_retries})...")
                last_error = "Extraction Failed"
                continue


            break

        except Exception as e:
            print(f"[ERROR] Task {task_id} API Error: {e}. Retrying ({attempt + 1}/{max_retries})...")
            last_error = str(e)
            time.sleep(2)


    if not code.strip():
        return {
            "task_id": task_id,
            "completion": "",
            "raw_output": raw_output,
            "label": 0,
            "result_info": f"Failed after {max_retries} retries. Last error: {last_error}"
        }


    try:
        if dataset_type == "mbpp":
            is_correct, info = check_correctness_mbpp(sample, code)
        else:
            is_correct, info = check_correctness(sample, code)
        
        return {
            "task_id": task_id,
            "completion": code,
            "raw_output": raw_output,
            "label": is_correct,
            "result_info": info
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "completion": code,
            "raw_output": raw_output,
            "label": 0,
            "result_info": f"Eval Error: {str(e)}"
        }

        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--test-file", required=True, help="Path to your dataset file")
    parser.add_argument("--dataset-type", choices=["humaneval", "mbpp"], default="humaneval", 
                       help="Type of dataset: humaneval or mbpp")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel threads for inference")
    parser.add_argument("--config-dir", default=str(Path(__file__).resolve().parent / "config"))
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Load model config
    model_cfg = load_model_config(args.model_name, Path(args.config_dir))
    port = args.port
    vllm_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    
    # Generation parameters
    gen_kwargs = build_generation_kwargs(model_cfg)

    # Load dataset
    print(f"[INFO] Loading {args.dataset_type} dataset from {args.test_file}...")
    
    import pandas as pd
    df = pd.read_parquet(args.test_file)
    # Keep evaluation independent of the optional Hugging Face datasets stack.
    # pandas already decoded the local parquet file into the records we need.
    dataset = df.to_dict(orient="records")
    
    
    print(f"[INFO] Loaded {len(dataset)} {args.dataset_type.upper()} samples.")

    # Evaluate in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_sample, sample, vllm_client, gen_kwargs, model_cfg, args.dataset_type): sample
            for sample in dataset
        }
        
        for future in tqdm(as_completed(futures), total=len(dataset), desc=f"Evaluating {args.dataset_type.upper()}"):
            results.append(future.result())

    # Compute metrics
    correct = sum(r["label"] for r in results)
    accuracy = correct / len(dataset)

    output_data = {
        "summary": {
            "dataset_type": args.dataset_type,
            "total": len(dataset),
            "correct": correct,
            "accuracy": accuracy
        },
        "details": results
    }

    # Save results
    out_path = Path(args.output_file) if args.output_file else Path(args.output_dir) / f"{args.model_name}_{args.dataset_type}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {args.dataset_type.upper()} Pass@1: {accuracy:.2%} ({correct}/{len(dataset)})")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
