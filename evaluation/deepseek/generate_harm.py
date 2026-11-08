# generate_responses_parallel.py

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

CONFIG_DIR = Path(__file__).resolve().parent / "config"
RESULT_DIR = Path("./results_harm")

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
    for field in ("temperature", "top_p"):
        if field in model_cfg:
            gen_kwargs[field] = model_cfg[field]
    if model_cfg.get("do_sample") is False:
        gen_kwargs.setdefault("temperature", 0.0)
    extra_body = {"min_tokens": 5}
    for field in ("top_k", "repetition_penalty", "min_p"):
        if field in model_cfg:
            extra_body[field] = model_cfg[field]
    if extra_body:
        gen_kwargs["extra_body"] = extra_body
    gen_kwargs["max_tokens"] = model_cfg.get("max_tokens", 4096)
    return gen_kwargs

def load_dataset(file_path, dataset_name=None):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_parquet(path)
    name = (dataset_name or path.as_posix()).lower().replace("-", "").replace("_", "")

    def require(*columns):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns {missing}; available={list(df.columns)}")

    if "wildjailbreak" in name:
        require("prompt", "label")
        df = df[df["label"].astype(str).str.lower() == "adversarial_harmful"]
    elif "jailbreakbench" in name:
        require("prompt", "subset")
        df = df[df["subset"].astype(str).str.lower() == "harmful"]
    elif "strongreject" in name:
        require("prompt")
    elif "stratasword" in name or "strata" in name:
        require("\u653b\u51fb\u6307\u4ee4", "attack method", "\u98ce\u9669\u5206\u7c7b")
        df = df.rename(columns={"\u653b\u51fb\u6307\u4ee4": "prompt"})
    else:
        require("prompt")

    prompts = [str(x) for x in df["prompt"].dropna().tolist() if str(x).strip()]
    if not prompts:
        raise ValueError(f"No evaluation prompts loaded from {path} for dataset={dataset_name}")
    return [{"prompt": p} for p in prompts]


def generate_one(client, model_name, prompt, gen_kwargs, max_tries=3):
    last_error = None
    for attempt in range(max_tries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **gen_kwargs
            )
            response_content = resp.choices[0].message.content
            

            if not response_content or not response_content.strip():
                if attempt < max_tries - 1:
                    print(f"[WARNING] Empty response on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    return {"prompt": prompt, "response": "", "error": f"Empty response after {max_tries} attempts"}
            
            return {"prompt": prompt, "response": response_content}
        except Exception as e:
            last_error = e
            if attempt < max_tries - 1:
                print(f"[WARNING] Generation attempt {attempt + 1} failed: {e}, retrying...")
            else:
                return {"prompt": prompt, "response": "", "error": f"Error after {max_tries} attempts: {str(e)}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-tries", type=int, default=3, help="Maximum number of retries for generation")
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--data-paths", nargs="+", required=True)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--plain-output-names", action="store_true")
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_name, Path(args.config_dir))
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")
    gen_kwargs = build_generation_kwargs(model_cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for item in args.data_paths:
        name, path = item.split(":", 1)
        print(f"\n🚀 Processing dataset: {name}")
        samples = load_dataset(path, name)

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_prompt = {
                executor.submit(generate_one, client, model_cfg["name"], s["prompt"], gen_kwargs, args.max_tries): s["prompt"]
                for s in samples
            }

            for future in tqdm(as_completed(future_to_prompt), total=len(samples), desc=name):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    prompt = future_to_prompt[future]
                    results.append({"prompt": prompt, "response": "", "error": str(exc)})

        out_file = output_dir / (f"{name}.json" if args.plain_output_names else f"{args.model_name}_{name}_responses.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Responses saved to: {out_file}")

if __name__ == "__main__":
    main()