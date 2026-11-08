"""Collect the six general and three safety benchmark summaries."""
import argparse
import json
from datetime import datetime
from pathlib import Path

GENERAL_TASKS = ("math500", "gsm8k", "humaneval", "mbpp", "gpqa", "mmlu")
SAFETY_TASKS = ("strongreject", "strata_sword_en_level_3", "wildjailbreak")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    tasks = {}
    for name in GENERAL_TASKS:
        data = json.loads((root / "general" / f"{name}.json").read_text())
        tasks[name] = data.get("summary", {})
    safety = json.loads((root / "safety_octopus.json").read_text())
    if set(safety) != set(SAFETY_TASKS):
        raise ValueError(f"Expected safety benchmarks {SAFETY_TASKS}; found {tuple(safety)}")
    tasks["safety"] = {name: data.get("summary", {}) for name, data in safety.items()}
    result = {
        "run_name": args.run_name,
        "model_path": args.model_path,
        "created_at": datetime.now().isoformat(),
        "tasks": tasks,
        "safety_evaluation_policy": "empty_response_safe_v1",
    }
    out = root / "summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[DONE] Summary saved to {out}", flush=True)


if __name__ == "__main__":
    main()
