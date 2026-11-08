"""Run the verified DeepSeek or Qwen evaluation suite in the active evaluation environment."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPES = {"deepseek_v2": "deepseek", "qwen3_moe": "qwen"}


def local_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def gpu_ids(value):
    ids = [x.strip() for x in value.split(",")]
    if not ids or any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid GPU list: {value!r}")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local model or full checkpoint directory.")
    parser.add_argument("--judge", default=os.environ.get("OCTOPUS_MODEL_PATH"),
                        help="Local Octopus-SEval model directory.")
    parser.add_argument("--architecture", choices=["deepseek", "qwen"],
                        help="Normally inferred from the model config.")
    parser.add_argument("--data-root", default="eval_data")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--run-name", help="Result subdirectory name.")
    parser.add_argument("--target-gpus", default="0,1,2,3")
    parser.add_argument("--judge-gpus", default="4,5,6,7")
    parser.add_argument("--preflight-only", action="store_true", help="Check data and imports without allocating GPUs.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and settings without running evaluation.")
    args = parser.parse_args()
    if not args.preflight_only and not args.model:
        parser.error("--model is required unless --preflight-only is used")
    if not args.preflight_only and not args.dry_run and not args.judge:
        parser.error("--judge or OCTOPUS_MODEL_PATH is required")
    model = local_path(args.model) if args.model else None
    architecture = args.architecture
    if model:
        config = json.loads((model / "config.json").read_text())
        detected = MODEL_TYPES.get(config.get("model_type"))
        if not detected or (architecture and architecture != detected):
            raise ValueError(f"Unsupported or mismatched model_type: {config.get('model_type')!r}")
        architecture = detected
    architecture = architecture or "deepseek"
    judge = local_path(args.judge) if args.judge else None
    if judge and not (judge / "config.json").is_file():
        raise FileNotFoundError(judge / "config.json")
    data = local_path(args.data_root)
    if not data.is_dir():
        raise FileNotFoundError(data)
    target_gpus, judge_gpus = gpu_ids(args.target_gpus), gpu_ids(args.judge_gpus)
    if set(target_gpus) & set(judge_gpus):
        raise ValueError("Target and judge GPU lists must not overlap")
    name = args.run_name or (f"{model.parent.name}_{model.name}" if model else "preflight")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError("--run-name must contain only letters, digits, '.', '_' or '-'")
    results = local_path(args.output_dir)
    settings = dict(architecture=architecture, model_path=str(model) if model else None,
                    judge_path=str(judge) if judge else None, data_root=str(data),
                    results=str(results / name), target_gpus=target_gpus, judge_gpus=judge_gpus,
                    math_judge="exact", safety_policy="empty_response_safe_v1", python=sys.executable)
    print(json.dumps(settings, indent=2), flush=True)
    if args.dry_run:
        return
    packages = ("torch", "transformers", "vllm", "openai", "pandas", "pyarrow",
                "PyYAML", "tqdm", "math-verify", "latex2sympy2-extended", "sympy")
    versions = {name: version(name) for name in packages}
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("NCCL_CUMEM_ENABLE", None)
    env.update(EVAL_ARCHITECTURE=architecture, EVAL_PYTHON=sys.executable,
               VLLM_BIN=shutil.which("vllm") or str(Path(sys.executable).parent / "vllm"),
               DATA_ROOT=str(data), RESULTS_ROOT=str(results),
               TARGET_GPUS=",".join(target_gpus), OCTOPUS_GPUS=",".join(judge_gpus),
               TARGET_TP_SIZE=str(len(target_gpus)), OCTOPUS_TP_SIZE=str(len(judge_gpus)),
               OCTOPUS_DISABLE_CUSTOM_ALL_REDUCE="1", MATH_JUDGE_MODE="exact")
    # Use the currently active environment; never reactivate a machine-specific venv.
    env.pop("EVAL_ACTIVATE", None)
    if args.preflight_only:
        import subprocess
        for path in (ROOT / "evaluation" / architecture).glob("*.py"):
            subprocess.run([sys.executable, str(path), "--help"], env=env,
                           stdout=subprocess.DEVNULL, check=True)
        command = ["bash", str(ROOT / "evaluation/run_suite.sh"), "--preflight-only"]
    else:
        manifest = results / name / "evaluation_config.json"
        if manifest.is_file():
            previous = json.loads(manifest.read_text())
            if any(previous.get(key) != settings[key] for key in ("architecture", "model_path", "judge_path", "data_root")):
                raise ValueError("Existing results belong to a different evaluation; choose a new --run-name")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(dict(settings, versions=versions), indent=2))
        command = ["bash", str(ROOT / "evaluation/run_suite.sh"), str(model), name, str(judge)]
    os.execvpe("bash", command, env)


if __name__ == "__main__":
    main()
