"""Validate inputs and launch either independent MESA architecture trainer."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {
    "deepseek": dict(model_type="deepseek_v2", layers=27, moe_layers=26, experts=64,
                     top_k=6, offset=1, trainer="MESADeepSeekTrainer", template="deepseek", attention="disabled"),
    "qwen": dict(model_type="qwen3_moe", layers=48, moe_layers=48, experts=128,
                 top_k=8, offset=0, trainer="MESAQwenTrainer", template="qwen3_nothink", attention="sdpa"),
}


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def prepare(args):
    config = args.config or f"configs/{args.architecture or 'deepseek'}_mesa.yaml"
    cfg = yaml.safe_load(local_path(config).read_text())
    if args.model:
        cfg["model_name_or_path"] = cfg["mesa_baseline_model"] = args.model
    if args.output:
        cfg["output_dir"] = args.output
    for key in ["model_name_or_path", "mesa_baseline_model", "freeze_expert_selection_file",
                "deepspeed", "dataset_dir", "output_dir"]:
        cfg[key] = str(local_path(cfg[key]).resolve())
    model = Path(cfg["model_name_or_path"])
    assert model == Path(cfg["mesa_baseline_model"]), "Initial and reference models must match."
    mc = json.loads((model / "config.json").read_text())
    architecture = next((name for name, spec in ARCHITECTURES.items()
                         if spec["model_type"] == mc.get("model_type")), None)
    assert architecture is not None, f"Unsupported model_type: {mc.get('model_type')!r}"
    if args.architecture:
        assert architecture == args.architecture, "Architecture disagrees with model config."
    spec = ARCHITECTURES[architecture]
    experts = mc.get("n_routed_experts") if architecture == "deepseek" else mc.get("num_experts")
    assert experts == spec["experts"] and mc["num_hidden_layers"] == spec["layers"]
    assert mc["num_experts_per_tok"] == spec["top_k"]
    assert cfg["freeze_expert_layer_offset"] == spec["offset"], "Wrong expert layer offset."
    assert cfg["freeze_trainable_layers"] == spec["layers"]
    assert cfg["template"] == spec["template"] and cfg["flash_attn"] == spec["attention"]
    if architecture == "deepseek":
        assert mc.get("aux_loss_alpha") == 0.001, "Expected DeepSeek aux_loss_alpha=0.001."
    index = model / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        assert shards and all((model / f).is_file() and (model / f).stat().st_size > 0 for f in shards), "Missing model shards."
    else:
        assert list(model.glob("*.safetensors")) or list(model.glob("pytorch_model*.bin")), "No model weights."
    selected = json.loads(Path(cfg["freeze_expert_selection_file"]).read_text())["selected_experts"]
    pairs = {(int(x["layer_id"]), int(x["expert_id"])) for x in selected}
    assert pairs and len(pairs) == len(selected), "Empty or duplicate expert selection."
    assert all(0 <= layer < spec["moe_layers"] and 0 <= expert < spec["experts"] for layer, expert in pairs)
    info = json.loads((Path(cfg["dataset_dir"]) / "dataset_info.json").read_text())
    for name in cfg["dataset"].split(","):
        assert (Path(cfg["dataset_dir"]) / info[name.strip()]["file_name"]).is_file()
    assert Path(cfg["deepspeed"]).is_file()
    assert cfg["use_mesa"] and cfg["finetuning_type"] == "freeze"
    assert cfg["disable_gradient_checkpointing"] and not cfg["use_reentrant_gc"]
    for key in ["mesa_router_loss_weight"]:
        assert math.isfinite(float(cfg[key])) and cfg[key] >= 0, f"Invalid {key}"
    print(json.dumps(dict(architecture=architecture, trainer=spec["trainer"],
        matching_KL="target || model", selected_experts=len(pairs),
        expert_fraction=len(pairs) / (spec["moe_layers"] * spec["experts"]),
        rw=cfg["mesa_router_loss_weight"],
        temperature=2.0, learning_rate=cfg["learning_rate"],
        native_aux_policy="DeepSeek alpha=0.001" if architecture == "deepseek" else "unchanged from Qwen config",
        config=cfg), indent=2), flush=True)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--config", help="Explicit YAML overrides the architecture preset.")
    parser.add_argument("--model", help="Local model directory; overrides initial and reference paths.")
    parser.add_argument("--output", help="Output path relative to the repository unless absolute.")
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC_PER_NODE", "8")))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/config without GPU allocation or training.")
    args = parser.parse_args()
    cfg = prepare(args)
    if args.dry_run:
        return
    assert args.nproc > 0
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Choose --output for a fresh run.")
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
               f"--nproc_per_node={args.nproc}", "--master_addr=127.0.0.1",
               f"--master_port={args.master_port}", str(ROOT / "scripts/train.py"), str(resolved)]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
