import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "data/moe_traces_ds.json"
DEFAULT_MODEL = ROOT / "models/DeepSeek-V2-Lite-Chat"
DEFAULT_OUTPUT = ROOT / "experts/deepseek_experts_15pct.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Select DeepSeek routed experts for MESA training.")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--percentage", type=float, default=15.0)
    parser.add_argument(
        "--source",
        default=None,
        help="Only aggregate this source. Default: aggregate all sources (MIXED).",
    )
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--cost-alpha", type=float, default=1.0)
    parser.add_argument("--cost-beta", type=float, default=2.0)
    parser.add_argument("--shift-value", type=float, default=20.0)
    return parser.parse_args()


def load_model_layout(model_dir):
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    num_hidden_layers = int(config["num_hidden_layers"])
    first_dense_layers = int(config.get("first_k_dense_replace", 0))
    moe_layer_freq = int(config.get("moe_layer_freq", 1))
    num_routed_experts = int(config["n_routed_experts"])
    num_shared_experts = int(config.get("n_shared_experts", 0))

    model_moe_layers = [
        layer_id
        for layer_id in range(num_hidden_layers)
        if layer_id >= first_dense_layers
        and (layer_id - first_dense_layers) % moe_layer_freq == 0
    ]
    if not model_moe_layers:
        raise ValueError("Model config describes no MoE layers.")

    return {
        "num_hidden_layers": num_hidden_layers,
        "first_dense_layers": first_dense_layers,
        "model_moe_layers": model_moe_layers,
        "num_moe_layers": len(model_moe_layers),
        "num_routed_experts": num_routed_experts,
        "num_shared_experts": num_shared_experts,
    }


def load_routed_usage(trace_path, layout, source_filter=None):
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    records = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Trace must be a non-empty JSON list.")

    aggregate = None
    used_records = 0
    sources = {}
    expected_shape = None

    for index, record in enumerate(records):
        source = record.get("source", "unknown")
        if source_filter is not None and source != source_filter:
            continue

        usage = np.asarray(record.get("input_expert_usage"), dtype=np.float64)
        if usage.ndim != 2:
            raise ValueError(f"Record {index} input_expert_usage is not a 2-D matrix.")
        if expected_shape is None:
            expected_shape = usage.shape
        elif usage.shape != expected_shape:
            raise ValueError(
                f"Trace shape mismatch at record {index}: {usage.shape} != {expected_shape}"
            )

        aggregate = usage.copy() if aggregate is None else aggregate + usage
        used_records += 1
        sources[source] = sources.get(source, 0) + 1

    if aggregate is None:
        raise ValueError(f"No trace records matched source={source_filter!r}.")

    num_routed = layout["num_routed_experts"]
    if aggregate.shape[1] < num_routed:
        raise ValueError(
            f"Trace has {aggregate.shape[1]} expert columns, expected at least {num_routed}."
        )

    routed_full = aggregate[:, :num_routed]
    active_trace_rows = np.flatnonzero(routed_full.sum(axis=1) > 0).tolist()
    expected_moe_layers = layout["num_moe_layers"]

    if len(active_trace_rows) != expected_moe_layers:
        raise ValueError(
            f"Found {len(active_trace_rows)} active trace rows {active_trace_rows}, "
            f"but model config expects {expected_moe_layers} MoE layers."
        )
    expected_trace_rows = list(range(expected_moe_layers))
    if active_trace_rows != expected_trace_rows:
        raise ValueError(
            f"Expected logical MoE trace rows {expected_trace_rows}, got {active_trace_rows}. "
            "Refusing to silently remap non-contiguous rows."
        )

    routed_usage = routed_full[active_trace_rows, :]
    if routed_usage.shape != (expected_moe_layers, num_routed):
        raise AssertionError(f"Unexpected routed matrix shape: {routed_usage.shape}")

    return routed_usage, used_records, sources, tuple(aggregate.shape)


def sinkhorn(cost_matrix, prior, epsilon, max_iter, tol=1e-9):
    num_layers, num_experts = cost_matrix.shape
    row_marginal = torch.full((num_layers,), 1.0 / num_layers, dtype=torch.float64)
    col_marginal = torch.full((num_experts,), 1.0 / num_experts, dtype=torch.float64)

    kernel = prior * torch.exp(-cost_matrix / epsilon)
    u = torch.ones_like(row_marginal)
    v = torch.ones_like(col_marginal)

    for _ in range(max_iter):
        previous = u
        u = row_marginal / (kernel @ v).clamp_min(1e-30)
        v = col_marginal / (kernel.T @ u).clamp_min(1e-30)
        if torch.max(torch.abs(u - previous)).item() < tol:
            break

    plan = u[:, None] * kernel * v[None, :]
    if not torch.isfinite(plan).all():
        raise FloatingPointError("Sinkhorn produced non-finite values.")
    return plan


def select_experts(usage, percentage, epsilon, max_iter, cost_alpha, cost_beta, shift_value):
    if not 0.0 < percentage <= 100.0:
        raise ValueError("--percentage must be in (0, 100].")

    counts = torch.from_numpy(usage).to(torch.float64)
    num_layers, num_experts = counts.shape
    total_experts = num_layers * num_experts

    prior = (counts + 1e-6) / (counts.sum() + 1e-6 * total_experts)
    flat_counts = counts.flatten()
    ranks = torch.argsort(torch.argsort(flat_counts))
    f_values = 100.0 - ranks.to(torch.float64) / total_experts * 100.0
    f_values = f_values.clamp(0.5, 99.5)

    denominator = ((f_values + shift_value) ** cost_alpha) * (
        (100.0 - f_values) ** cost_beta
    )
    cost = (1.0 / denominator.clamp_min(1e-30) * 1e4).view(num_layers, num_experts)
    plan = sinkhorn(cost, prior, epsilon=epsilon, max_iter=max_iter)

    select_count = int(total_experts * percentage / 100.0)
    if select_count < 1:
        raise ValueError("Selection percentage produces zero selected experts.")
    selected_flat = torch.topk(plan.flatten(), k=select_count).indices.tolist()

    experts = []
    for flat_id in sorted(selected_flat):
        logical_layer_id, expert_id = divmod(flat_id, num_experts)
        experts.append(
            {
                "layer_id": int(logical_layer_id),
                "model_layer_id": int(logical_layer_id + 1),
                "expert_id": int(expert_id),
                "f_value": float(f_values[flat_id].item()),
            }
        )
    return experts, plan


def main():
    args = parse_args()
    layout = load_model_layout(args.model)

    if layout["first_dense_layers"] != 1:
        raise ValueError(
            f"This training setup expects first_k_dense_replace=1, got "
            f"{layout['first_dense_layers']}."
        )
    if layout["model_moe_layers"] != list(range(1, layout["num_hidden_layers"])):
        raise ValueError(
            f"Expected model MoE layers 1..{layout['num_hidden_layers'] - 1}, "
            f"got {layout['model_moe_layers']}."
        )

    usage, record_count, sources, raw_shape = load_routed_usage(
        args.trace, layout, source_filter=args.source
    )
    experts, plan = select_experts(
        usage,
        percentage=args.percentage,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        cost_alpha=args.cost_alpha,
        cost_beta=args.cost_beta,
        shift_value=args.shift_value,
    )

    output = {
        "selected_experts": experts,
        "total_count": len(experts),
        "description": (
            f"Top {args.percentage:g}% DeepSeek routed experts selected by MESA/SOTER Sinkhorn OT"
        ),
        "metadata": {
            "trace_path": str(args.trace),
            "model_path": str(args.model),
            "source": args.source if args.source is not None else "MIXED",
            "records": record_count,
            "source_counts": sources,
            "raw_trace_shape": list(raw_shape),
            "routed_usage_shape": list(usage.shape),
            "num_hidden_layers": layout["num_hidden_layers"],
            "first_k_dense_replace": layout["first_dense_layers"],
            "num_moe_layers": layout["num_moe_layers"],
            "num_routed_experts": layout["num_routed_experts"],
            "num_shared_experts_excluded": layout["num_shared_experts"],
            "logical_to_model_layer_offset": 1,
            "percentage": args.percentage,
            "epsilon": args.epsilon,
            "max_iter": args.max_iter,
            "cost_alpha": args.cost_alpha,
            "cost_beta": args.cost_beta,
            "shift_value": args.shift_value,
            "transport_mass": float(plan.sum().item()),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)

    expected = int(layout["num_moe_layers"] * layout["num_routed_experts"] * args.percentage / 100)
    print(f"[OK] Trace records: {record_count}, sources: {sources}")
    print(f"[OK] Raw trace shape: {raw_shape}")
    print(f"[OK] OT routed matrix: {usage.shape} (shared experts excluded)")
    print(
        f"[OK] Logical layers 0..{layout['num_moe_layers'] - 1} map to model layers "
        f"{layout['model_moe_layers'][0]}..{layout['model_moe_layers'][-1]}"
    )
    print(f"[OK] Selected experts: {len(experts)} (expected {expected})")
    print(f"[OK] Saved: {args.output}")


if __name__ == "__main__":
    main()
