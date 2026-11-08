"""Select Qwen experts using the same float64 OT numerics as DeepSeek."""
import argparse
import json
from pathlib import Path

import numpy as np

from select_deepseek_experts import select_experts


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--percentage', type=float, default=10.)
    parser.add_argument('--output', type=Path, default=root/'experts/qwen_experts_10pct_reselected.json')
    parser.add_argument('--source', default=None)
    parser.add_argument('--epsilon', type=float, default=.01)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--cost-alpha', type=float, default=1.)
    parser.add_argument('--cost-beta', type=float, default=2.)
    parser.add_argument('--shift-value', type=float, default=20.)
    args = parser.parse_args()
    records = json.loads(args.trace.read_text())
    usage = None
    sources = {}
    for index, row in enumerate(records):
        if args.source is not None and row.get('source') != args.source:
            continue
        counts = np.asarray(row['input_expert_usage'], dtype=np.float64)
        if counts.shape != (48, 128) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(f'Record {index}: expected finite nonnegative Qwen [48,128] counts.')
        usage = counts.copy() if usage is None else usage + counts
        name = row.get('source', 'unknown')
        sources[name] = sources.get(name, 0) + 1
    if usage is None:
        raise ValueError('No matching trace records.')
    if args.epsilon <= 0 or args.max_iter < 1:
        raise ValueError('Invalid solver settings.')
    chosen, plan = select_experts(usage, args.percentage, args.epsilon, args.max_iter,
                                  args.cost_alpha, args.cost_beta, args.shift_value)
    # The shared DeepSeek function returns model_layer_id=logical+1. Qwen has no
    # initial dense layer, so its model layer ID equals its logical MoE layer ID.
    for expert in chosen:
        expert['model_layer_id'] = expert['layer_id']
    assert len(chosen) == int(48 * 128 * args.percentage / 100.)
    output = dict(selected_experts=chosen, total_count=len(chosen),
                  description=f'Top {args.percentage:g}% Qwen routed experts selected by float64 MESA OT',
                  metadata=dict(trace_path=str(args.trace), source=args.source or 'MIXED',
                                source_counts=sources, percentage=args.percentage,
                                logical_to_model_layer_offset=0, dtype='float64',
                                rank_clipping='before_cost:[0.5,99.5]',
                                denominator_protection='clamp_min(1e-30)',
                                epsilon=args.epsilon, max_iter=args.max_iter,
                                cost_alpha=args.cost_alpha, cost_beta=args.cost_beta,
                                shift_value=args.shift_value, transport_mass=float(plan.sum())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    temporary.replace(args.output)
    print(f'Selected {len(chosen)} experts; saved to {args.output}')


if __name__ == '__main__':
    main()
