"""qwen routed-expert trace using vLLM eager execution.
Preserves historical prefill+decode counts and logical layer indexing.
This is not a prompt-only trace; max_new defaults to 64 for legacy compatibility.
"""
MODEL_TYPE='qwen3_moe'
SAMPLE_LIMIT=500
EXPECTED_MOE_LAYERS=48
LAYER_OFFSET=0
import os
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
import re
import json
import torch
import random
import argparse
import pathlib
import numpy as np
from vllm import LLM, SamplingParams

def get_num_experts(cfg: dict) -> int:
    if 'num_experts' in cfg:
        return int(cfg['num_experts'])
    if 'num_local_experts' in cfg:
        return int(cfg['num_local_experts'])
    if 'n_routed_experts' in cfg:
        routed = int(cfg['n_routed_experts'])
        shared = int(cfg.get('n_shared_experts', 0))
        return routed + shared
    raise KeyError('Could not find expert count field in model config.')

def install_hooks(exe, num_layers, num_experts, topk, arch):
    """Install MoE gate hooks inside a vLLM worker."""
    if arch == 'mixtral':
        pattern = re.compile('\\.block_sparse_moe\\.(?:gate|router)$')
    else:
        pattern = re.compile('\\.mlp\\.gate$')
    exe._trace_in = torch.zeros(num_layers, num_experts, dtype=torch.int32, device=exe.device)
    exe._sum_in = torch.zeros(num_layers, num_experts, dtype=torch.float32, device=exe.device)

    def make_hook(layer_idx):

        def hook(_module, _inputs, output):
            logits = output[0] if isinstance(output, (tuple, list)) else output
            prob = torch.softmax(logits, dim=-1)
            (values, indices) = prob.topk(topk, dim=-1)
            flat_idx = indices.reshape(-1, topk)
            flat_val = values.reshape(-1, topk)
            for token_idx in range(flat_idx.size(0)):
                for k in range(topk):
                    expert_id = flat_idx[token_idx, k].item()
                    gate_prob = flat_val[token_idx, k].item()
                    exe._trace_in[layer_idx, expert_id] += 1
                    exe._sum_in[layer_idx, expert_id] += gate_prob
        return hook
    layer_idx = 0
    for (name, module) in exe.model_runner.model.named_modules():
        if pattern.search(name):
            module.register_forward_hook(make_hook(layer_idx))
            layer_idx += 1
    print(f'[Info] Installed {layer_idx} MoE gate hooks.')
    return layer_idx

def reset_buffers(exe):
    exe._trace_in.zero_()
    exe._sum_in.zero_()

def collect_buffers(exe):
    return {'ti': exe._trace_in.cpu().tolist(), 'si': exe._sum_in.cpu().tolist()}

def compute_avg(counts, sums):
    count_arr = np.array(counts, dtype=np.float32)
    sum_arr = np.array(sums, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg = np.where(count_arr > 0, sum_arr / count_arr, 0.0)
    return avg.tolist()

def main():
    parser = argparse.ArgumentParser(description='Trace MoE expert routing for multiple prompts.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', default='moe_traces.json')
    parser.add_argument('--max_new', type=int, default=64)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples-per-domain', type=int, default=SAMPLE_LIMIT)
    parser.add_argument('--max-model-len', type=int, default=65536)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.95)
    args = parser.parse_args()
    if args.samples_per_domain<1:raise ValueError('samples-per-domain must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    with open(pathlib.Path(args.model) / 'config.json', encoding='utf-8') as f:
        cfg = json.load(f)
    model_type = cfg.get('model_type', '')
    if model_type != MODEL_TYPE:raise ValueError(f'Expected {MODEL_TYPE}, got {model_type}')
    arch = 'mixtral' if 'mixtral' in model_type or 'mistral' in model_type else 'qwen'
    num_layers = int(cfg['num_hidden_layers'])
    num_experts = get_num_experts(cfg)
    topk = int(next((cfg[key] for key in ('num_experts_per_tok', 'router_top_k') if key in cfg), 1))
    llm_kwargs = {'model': args.model, 'enforce_eager': True, 'trust_remote_code': True, 'dtype': args.dtype, 'seed': args.seed, 'gpu_memory_utilization': args.gpu_memory_utilization, 'max_model_len': args.max_model_len}
    print(f'[Info] Model: {args.model}')
    print(f"[Info] GPU memory utilization: {llm_kwargs['gpu_memory_utilization']}")
    llm = LLM(**llm_kwargs)
    params = SamplingParams(max_tokens=args.max_new, temperature=0.0)
    found=llm.collective_rpc(lambda exe: install_hooks(exe, num_layers=num_layers, num_experts=num_experts, topk=topk, arch=arch))
    assert found and all(x==EXPECTED_MOE_LAYERS for x in found), f'Wrong hook counts: {found}'
    with open(args.data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    results = []
    safe_cnt = 0
    help_cnt = 0
    tokenizer = llm.get_tokenizer()
    for (i, item) in enumerate(dataset):
        source = item.get('source')
        if source == 'PKU-SafeRLHF':
            if safe_cnt >= args.samples_per_domain:
                continue
            safe_cnt += 1
        else:
            if help_cnt >= args.samples_per_domain:
                continue
            help_cnt += 1
        prompt = item.get('prompt')
        if prompt is None:
            ins,extra=item.get('instruction',''),item.get('input','')
            prompt=f'{ins}\n{extra}' if extra else ins
        print(f'[{i + 1}/{len(dataset)}] Processing; safe={safe_cnt}, help={help_cnt}')
        prompt_token_num = len(tokenizer(prompt)['input_ids'])
        llm.collective_rpc(reset_buffers)
        outputs = list(llm.generate([prompt], params))
        _gen_text = outputs[0].outputs[0].text
        stats = llm.collective_rpc(collect_buffers, timeout=None)[0]
        avg_in = compute_avg(stats['ti'], stats['si'])
        results.append({'prompt': prompt, 'source': source, 'token_num': prompt_token_num, 'input_expert_usage': stats['ti'], 'input_avg_probs': avg_in})
    pathlib.Path(args.output_path).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    metadata=dict(model_type=MODEL_TYPE, samples_per_domain=args.samples_per_domain, max_new=args.max_new, trace_scope='prefill_and_decode', gate_probability='softmax in native logits dtype; mean over selected occurrences', logical_moe_layers=EXPECTED_MOE_LAYERS, logical_to_model_layer_offset=LAYER_OFFSET, records=len(results))
    pathlib.Path(args.output_path+'.metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'All traces saved to {pathlib.Path(args.output_path).resolve()}')
if __name__ == '__main__':
    main()