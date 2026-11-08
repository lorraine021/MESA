# Evaluation

Run from the repository root using `mesa-evaluation`.

```bash
conda activate mesa-evaluation
bash evaluate.sh --preflight-only

# DeepSeek; use a Qwen checkpoint with the same command to evaluate Qwen.
bash evaluate.sh \
  --model outputs/deepseek_mesa/checkpoint-500 \
  --judge /path/to/Octopus-SEval \
  --run-name deepseek_mesa
```

The architecture is inferred from `config.json`. By default, GPUs `0,1,2,3`
serve the evaluated model and GPUs `4,5,6,7` serve Octopus. Override these
with `--target-gpus` and `--judge-gpus`; tensor parallelism follows each list.
Use `--dry-run` to inspect settings without starting services.

Data default to `eval_data/`; use `--data-root` for another directory with the
same filenames. Each suite evaluates six general benchmarks (Math500, GSM8K,
HumanEval, MBPP, GPQA, MMLU) and three safety benchmarks (StrongREJECT,
Strata SWORD, WildJailbreak).

Results are written to `evaluation/results/<run-name>/`:

```text
evaluation_config.json    paths, environment versions and device settings
general/                 six general benchmark results and model outputs
safety_generations/      three safety benchmark response files
safety_octopus.json      safety judgements and counts
summary.json             combined benchmark scores
target_vllm.log           evaluated-model server log
octopus_vllm.log          judge-model server log
```