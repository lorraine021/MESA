# MESA

🌟 Our paper " MESA: Improving MoE Safety Alignment via Decentralized Expertise" has been accepted by ICML2026!

Training and evaluation code for DeepSeek-V2-Lite-Chat and Qwen3-30B-A3B-Instruct. Run the following commands from the repository root. Steps 2–3 are optional if you use the supplied expert JSON files. Our code is built on LLaMA-Factory (Apache-2.0).

## 1. Prepare environments and model paths

```bash
conda env create -f environment/training_environment.yaml
conda env create -f environment/evaluation_environment.yaml
conda activate mesa-training

export DEEPSEEK_MODEL=/path/to/DeepSeek-V2-Lite-Chat
export QWEN_MODEL=/path/to/Qwen3-30B-A3B-Instruct
```

Use complete local Hugging Face model directories. Training data are in
`data/stair_all.json`.

## 2. Collect expert activation statistics [Option]

```bash
conda activate mesa-evaluation

# DeepSeek
python scripts/get_deepseek_logits.py \
  --model "$DEEPSEEK_MODEL" --data_path data/stair_all.json \
  --output_path data/moe_traces_ds.json

# Qwen
python scripts/get_qwen_logits.py \
  --model "$QWEN_MODEL" --data_path data/stair_all.json \
  --output_path data/moe_traces_qwen.json
```

Each model uses 500 safety and 500 general prompts by default. Statistics include both prompt processing and generation, following the historical collection setup.

## 3. Select experts [Option]

These commands regenerate the expert JSON files used by the training configs.

```bash
conda activate mesa-training

python scripts/select_deepseek_experts.py \
  --model "$DEEPSEEK_MODEL" --trace data/moe_traces_ds.json \
  --percentage 15 --output experts/deepseek_experts_15pct.json

python scripts/select_qwen_experts.py \
  --model "$QWEN_MODEL" --trace data/moe_traces_qwen.json \
  --percentage 10 --output experts/qwen_experts_10pct.json
```

## 4. Train

We use 8 NVIDIA H20 GPUs (141GB VRAM each) with DeepSpeed ZeRO Stage 3. 

```bash
conda activate mesa-training

# Example: replace --config to select another Qwen setting
bash train.sh --config configs/deepseek_mesa.yaml --model "$DEEPSEEK_MODEL" --dry-run
bash train.sh --config configs/deepseek_mesa.yaml --model "$DEEPSEEK_MODEL"

bash train.sh --config configs/qwen_mesa.yaml --model "$QWEN_MODEL" --dry-run
bash train.sh --config configs/qwen_mesa.yaml --model "$QWEN_MODEL"
```

Outputs are saved under `outputs/`; use `--output outputs/new_run` for another run. Output directories must be empty.

## 5. Evaluate

```bash
conda activate mesa-evaluation
bash evaluate.sh --preflight-only
bash evaluate.sh --model outputs/deepseek_mesa/checkpoint-500 \
  --judge /path/to/Octopus-SEval --run-name deepseek_mesa
```

Use a Qwen checkpoint with the same command; the architecture is detected automatically. 

The nine dataset files are in `eval_data/`. Scores and model responses are saved to `evaluation/results/<run-name>/`. See [evaluation/README.md](evaluation/README.md) for device settings and outputs.
