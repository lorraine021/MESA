"""Select a supported architecture's independent, validated MESA trainer."""


def get_mesa_trainer_class(model):
    model_type = getattr(model.config, "model_type", None)
    if model_type == "deepseek_v2":
        from .mesa_deepseek_trainer import MESADeepSeekTrainer

        return MESADeepSeekTrainer
    if model_type == "qwen3_moe":
        from .mesa_qwen_trainer import MESAQwenTrainer

        return MESAQwenTrainer
    raise ValueError(
        f"MESA does not support model_type={model_type!r}; "
        "use DeepSeek-V2-Lite-Chat (deepseek_v2) or Qwen3-30B-A3B-Instruct (qwen3_moe)."
    )
