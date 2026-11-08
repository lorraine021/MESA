# Copyright 2025 the LlamaFactory team. Licensed under Apache-2.0.
import json
import math
import os
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from .trainer import CustomSeq2SeqTrainer

class MESADeepSeekTrainer(CustomSeq2SeqTrainer):
    """Selected-marginal OT with forward target-to-model KL."""
    _GATE_PATTERN = re.compile('(?:^|\\.)layers\\.(\\d+)\\.mlp\\.gate$')

    def _build_cost_matrix(self):
        shape = (len(self._find_moe_gates(self.model)), int(self.model.config.n_routed_experts))
        selected = torch.zeros(shape, dtype=torch.bool)
        for layer, expert in self.expert_f_map:
            if not (0 <= layer < shape[0] and 0 <= expert < shape[1]):
                raise ValueError(f'Selected expert {(layer, expert)} outside {shape}')
            selected[layer, expert] = True
        entries = json.loads(Path(self.finetuning_args.freeze_expert_selection_file).read_text())['selected_experts']
        file_set = {(int(e['layer_id']), int(e['expert_id'])) for e in entries}
        if len(file_set) != len(entries) or file_set != set(self.expert_f_map) or (not selected.any()):
            raise ValueError('Invalid selected set or mismatch with freeze expert selection file')
        multiplier = 2.0  # Fixed selected-expert odds multiplier.
        tolerance = float(self.finetuning_args.mesa_selected_ot_tolerance)
        max_iter = int(self.finetuning_args.mesa_selected_ot_max_iter)
        if not math.isfinite(multiplier) or multiplier < 1:
            raise ValueError('MESA_SELECTED_ODDS_MULTIPLIER must be finite and >= 1')
        if not math.isfinite(tolerance) or not 0 < tolerance < 0.01 or max_iter < 1:
            raise ValueError('Invalid selected-marginal Sinkhorn tolerance or iteration limit')
        self.selected_expert_mask = selected
        self.selected_odds_multiplier = multiplier
        self.selected_ot_tolerance = tolerance
        self.selected_ot_max_iter = max_iter
        self._selected_ot_reported_layers = set()
        self._last_selected_ot_stats = {}
        print(f'[MESA_SelectedMarginalOT] selected={int(selected.sum())}, shape={shape}, odds_multiplier={multiplier:g}, tol={tolerance:g}, max_iter={max_iter}; safety=KL projection of Base onto selected-biased batch marginals; general=original Base KL; no uniform columns or rank cost', flush=True)
        return None

    @staticmethod
    @torch.no_grad()
    def _kl_project_to_columns(prior, column, tolerance, max_iter):
        tokens = prior.shape[0]
        log_kernel = prior.log()
        log_column = column.log() + math.log(tokens)
        log_u = torch.zeros(tokens, device=prior.device, dtype=prior.dtype)
        log_v = torch.zeros_like(column)
        row_error = column_error = float('inf')
        for iteration in range(1, max_iter + 1):
            log_u = -torch.logsumexp(log_kernel + log_v[None, :], dim=1)
            log_v = log_column - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
            if iteration % 10 and iteration != max_iter:
                continue
            q = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
            row_error = float((q.sum(-1) - 1).abs().max())
            normalized = q / q.sum(-1, keepdim=True)
            column_error = float((normalized.mean(0) - column).abs().max())
            if row_error <= tolerance and column_error <= tolerance:
                return (normalized.detach(), dict(iterations=iteration, row_error=row_error, column_error=column_error))
        raise RuntimeError(f'Selected-marginal OT did not converge after {max_iter} iterations: row_error={row_error:.3g}, column_error={column_error:.3g}, tol={tolerance:.3g}. Increase MESA_SELECTED_OT_MAX_ITER or inspect Base probabilities.')

    @torch.no_grad()
    def _batch_sinkhorn_solver(self, probs_base, layer_idx):
        p = probs_base.detach().float()
        if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] != self.selected_expert_mask.shape[1]:
            raise ValueError('Unexpected Base router probability shape')
        if not 0 <= layer_idx < self.selected_expert_mask.shape[0]:
            raise IndexError(f'Invalid logical MoE layer {layer_idx}')
        if not torch.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
            raise ValueError('Base probabilities must be finite, nonnegative, and nonempty')
        p = p.clamp_min(1e-12)
        p = p / p.sum(-1, keepdim=True)
        selected = self.selected_expert_mask[layer_idx].to(p.device)
        base_column = p.mean(0)
        log_c = base_column.log() + selected.float() * math.log(self.selected_odds_multiplier)
        column = log_c.softmax(0)
        if self.selected_odds_multiplier == 1 or not bool(selected.any()) or bool(selected.all()):
            q = p
            stats = dict(iterations=0, row_error=float((q.sum(-1) - 1).abs().max()), column_error=float((q.mean(0) - column).abs().max()))
        else:
            q, stats = self._kl_project_to_columns(p, column, self.selected_ot_tolerance, self.selected_ot_max_iter)
        stats.update(base_selected_mass=float(base_column[selected].sum()), target_selected_mass=float(column[selected].sum()), achieved_selected_mass=float(q[:, selected].sum(-1).mean()), tokens=p.shape[0])
        self._last_selected_ot_stats[layer_idx] = stats
        if layer_idx not in self._selected_ot_reported_layers:
            print(f'[MESA_SelectedMarginalOT] layer={layer_idx} {json.dumps(stats)}', flush=True)
            self._selected_ot_reported_layers.add(layer_idx)
        return q.detach()

    def _compute_router_loss(self, model, baseline_model, inputs, source_mask):
        router_logits_model = self._router_capture
        router_logits_base = self._extract_router_logits(baseline_model, inputs)
        if not router_logits_model or not router_logits_base:
            print(f'[WARNING] No router logits extracted! Model: {len(router_logits_model)}, Baseline: {len(router_logits_base)}')
            return torch.tensor(0.0, device=self.args.device)
        total_router_loss = 0.0
        num_layers = len(router_logits_model)
        for layer_idx in sorted(router_logits_model.keys()):
            logits_model = router_logits_model[layer_idx]
            logits_base = router_logits_base[layer_idx]
            if logits_model.dim() == 2:
                batch_size = source_mask.shape[0]
                seq_len = logits_model.shape[0] // batch_size
                expanded_source_mask = source_mask.unsqueeze(1).expand(-1, seq_len).reshape(-1)
                probs_model = F.softmax(logits_model, dim=-1)
                probs_base = F.softmax(logits_base, dim=-1)
            else:
                continue
            safe_mask = expanded_source_mask.bool()
            general_mask = ~safe_mask
            layer_loss = 0.0
            if safe_mask.any():
                logits_model_safe = logits_model[safe_mask].float()
                probs_base_safe = probs_base[safe_mask].float()
                probs_model_safe = probs_model[safe_mask].float()
                pi_star = self._batch_sinkhorn_solver(probs_base_safe, layer_idx)
                log_probs_model = F.log_softmax(logits_model_safe, dim=-1)
                probs_model_safe_calc = torch.exp(log_probs_model)
                safe_loss = F.kl_div(log_probs_model, pi_star.detach(), reduction='batchmean')
                layer_loss += safe_loss
            if general_mask.any():
                logits_model_general = logits_model[general_mask].float()
                logits_base_general = logits_base[general_mask].float()
                log_probs_model = F.log_softmax(logits_model_general / self.kl_temperature, dim=-1)
                probs_base_general = F.softmax(logits_base_general / self.kl_temperature, dim=-1)
                probs_model_general = torch.exp(log_probs_model)
                general_loss = F.kl_div(log_probs_model, probs_base_general.detach(), reduction='batchmean')
                layer_loss += general_loss
            total_router_loss += layer_loss
        avg_router_loss = total_router_loss / num_layers if num_layers > 0 else total_router_loss
        return avg_router_loss


    def _find_moe_gates(self, model):
        gates = []
        for name, module in model.named_modules():
            match = self._GATE_PATTERN.search(name)
            weight = getattr(module, 'weight', None)
            if match is not None and weight is not None:
                gates.append((int(match.group(1)), name, module))
        gates.sort(key=lambda x: x[0])
        if not gates:
            raise RuntimeError("No DeepSeek MoE gates found (expected '*.layers.N.mlp.gate').")
        ids = [x[0] for x in gates]
        offset = int(self.finetuning_args.freeze_expert_layer_offset)
        if ids[0] != offset:
            raise RuntimeError(f'First MoE model layer is {ids[0]}, but layer offset is {offset}.')
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise RuntimeError(f'DeepSeek MoE layers are not contiguous: {ids}')
        expected_experts = int(getattr(self.model.config, 'n_routed_experts'))
        if expected_experts <= 0:
            raise RuntimeError(f'Invalid n_routed_experts={expected_experts}.')
        return gates

    def _register_router_hooks(self, model, capture_dict, detach: bool):
        handles = []
        for logical_id, (model_id, name, module) in enumerate(self._find_moe_gates(model)):

            def hook(gate, inputs, output, logical_id=logical_id, model_id=model_id, name=name):
                if not inputs:
                    raise RuntimeError(f'Gate {name} produced no hook input.')
                hidden = inputs[0]
                logits = F.linear(hidden.reshape(-1, hidden.shape[-1]).float(), gate.weight.float())
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f'Non-finite router logits at logical layer {logical_id} (model layer {model_id}, {name}).')
                capture_dict[logical_id] = logits.detach() if detach else logits
            handles.append(module.register_forward_hook(hook))
        return handles

    def __init__(self, baseline_model=None, expert_f_map=None, **kwargs):
        super().__init__(**kwargs)
        self.baseline_model = baseline_model
        self.expert_f_map = expert_f_map
        self.router_loss_weight = self.finetuning_args.mesa_router_loss_weight
        self.kl_temperature = 2.0  # Fixed general-stream KL temperature.
        self.safe_data_source = self.finetuning_args.mesa_safe_data_source
        if expert_f_map is not None:
            self.cost_matrix = self._build_cost_matrix()
        else:
            self.cost_matrix = None
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        self._router_capture = {}
        self._train_hooks = []
        self._moe_input_grad_hooks = []
        print(f'[MESA_Trainer] Initialized with:')
        print(f'  - Router loss weight: {self.router_loss_weight}')

    def _ensure_train_hooks(self, model):
        if not self._moe_input_grad_hooks:

            def require_moe_input_grad(module, args):
                if args and torch.is_tensor(args[0]) and torch.is_floating_point(args[0]):
                    if not args[0].requires_grad:
                        args[0].requires_grad_(True)
                return args
            for module in model.modules():
                if module.__class__.__name__ == 'DeepseekV2MoE':
                    self._moe_input_grad_hooks.append(module.register_forward_pre_hook(require_moe_input_grad, prepend=True))
            print(f'[MESA_Trainer] Registered {len(self._moe_input_grad_hooks)} DeepSeek MoE input-gradient hooks for ZeRO-3')
        if not self._train_hooks:
            self._train_hooks = self._register_router_hooks(model, self._router_capture, detach=False)
            print(f'[MESA_Trainer] Registered {len(self._train_hooks)} router capture hooks on training model (with grad)')

    def _extract_router_logits(self, model, inputs):
        router_logits = {}
        handles = self._register_router_hooks(model, router_logits, detach=True)
        model_forward_inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'position_ids']}
        with torch.no_grad():
            params_to_gather = []
            for param in model.parameters():
                if hasattr(param, 'ds_id'):
                    params_to_gather.append(param)
            if params_to_gather:
                try:
                    from deepspeed import zero
                    with zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        _ = model(**model_forward_inputs, use_cache=False)
                except ImportError:
                    _ = model(**model_forward_inputs, use_cache=False)
            else:
                _ = model(**model_forward_inputs, use_cache=False)
        for h in handles:
            h.remove()
        return router_logits



    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        source = inputs.get('source', None)
        model_inputs = {k: v for k, v in inputs.items() if k != 'source'}
        self._router_capture.clear()
        self._ensure_train_hooks(model)
        outputs = model(**model_inputs)
        if self.label_smoother is not None and 'labels' in model_inputs:
            sft_loss = self.label_smoother(outputs, model_inputs['labels'])
        else:
            sft_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        router_loss = torch.tensor(0.0, device=sft_loss.device)
        if self.baseline_model is not None and source is not None:
            if self.baseline_model.device != model_inputs['input_ids'].device:
                self.baseline_model = self.baseline_model.to(model_inputs['input_ids'].device)
            if isinstance(source, list):
                source_mask = torch.tensor([1 if s == self.safe_data_source else 0 for s in source], device=sft_loss.device)
            else:
                source_mask = source
            router_loss = self._compute_router_loss(model, self.baseline_model, model_inputs, source_mask)
        elif self.state.global_step % 10 == 0:
            if self.baseline_model is None:
                print(f'  [WARNING] baseline_model is None, skipping router loss')
            if source is None:
                print(f'  [WARNING] source is None, skipping router loss')
        if not isinstance(router_loss, torch.Tensor):
            router_loss = torch.tensor(router_loss, device=sft_loss.device)
        sft_loss = torch.nan_to_num(sft_loss, nan=0.0)
        router_loss = torch.nan_to_num(router_loss, nan=0.0)
        total_loss = sft_loss + self.router_loss_weight * router_loss
        self.log({'sft_loss': sft_loss.item(), 'router_loss': router_loss.item(), 'total_loss': total_loss.item()})
        return (total_loss, outputs) if return_outputs else total_loss
