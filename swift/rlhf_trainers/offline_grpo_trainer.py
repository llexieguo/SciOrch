# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn as nn
from contextlib import nullcontext
from functools import partial
from peft import PeftModel
from transformers import PreTrainedModel
from transformers import Trainer as HfTrainer
from trl.trainer.utils import selective_log_softmax
from typing import Dict, List, Optional, Union

from swift.trainers import DataLoaderMixin, SwiftMixin
from swift.utils import get_logger
from .rlhf_mixin import RLHFTrainerMixin

logger = get_logger()


class OfflineGrpoTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HfTrainer):
    """Offline Dr. GRPO trainer for pre-collected (prompt, response, reward) data.

    Implements the Dr. GRPO advantage estimator (length-normalised reward) over
    variable-size prompt groups:

        r̃_i = r_i / |y_i|              (length-normalised reward)
        A_i  = r̃_i - mean_j(r̃_j)     (group-mean-centred advantage)

    where |y_i| is the number of response characters (used as a proxy for token
    count at dataset-preprocessing time; see OfflineGrpoArguments for an option
    to supply a pre-tokenised length column).

    The policy-gradient update is the same as Offline REINFORCE++:
        L_pg = -A_i * log π_θ(y_i | x_i)
    with an optional KL penalty against the reference model for stability.

    Group sizes are variable — there is no requirement that every prompt has the
    same number of responses.
    """

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 *_args,
                 reward_key: str = 'reward',
                 sample_weight_key: Optional[str] = None,
                 **kwargs):
        self.is_peft_model = isinstance(model, PeftModel)
        self.ref_adapter_name = getattr(kwargs.get('args'), 'ref_adapter_name', None)
        self.model_adapter_name = None
        self.reward_key = reward_key
        self.sample_weight_key = sample_weight_key
        super().__init__(model, ref_model, *_args, **kwargs)

    def create_loss_and_eval_metric(self, args):
        return {}

    def _get_data_collator(self, args, template):
        padding_to = template.max_length if args.tuner_type == 'longlora' else None
        base_collator = partial(template.data_collator, padding_to=padding_to)

        reward_key = self.reward_key
        sample_weight_key = self.sample_weight_key

        def offline_grpo_collator(batch, **kwargs):
            rewards = []
            advantages = []
            sample_weights = []
            for b in batch:
                extra = b.get('_extra_kwargs', {})
                rewards.append(float(extra.get(reward_key, 0.0)))
                advantages.append(float(extra.get('_advantage', 0.0)))
                sample_weights.append(float(extra.get(sample_weight_key, 1.0)) if sample_weight_key else 1.0)
            result = base_collator(batch, **kwargs)
            result['rewards'] = torch.tensor(rewards, dtype=torch.float32)
            result['advantages'] = torch.tensor(advantages, dtype=torch.float32)
            result['sample_weights'] = torch.tensor(sample_weights, dtype=torch.float32)
            return result

        return offline_grpo_collator

    def null_ref_context(self):
        if self.is_peft_model and not self.ref_adapter_name:
            return self.accelerator.unwrap_model(self.model).disable_adapter()
        return nullcontext()

    def _compute_logps(self, model, inputs, labels, loss_mask):
        """Compute per-token log probabilities."""
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits
        if logits.shape[1] != labels.shape[1]:
            logits = logits[:, -labels.shape[1]:]
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        shift_mask = loss_mask[:, 1:]
        safe_labels = shift_labels.clone()
        safe_labels[~shift_mask] = 0
        per_token_logps = selective_log_softmax(shift_logits, safe_labels)
        per_token_logps = per_token_logps * shift_mask
        return per_token_logps, shift_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards = inputs.pop('rewards').to(self.args.device)
        advantages = inputs.pop('advantages').to(self.args.device)
        sample_weights = inputs.pop('sample_weights').to(self.args.device)
        labels = inputs.pop('labels')
        loss_mask = labels != -100

        per_token_logps, shift_mask = self._compute_logps(model, inputs, labels, loss_mask)

        with torch.no_grad():
            if self.ref_model is not None:
                ref_per_token_logps, _ = self._compute_logps(self.ref_model, inputs, labels, loss_mask)
            else:
                with self.null_ref_context():
                    ref_per_token_logps, _ = self._compute_logps(model, inputs, labels, loss_mask)

        num_tokens = shift_mask.sum(-1).clamp(min=1)

        if self.args.whiten_advantages and advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = advantages / (adv_std + 1e-8)

        advantages_expanded = advantages.unsqueeze(-1).expand_as(per_token_logps)

        # Dr. GRPO policy gradient: -A_i^Dr * log π (no IS ratio, no clipping)
        # Advantages are already length-normalised at dataset-prep time.
        pg_loss = -advantages_expanded * per_token_logps
        pg_loss = (pg_loss * shift_mask).sum(-1) / num_tokens

        # KL penalty (same options as offline_reinforce for consistency)
        kl_estimator = getattr(self.args, 'kl_estimator', 'k3')
        seq_mean_log_ratio = None
        if kl_estimator == 'k3':
            log_ratio = ref_per_token_logps - per_token_logps
            per_token_kl = torch.exp(log_ratio) - log_ratio - 1
            kl = (per_token_kl * shift_mask).sum(-1) / num_tokens
        elif kl_estimator == 'gspo':
            log_ratio = (ref_per_token_logps - per_token_logps) * shift_mask
            seq_mean_log_ratio = log_ratio.sum(-1) / num_tokens
            seq_mean_log_ratio = torch.clamp(seq_mean_log_ratio, min=-20, max=20)
            seq_kl = torch.exp(seq_mean_log_ratio) - seq_mean_log_ratio - 1
            kl = torch.clamp(seq_kl * num_tokens.to(seq_kl.dtype), min=0.0, max=10.0)
        elif kl_estimator == 'k1':
            per_token_kl = per_token_logps - ref_per_token_logps
            kl = (per_token_kl * shift_mask).sum(-1) / num_tokens
        else:
            raise ValueError(
                f"Unknown kl_estimator: {kl_estimator}. Possible values are 'k1', 'k3', and 'gspo'.")

        per_sample_loss = pg_loss + self.args.kl_coef * kl
        weight_sum = sample_weights.sum().clamp_min(1e-8)
        loss = (per_sample_loss * sample_weights).sum() / weight_sum
        pg_loss_scalar = (pg_loss * sample_weights).sum() / weight_sum
        kl_loss = (kl * sample_weights).sum() / weight_sum

        metrics = {
            'offline_grpo/pg_loss': pg_loss_scalar.detach().item(),
            'offline_grpo/kl': kl_loss.detach().item(),
            'offline_grpo/kl_abs': kl_loss.detach().abs().item(),
            'offline_grpo/seq_mean_log_ratio': (
                seq_mean_log_ratio.detach().mean().item() if seq_mean_log_ratio is not None else 0.0),
            'offline_grpo/seq_mean_log_ratio_abs': (
                seq_mean_log_ratio.detach().abs().mean().item() if seq_mean_log_ratio is not None else 0.0),
            'offline_grpo/reward_mean': rewards.mean().item(),
            'offline_grpo/advantage_mean': advantages.mean().item(),
            'offline_grpo/advantage_std': advantages.std().item() if advantages.numel() > 1 else 0.0,
            'offline_grpo/advantage_abs_mean': advantages.detach().abs().mean().item(),
            'offline_grpo/sample_weight_mean': sample_weights.mean().item(),
            'offline_grpo/num_tokens_mean': num_tokens.float().mean().item(),
        }
        self.store_metrics(metrics, train_eval='train')

        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps

        if return_outputs:
            return loss, metrics
        return loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only=False, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, metrics = self.compute_loss(model, inputs, return_outputs=True)
            self.store_metrics(metrics, train_eval='eval')
            if prediction_loss_only:
                return loss.detach(), None, None
            logits = torch.tensor(
                [metrics.get('offline_grpo/reward_mean', 0)], device=self.accelerator.device)
            labels = torch.zeros(logits.shape[0], device=self.accelerator.device)
            return loss.detach(), logits, labels

    def store_metrics(self, metrics, train_eval='train'):
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs, start_time=None):
        train_eval = 'train' if 'loss' in logs else 'eval'
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        mode = 'train' if self.model.training else 'eval'
        custom_metrics = self.custom_metrics[mode]
        prefix = 'eval_' if mode == 'eval' else ''
        logs.update(self.compute_custom_metrics(custom_metrics, prefix))
        return HfTrainer.log(self, logs, start_time)
