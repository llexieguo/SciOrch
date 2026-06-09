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


class OfflinePPOTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HfTrainer):
    """Offline PPO trainer for pre-collected (prompt, response, reward) data.

    Uses PPO clipped surrogate objective with pre-computed rewards.
    The reference model provides the baseline policy for computing
    the importance sampling ratio and KL penalty.
    """

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 *_args,
                 reward_key: str = 'expected_acc_reward',
                 **kwargs):
        self.is_peft_model = isinstance(model, PeftModel)
        self.ref_adapter_name = getattr(kwargs.get('args'), 'ref_adapter_name', None)
        self.model_adapter_name = None
        self.reward_key = reward_key
        super().__init__(model, ref_model, *_args, **kwargs)

    def create_loss_and_eval_metric(self, args):
        return {}

    def _get_data_collator(self, args, template):
        padding_to = template.max_length if args.tuner_type == 'longlora' else None
        base_collator = partial(template.data_collator, padding_to=padding_to)

        reward_key = self.reward_key

        def offline_ppo_collator(batch, **kwargs):
            rewards = []
            for b in batch:
                extra = b.get('_extra_kwargs', {})
                rewards.append(float(extra.get(reward_key, 0.0)))
            result = base_collator(batch, **kwargs)
            result['rewards'] = torch.tensor(rewards, dtype=torch.float32)
            return result

        return offline_ppo_collator

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

        advantages = rewards
        if self.args.whiten_rewards and rewards.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_ratio = per_token_logps - ref_per_token_logps
        ratio = torch.exp(log_ratio)

        advantages_expanded = advantages.unsqueeze(-1).expand_as(per_token_logps)

        pg_loss1 = -advantages_expanded * ratio
        pg_loss2 = -advantages_expanded * torch.clamp(
            ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = (pg_loss * shift_mask).sum(-1) / num_tokens
        pg_loss = pg_loss.mean()

        per_token_kl = per_token_logps - ref_per_token_logps
        kl = (per_token_kl * shift_mask).sum(-1) / num_tokens
        kl_loss = kl.mean()

        loss = pg_loss + self.args.kl_coef * kl_loss

        metrics = {
            'offline_ppo/pg_loss': pg_loss.detach().item(),
            'offline_ppo/kl': kl_loss.detach().item(),
            'offline_ppo/reward_mean': rewards.mean().item(),
            'offline_ppo/mean_ratio': ratio[shift_mask].mean().detach().item(),
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
            logits = torch.tensor([metrics.get('offline_ppo/reward_mean', 0)], device=self.accelerator.device)
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
