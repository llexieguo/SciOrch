# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import peft
from contextlib import nullcontext
from packaging import version
from typing import List, Optional, Union

from swift.arguments import BaseArguments, RLHFArguments
from swift.dataset import DatasetLoader, load_dataset
from swift.model import get_model_info_meta
from swift.sequence_parallel import sequence_parallel
from swift.tuner_plugin import Tuner, tuners_map
from swift.tuners import Swift
from swift.utils import (HfConfigFactory, disable_deepspeed_zero3, get_logger, get_model_parameter_info,
                         safe_snapshot_download)
from ..utils import prepare_adapter
from .kto import prepare_kto_dataset
from .sft import SwiftSft

logger = get_logger()


class SwiftRLHF(SwiftSft):
    args_class = RLHFArguments
    args: args_class

    @staticmethod
    def _get_model_task_type(model_dir):
        task_type = None
        num_labels = None
        if os.path.exists(os.path.join(model_dir, 'args.json')):
            model_args = BaseArguments.from_pretrained(model_dir)
            if hasattr(model_args, 'task_type'):
                task_type = model_args.task_type
            if hasattr(model_args, 'num_labels'):
                num_labels = model_args.num_labels
            if task_type == 'seq_cls' and num_labels is None:
                num_labels = 1
        else:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            if hasattr(model_config, 'architectures') and model_config.architectures:
                if any('sequenceclassification' in arch.lower() for arch in model_config.architectures):
                    task_type = 'seq_cls'
                    num_labels = getattr(model_config, 'num_labels', None) or 1

            if task_type is None:
                if hasattr(model_config, 'num_labels'):
                    num_labels = model_config.num_labels
                    # PretrainedConfig default num_labels = 2
                    if num_labels == 1:
                        task_type = 'seq_cls'
        return task_type, num_labels

    def _prepare_single_model(self, key, origin_key, model_type, model_revision):
        args = self.args
        origin_key = origin_key or key
        model_id_or_path = getattr(args, f'{key}_model')
        if model_id_or_path is None:
            return

        if args.rlhf_type == 'ppo' and key == 'reward' and isinstance(model_id_or_path, (list, tuple)):
            assert len(model_id_or_path) == 1, f'model_id_or_path: {model_id_or_path}'
            model_id_or_path = model_id_or_path[0]

        if model_type is None:
            model_info, _ = get_model_info_meta(model_id_or_path)
            model_type = model_info.model_type

        if isinstance(model_id_or_path, list):
            # value model in PPO
            model_id_or_path = model_id_or_path[0]

        model_dir = safe_snapshot_download(
            model_id_or_path=model_id_or_path,
            revision=model_revision,
            download_model=False,
            use_hf=args.use_hf,
            hub_token=args.hub_token,
        )
        task_type, num_labels = self._get_model_task_type(model_dir)
        context = nullcontext()
        if key == 'teacher' and args.teacher_deepspeed:
            if args.teacher_deepspeed.get('zero_optimization', {}).get('stage') != 3:
                context = disable_deepspeed_zero3()
        with context:
            model, processor = args.get_model_processor(
                model=model_id_or_path,
                model_type=model_type,
                revision=model_revision,
                task_type=task_type,
                num_labels=num_labels)

        adapters = args.adapters if key == 'ref' else args.reward_adapters
        model = prepare_adapter(args, model, adapters)
        if origin_key in {'ref', 'reward', 'teacher'}:
            if self.args.sequence_parallel_size > 1:
                sequence_parallel.prepare(
                    self.args.sequence_parallel_size, model, processor, padding_free=args.padding_free)
            model.requires_grad_(False).eval()
        else:
            model = self.prepare_model(args, model, task_type=task_type)
            logger.info(f'value_model: {model}')
            model_parameter_info = get_model_parameter_info(model)
            self.train_msg['value_model_parameter_info'] = model_parameter_info
            logger.info(f'value_model_parameter_info: {model_parameter_info}')

        HfConfigFactory.set_config_attr(model.config, 'use_cache', False)
        return model, processor

    def _prepare_model_tokenizer(self):
        # prepare ref/reward/value model
        args = self.args
        # Handle ref and value models
        for key in ['ref', 'value', 'teacher']:
            setattr(self, f'{key}_model', None)
            if key == 'ref' and args.rlhf_type == 'gkd':
                continue
            if key == 'value' and args.rlhf_type not in {'ppo'}:
                continue
            if key == 'teacher' and args.rlhf_type != 'gkd':
                continue
            if key in {'value', 'teacher'} and args.rlhf_type in {'offline_ppo', 'offline_reinforce', 'offline_grpo'}:
                continue
            model_key = 'reward' if key == 'value' else key
            model_type = getattr(args, f'{model_key}_model_type')
            model_revision = getattr(args, f'{model_key}_model_revision')
            if key == 'value':
                model_type = model_type[0] if model_type else None
                model_revision = model_revision[0] if model_revision else None

            result = self._prepare_single_model(model_key, key, model_type, model_revision)
            if result is not None:
                model, _ = result
                setattr(self, f'{key}_model', model)

        # Handle reward model(s)
        self.reward_model = None
        if hasattr(args, 'reward_model') and args.reward_model is not None:
            rms = args.reward_model if isinstance(args.reward_model, list) else [args.reward_model]
            num_rms = len(rms)
            rm_types = args.reward_model_type if args.reward_model_type else [None] * num_rms
            rm_templates = args.reward_template if args.reward_template else [None] * num_rms
            rm_revisions = args.reward_model_revision if args.reward_model_revision else [None] * num_rms
            assert len(rms) == len(rm_types) == len(rm_templates) == len(rm_revisions)

            self.reward_model = []
            if args.rlhf_type == 'grpo':
                self.reward_template = []

            for reward_model_path, rm_type, rm_template, rm_revision in zip(rms, rm_types, rm_templates, rm_revisions):
                args.reward_model = reward_model_path  # Temporarily set for prepare_single_model
                result = self._prepare_single_model('reward', None, rm_type, rm_revision)
                if result is not None:
                    model, processor = result
                    self.reward_model.append(model)

                    if args.rlhf_type == 'grpo':
                        template_type = rm_template or processor.model_meta.template
                        reward_template = self.args.get_template(processor, template_type=template_type)
                        if reward_template.use_model:
                            reward_template.model = model
                        self.reward_template.append(reward_template)
            args.reward_model = rms  # Restore original value
            if args.rlhf_type != 'grpo' and self.reward_model:
                assert len(self.reward_model) <= 1
                self.reward_model = self.reward_model[0]

        super()._prepare_model_tokenizer()

    @classmethod
    def prepare_model(cls, args, model, *, template=None, train_dataset=None, task_type=None):
        model = super().prepare_model(args, model, template=template, train_dataset=train_dataset, task_type=task_type)
        if args.ref_adapters:
            if args.tuner_type in tuners_map:
                tuner: Tuner = tuners_map[args.tuner_type]
            else:
                tuner = Swift
            assert len(args.ref_adapters) == 1, f'args.ref_adapters: {args.ref_adapters}'
            # is_trainable: fix peft0.18.1
            kwargs = {}
            if version.parse(peft.__version__) >= version.parse('0.18'):
                kwargs['is_trainable'] = True
            model = tuner.from_pretrained(model, args.ref_adapters[0], adapter_name='ref_adapter', **kwargs)
            assert args.rlhf_type in {'dpo', 'kto',
                                      'grpo'}, 'Currently, only DPO, KTO, and GRPO support `ref_adapters`.'
            args.training_args.ref_adapter_name = 'ref_adapter'
        return model

    def _prepare_template(self) -> None:
        args = self.args
        super()._prepare_template()
        mode_mapping = {
            'kto': 'kto',
            'gkd': 'train',
            'ppo': 'transformers',
            'grpo': 'train',
            'offline_ppo': 'train',
            'offline_reinforce': 'train',
            'offline_grpo': 'train',
        }
        self.template.set_mode(mode_mapping.get(args.rlhf_type, 'rlhf'))

        if args.rlhf_type == 'ppo':
            args.training_args.stop_token_id = self.template.template_meta.stop_token_id

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        elif args.rlhf_type == 'offline_ppo':
            train_dataset, val_dataset = self._prepare_offline_ppo_dataset(train_dataset, val_dataset)
        elif args.rlhf_type == 'offline_reinforce':
            train_dataset, val_dataset = self._prepare_offline_reinforce_dataset(train_dataset, val_dataset)
        elif args.rlhf_type == 'offline_grpo':
            train_dataset, val_dataset = self._prepare_offline_grpo_dataset(train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _map_weighted_reward_columns(self, dataset, component_keys, weights, out_key: str, log_tag: str):
        """Overwrite ``out_key`` with sum_i weights[i] * row[component_keys[i]]."""
        if dataset is None or not component_keys:
            return dataset
        ws = weights or [1.0] * len(component_keys)

        def _row(example):
            total = 0.0
            for k, w in zip(component_keys, ws):
                v = example.get(k)
                try:
                    total += w * float(v) if v is not None else 0.0
                except (TypeError, ValueError):
                    total += 0.0
            example[out_key] = total
            return example

        dataset = dataset.map(_row)
        logger.info(
            '%s composite reward: %s = %s',
            log_tag,
            out_key,
            ' + '.join(f'{w}*{k}' for k, w in zip(component_keys, ws)),
        )
        return dataset

    def _prepare_offline_ppo_dataset(self, train_dataset, val_dataset):
        """Append the pre-collected answer as an assistant message to the conversation."""
        args = self.args
        answer_key = args.offline_ppo_answer_key
        reward_key = args.offline_ppo_reward_key

        def _add_answer_to_messages(example):
            answer = example.get(answer_key)
            if answer is not None:
                messages = example.get('messages', [])
                if messages and messages[-1]['role'] != 'assistant':
                    messages = list(messages)
                    messages.append({'role': 'assistant', 'content': str(answer)})
                    example['messages'] = messages
            return example

        if train_dataset is not None:
            train_dataset = train_dataset.map(_add_answer_to_messages)
            train_dataset = self._map_weighted_reward_columns(
                train_dataset,
                args.offline_ppo_reward_keys,
                args.offline_ppo_reward_weights,
                reward_key,
                'Offline PPO',
            )
        if val_dataset is not None:
            val_dataset = val_dataset.map(_add_answer_to_messages)
            val_dataset = self._map_weighted_reward_columns(
                val_dataset,
                args.offline_ppo_reward_keys,
                args.offline_ppo_reward_weights,
                reward_key,
                'Offline PPO',
            )
        return train_dataset, val_dataset

    def _prepare_offline_reinforce_dataset(self, train_dataset, val_dataset):
        """Prepare dataset for offline REINFORCE++ with group-level advantage computation.

        Groups samples by prompt text, computes advantage = reward - group_mean(reward)
        for groups with 2+ solutions. Singletons get advantage = 0.

        If ``offline_reinforce_advantage_key`` is set, skip group computation and use
        the pre-computed advantage column directly (e.g. TD / Q-V advantage).
        """
        from collections import defaultdict
        args = self.args
        answer_key = args.offline_reinforce_answer_key
        reward_key = args.offline_reinforce_reward_key
        advantage_key = getattr(args, 'offline_reinforce_advantage_key', None)

        def _add_answer_to_messages(example):
            answer = example.get(answer_key)
            if answer is not None:
                messages = example.get('messages', [])
                if messages and messages[-1]['role'] != 'assistant':
                    messages = list(messages)
                    messages.append({'role': 'assistant', 'content': str(answer)})
                    example['messages'] = messages
            return example

        def _compute_group_advantages(dataset):
            if dataset is None:
                return dataset

            dataset = dataset.map(_add_answer_to_messages)
            dataset = self._map_weighted_reward_columns(
                dataset,
                args.offline_reinforce_reward_keys,
                args.offline_reinforce_reward_weights,
                reward_key,
                'Offline REINFORCE++',
            )

            # Fast path: use precomputed advantage column (e.g. TD advantage from MCTS).
            if advantage_key:
                adv_values = [float(dataset[i].get(advantage_key, 0.0)) for i in range(len(dataset))]
                n_nonzero = sum(1 for v in adv_values if v != 0.0)
                if adv_values:
                    import statistics as _stats
                    _mean = _stats.mean(adv_values)
                    _std = _stats.pstdev(adv_values) if len(adv_values) > 1 else 0.0
                    _min, _max = min(adv_values), max(adv_values)
                    logger.info(
                        f'Offline REINFORCE++ using precomputed advantage column '
                        f'"{advantage_key}": {len(adv_values)} samples, {n_nonzero} non-zero '
                        f'({100*n_nonzero/max(1,len(adv_values)):.1f}%), '
                        f'mean={_mean:.4f} std={_std:.4f} min={_min:.4f} max={_max:.4f}')

                def _set_precomputed(example, idx):
                    example['_advantage'] = adv_values[idx]
                    return example

                dataset = dataset.map(_set_precomputed, with_indices=True)
                return dataset

            prompt_groups = defaultdict(list)
            for idx in range(len(dataset)):
                row = dataset[idx]
                messages = row.get('messages', [])
                prompt_parts = []
                for msg in messages:
                    if msg['role'] == 'assistant':
                        break
                    prompt_parts.append(f"{msg['role']}:{msg['content']}")
                prompt_key = '||'.join(prompt_parts)
                prompt_groups[prompt_key].append(idx)

            advantages = [0.0] * len(dataset)
            n_groups = len(prompt_groups)
            n_multi = sum(1 for idxs in prompt_groups.values() if len(idxs) > 1)
            n_singleton = n_groups - n_multi

            use_rank = getattr(args, 'offline_reinforce_use_rank_advantage', False)
            for prompt_key, indices in prompt_groups.items():
                if len(indices) < 2:
                    continue
                group_rewards = []
                for idx in indices:
                    r = float(dataset[idx].get(reward_key, 0.0))
                    group_rewards.append(r)
                if use_rank:
                    # Rank-based advantage: rank / (G-1) - 0.5, ties → 0
                    # For G=2: winner=+0.5, loser=-0.5, tie=0
                    G = len(group_rewards)
                    sorted_rewards = sorted(set(group_rewards))
                    if len(sorted_rewards) == 1:
                        # all tied → no signal
                        for idx in indices:
                            advantages[idx] = 0.0
                    else:
                        reward_to_rank = {r: i for i, r in enumerate(sorted_rewards)}
                        for idx, r in zip(indices, group_rewards):
                            advantages[idx] = reward_to_rank[r] / (G - 1) - 0.5
                else:
                    group_mean = sum(group_rewards) / len(group_rewards)
                    for idx, r in zip(indices, group_rewards):
                        advantages[idx] = r - group_mean

            logger.info(
                f'Offline REINFORCE++ dataset stats: {len(dataset)} samples, '
                f'{n_groups} unique prompts, {n_multi} groups with 2+ solutions, '
                f'{n_singleton} singletons (advantage=0)')

            def _set_advantage(example, idx):
                example['_advantage'] = advantages[idx]
                return example

            dataset = dataset.map(_set_advantage, with_indices=True)
            return dataset

        train_dataset = _compute_group_advantages(train_dataset)
        val_dataset = _compute_group_advantages(val_dataset)
        return train_dataset, val_dataset

    def _prepare_offline_grpo_dataset(self, train_dataset, val_dataset):
        """Prepare dataset for offline Dr. GRPO with length-normalised group advantages.

        For each group of responses sharing the same prompt:
            r̃_i = r_i / max(n_tokens_i, 1)      (length-normalised reward)
            A_i  = r̃_i - mean_j(r̃_j)            (group-mean-centred advantage)

        Response token counts are computed directly by tokenizing the raw answer text
        (add_special_tokens=False), so no pre-computed length column is needed.
        Group sizes are variable.  Singletons receive advantage = 0.
        """
        from collections import defaultdict
        args = self.args
        answer_key = args.offline_grpo_answer_key
        reward_key = args.offline_grpo_reward_key
        tokenizer = self.processor.tokenizer

        def _add_answer_to_messages(example):
            answer = example.get(answer_key)
            if answer is not None:
                messages = example.get('messages', [])
                if messages and messages[-1]['role'] != 'assistant':
                    messages = list(messages)
                    messages.append({'role': 'assistant', 'content': str(answer)})
                    example['messages'] = messages
            return example

        def _compute_dr_grpo_advantages(dataset):
            if dataset is None:
                return dataset

            dataset = dataset.map(_add_answer_to_messages)
            dataset = self._map_weighted_reward_columns(
                dataset,
                args.offline_grpo_reward_keys,
                args.offline_grpo_reward_weights,
                reward_key,
                'Offline Dr. GRPO',
            )

            # Pre-compute token lengths for all answers in one pass.
            all_answers = [str(dataset[idx].get(answer_key, '') or '') for idx in range(len(dataset))]
            token_lengths = [
                max(len(ids), 1)
                for ids in tokenizer(all_answers, add_special_tokens=False)['input_ids']
            ]

            # Group samples by prompt (all turns before the last assistant message).
            prompt_groups = defaultdict(list)
            for idx in range(len(dataset)):
                row = dataset[idx]
                messages = row.get('messages', [])
                prompt_parts = []
                for msg in messages:
                    if msg['role'] == 'assistant':
                        break
                    prompt_parts.append(f"{msg['role']}:{msg['content']}")
                prompt_key = '||'.join(prompt_parts)
                prompt_groups[prompt_key].append(idx)

            advantages = [0.0] * len(dataset)
            n_groups = len(prompt_groups)
            n_multi = sum(1 for idxs in prompt_groups.values() if len(idxs) > 1)
            n_singleton = n_groups - n_multi

            for prompt_key, indices in prompt_groups.items():
                if len(indices) < 2:
                    # singleton → no contrastive signal, advantage stays 0
                    continue

                r_tilde = []
                for idx in indices:
                    r = float(dataset[idx].get(reward_key, 0.0))
                    r_tilde.append(r / token_lengths[idx])

                group_mean = sum(r_tilde) / len(r_tilde)
                for idx, rt in zip(indices, r_tilde):
                    advantages[idx] = rt - group_mean

            logger.info(
                f'Offline Dr. GRPO dataset stats: {len(dataset)} samples, '
                f'{n_groups} unique prompts, {n_multi} groups with 2+ responses '
                f'(variable group size), {n_singleton} singletons (advantage=0). '
                f'Token lengths computed via tokenizer (add_special_tokens=False).')

            def _set_advantage(example, idx):
                example['_advantage'] = advantages[idx]
                return example

            dataset = dataset.map(_set_advantage, with_indices=True)
            return dataset

        train_dataset = _compute_dr_grpo_advantages(train_dataset)
        val_dataset = _compute_dr_grpo_advantages(val_dataset)
        return train_dataset, val_dataset

    def _prepare_chord_sft_dataset(self):
        # prepare expert sft dataset for chord
        args = self.args
        assert hasattr(args, 'chord_sft_dataset') and args.chord_sft_dataset
        dataset_kwargs = args.get_dataset_kwargs()
        chord_sft_datasets = []
        # TODO: validatition
        chord_sft_dataset, _ = load_dataset(
            args.chord_sft_dataset, split_dataset_ratio=0, shuffle=args.dataset_shuffle, **dataset_kwargs)
        chord_sft_dataset, _ = self._encode_dataset(chord_sft_dataset, None, pre_process=True)
        chord_sft_datasets.append(chord_sft_dataset)
        chord_sft_dataset = DatasetLoader.concat_datasets(chord_sft_datasets)
        datasets = [chord_sft_dataset, None]
        datasets = self._post_process_datasets(datasets)
        return datasets

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        for key in ['ref', 'reward', 'value', 'teacher']:
            key = f'{key}_model'
            model = getattr(self, key, None)
            if self.args.rlhf_type in ('offline_ppo', 'offline_reinforce', 'offline_grpo'):
                if key == 'ref_model' and model is not None:
                    trainer_kwargs[key] = model
                continue
            if model or self.args.rlhf_type == 'ppo' and key != 'teacher_model':
                trainer_kwargs[key] = model
        if self.args.rlhf_type == 'offline_ppo':
            trainer_kwargs['reward_key'] = self.args.offline_ppo_reward_key
        if self.args.rlhf_type == 'offline_reinforce':
            trainer_kwargs['reward_key'] = self.args.offline_reinforce_reward_key
            trainer_kwargs['sample_weight_key'] = self.args.offline_reinforce_sample_weight_key
        if self.args.rlhf_type == 'offline_grpo':
            trainer_kwargs['reward_key'] = self.args.offline_grpo_reward_key
        if hasattr(self, 'reward_template'):
            trainer_kwargs['reward_template'] = self.reward_template
        if self.args.rlhf_type in ['grpo', 'gkd']:
            trainer_kwargs['vllm_client'] = self.args.vllm_client
        if self.args.rlhf_type == 'grpo':
            trainer_kwargs['reward_funcs'] = self.args.reward_funcs
            if self.args.chord_sft_dataset:
                trainer_kwargs['chord_sft_dataset'], _ = self._prepare_chord_sft_dataset()
        if self.args.rlhf_type == 'gkd':
            if self.args.teacher_deepspeed:
                trainer_kwargs['teacher_deepspeed_config'] = self.args.teacher_deepspeed
            trainer_kwargs['gkd_logits_topk'] = self.args.gkd_logits_topk
            if self.args.teacher_model_server:
                trainer_kwargs['teacher_model_server'] = self.args.teacher_model_server
            trainer_kwargs['teacher_use_disable_adapter'] = getattr(self.args, '_teacher_use_disable_adapter', False)
        return trainer_kwargs


def rlhf_main(args: Optional[Union[List[str], RLHFArguments]] = None):
    return SwiftRLHF(args).main()
