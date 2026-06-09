# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from copy import deepcopy
from dataclasses import dataclass
from datasets import Dataset as HfDataset
from typing import Any, Dict, List, Optional, Sequence

from swift.utils import get_logger

logger = get_logger()

_PAIR_KEYS = ['images', 'videos', 'audios', 'tools', 'objects']

DEFAULT_OPSD_TEACHER_PROMPT = (
    '{prompt}\n\n'
    'Here is a candidate answer that may be correct:\n'
    '{answer}\n\n'
    'Please carefully verify it, then articulate your own reasoning and final answer.'
)

DEFAULT_OPSD_TEACHER_PROMPT_WITH_LABEL = (
    '{prompt}\n\n'
    'Here is a candidate answer that may be correct:\n'
    '{answer}\n\n'
    'The ground-truth answer is:\n'
    '{label}\n\n'
    'Please carefully verify the candidate answer against the ground-truth answer, '
    'then articulate your own reasoning and final answer.'
)


@dataclass
class _PreparedSample:
    prompt_messages: List[Dict[str, Any]]
    answer: Optional[str]
    label: Optional[str]
    judge_score: Optional[float]
    rank_score: Optional[float]
    base_row: Dict[str, Any]
    prompt_text: str
    prompt_key: str


def apply_ppo_data_transform(dataset: Optional[HfDataset], args, *, split: str) -> Optional[HfDataset]:
    transform = getattr(args, 'ppo_data_transform', 'none')
    if dataset is None or transform in {None, 'none'}:
        return dataset
    if not isinstance(dataset, HfDataset):
        raise ValueError('`--ppo_data_transform` currently only supports non-streaming datasets.')

    if transform == 'sft':
        return _transform_to_sft(dataset, args, split=split)
    if transform == 'dpo':
        return _transform_to_dpo(dataset, args, split=split)
    if transform == 'opsd':
        return _transform_to_opsd(dataset, args, split=split)
    raise ValueError(f'Unsupported `ppo_data_transform`: {transform}')


def _transform_to_sft(dataset: HfDataset, args, *, split: str) -> HfDataset:
    rows = []
    dropped = 0
    for row in dataset:
        sample = _prepare_sample(row, args)
        if sample.answer is None or not _passes_threshold(sample.judge_score, args):
            dropped += 1
            continue
        item = dict(sample.base_row)
        item['messages'] = sample.prompt_messages + [{'role': 'assistant', 'content': sample.answer}]
        rows.append(item)
    logger.info(
        'PPO data transform (%s -> SFT): kept %s / %s samples.',
        split,
        len(rows),
        len(rows) + dropped,
    )
    return _build_dataset(rows, dataset)


def _transform_to_opsd(dataset: HfDataset, args, *, split: str) -> HfDataset:
    rows = []
    dropped = 0
    include_label = getattr(args, 'ppo_data_include_label_in_teacher_prompt', False)
    teacher_template = getattr(args, 'ppo_data_teacher_prompt', None)
    for row in dataset:
        sample = _prepare_sample(row, args)
        if sample.answer is None or not _passes_threshold(sample.judge_score, args):
            dropped += 1
            continue
        if not sample.prompt_text:
            dropped += 1
            continue
        item = dict(sample.base_row)
        item['messages'] = sample.prompt_messages
        item['teacher_prompt'] = _build_opsd_teacher_prompt(
            sample=sample, teacher_template=teacher_template, include_label=include_label)
        rows.append(item)
    logger.info(
        'PPO data transform (%s -> OPSD): kept %s / %s samples.',
        split,
        len(rows),
        len(rows) + dropped,
    )
    return _build_dataset(rows, dataset)


def _transform_to_dpo(dataset: HfDataset, args, *, split: str) -> HfDataset:
    grouped: Dict[str, List[_PreparedSample]] = {}
    skipped = 0
    skipped_missing_answer = 0
    skipped_missing_score = 0
    for row in dataset:
        sample = _prepare_sample(row, args)
        if sample.answer is None:
            skipped += 1
            skipped_missing_answer += 1
            continue
        if sample.rank_score is None:
            skipped += 1
            skipped_missing_score += 1
            continue
        grouped.setdefault(sample.prompt_key, []).append(sample)

    rows = []
    no_pair = 0
    for samples in grouped.values():
        if len(samples) < 2:
            no_pair += 1
            continue
        samples.sort(key=lambda sample: sample.rank_score, reverse=True)
        chosen = samples[0]
        rejected = next((sample for sample in reversed(samples) if sample.answer != chosen.answer), None)
        if rejected is None:
            no_pair += 1
            continue
        item = dict(chosen.base_row)
        item['messages'] = chosen.prompt_messages + [{'role': 'assistant', 'content': chosen.answer}]
        item['rejected_response'] = rejected.answer
        rows.append(item)

    logger.info(
        'PPO data transform (%s -> DPO): built %s pairs from %s prompts, skipped %s raw samples '
        '(missing answer: %s, missing score: %s) and %s prompts without valid pairs.',
        split,
        len(rows),
        len(grouped),
        skipped,
        skipped_missing_answer,
        skipped_missing_score,
        no_pair,
    )
    return _build_dataset(rows, dataset)


def _prepare_sample(row: Dict[str, Any], args) -> _PreparedSample:
    prompt_messages, answer = _split_prompt_and_answer(row, getattr(args, 'ppo_data_answer_key', 'answer'))
    label = _maybe_stringify(row.get(getattr(args, 'ppo_data_label_key', None)))
    judge_score = _safe_float(row.get(getattr(args, 'ppo_data_judge_key', 'expected_acc_reward')))
    score_keys = getattr(args, 'ppo_data_score_keys', None) or [getattr(args, 'ppo_data_judge_key',
                                                                          'expected_acc_reward')]
    score_weights = getattr(args, 'ppo_data_score_weights', None) or [1.0] * len(score_keys)
    rank_score = 0.0
    has_score = False
    for key, weight in zip(score_keys, score_weights):
        value = _safe_float(row.get(key))
        if value is not None:
            rank_score += weight * value
            has_score = True
    if not has_score:
        rank_score = None
    prompt_text = _get_last_user_content(prompt_messages)
    return _PreparedSample(
        prompt_messages=prompt_messages,
        answer=answer,
        label=label,
        judge_score=judge_score,
        rank_score=rank_score,
        base_row=_build_base_row(row),
        prompt_text=prompt_text,
        prompt_key=_serialize_prompt_key(prompt_messages, row),
    )


def _build_base_row(row: Dict[str, Any]) -> Dict[str, Any]:
    messages = [dict(message) for message in row.get('messages', [])]
    base_row = {'messages': messages}
    for key in _PAIR_KEYS + ['channel']:
        if key in row:
            base_row[key] = deepcopy(row[key])
    return base_row


def _split_prompt_and_answer(row: Dict[str, Any], answer_key: str):
    messages = [dict(message) for message in row.get('messages', [])]
    answer = row.get(answer_key)
    if answer is None and messages and messages[-1]['role'] == 'assistant':
        answer = messages.pop()['content']
    elif messages and messages[-1]['role'] == 'assistant':
        messages.pop()
    if answer is not None:
        answer = str(answer)
    return messages, answer


def _build_opsd_teacher_prompt(*, sample: _PreparedSample, teacher_template: Optional[str], include_label: bool) -> str:
    label = sample.label if include_label and sample.label is not None else ''
    if teacher_template is None:
        if label:
            teacher_template = DEFAULT_OPSD_TEACHER_PROMPT_WITH_LABEL
        else:
            teacher_template = DEFAULT_OPSD_TEACHER_PROMPT
    return teacher_template.format(prompt=sample.prompt_text, answer=sample.answer, label=label)


def _passes_threshold(value: Optional[float], args) -> bool:
    if value is None:
        return False
    return value > getattr(args, 'ppo_data_judge_threshold', 0.5)


def _get_last_user_content(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get('role') == 'user':
            return str(message.get('content', ''))
    return ''


def _serialize_prompt_key(messages: Sequence[Dict[str, Any]], row: Dict[str, Any]) -> str:
    payload = {'messages': list(messages)}
    for key in _PAIR_KEYS:
        if key in row:
            payload[key] = row[key]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _build_dataset(rows: List[Dict[str, Any]], dataset: HfDataset) -> HfDataset:
    if rows:
        return HfDataset.from_list(rows)
    return dataset.select([])
