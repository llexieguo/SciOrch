#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import math
import mimetypes
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sciorch.core.memory import MainMemory
from sciorch.prompts.main_reasoning import build_main_prompt, build_model_guidance_text, format_options_with_letters
from sciorch.prompts.sub_reasoning import build_sub_prompt
from sciorch.adapters.reasoning.dataset import _parse_answer_index
from sciorch.types import AttemptRecord, DelegateResult, ReasoningSample


JsonDict = dict[str, Any]
FormatterFn = Callable[[dict[str, Any]], dict[str, Any] | None]

FALLBACK_ORCHESTRA_REASONS: frozenset[str] = frozenset({
    "Submit rejected by guardrail; final answer was not parseable.",
    "Submit rejected by guardrail; no informative delegate findings yet.",
    "Malformed action from MainAgent response; fallback to delegate.",
    "Forced final-turn submit after unparsable final answer.",
    "Forced final-turn submit after model requested another delegation.",
    "Forced final-turn submit after malformed MainAgent response.",
    "Submit-only recovery call failed.",
})


@dataclass
class RewardStats:
    reward: float
    terminal_count: int
    submit_count: int
    correct_count: int
    has_open_descendant: bool


@dataclass
class StrictSftSelection:
    path_node_ids: list[str]
    keep_node_ids: set[str]
    legal_delegate_node_ids: set[str]
    legal_submit_node_ids: set[str]
    selected_path_count: int = 0
    skip_reason: str | None = None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_run_and_samples_dir(input_path: str | Path) -> tuple[Path, Path]:
    path = Path(input_path).expanduser().resolve()
    if path.name == "samples":
        return path.parent, path
    if (path / "samples").is_dir():
        return path, path / "samples"
    raise ValueError(f"Expected a run dir or samples dir, got: {path}")


def parse_formatter(spec: str | None) -> FormatterFn | None:
    if not spec:
        return None
    path_text, _, func_name = spec.partition(":")
    formatter_path = Path(path_text).expanduser().resolve()
    if not formatter_path.exists():
        raise FileNotFoundError(f"Formatter file not found: {formatter_path}")
    func_name = func_name or "format_record"
    module_name = f"msswift_formatter_{formatter_path.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, formatter_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Failed to load formatter module from {formatter_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    formatter = getattr(module, func_name, None)
    if formatter is None or not callable(formatter):
        raise AttributeError(f"Formatter function '{func_name}' not found in {formatter_path}")
    return formatter


def parse_template_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")
    return path


def parse_prompt_config_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prompt config file not found: {path}")
    return path


def load_prompt_config(path: Path | None) -> JsonDict:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt config must be a YAML mapping object: {path}")
    return payload


def canonical_json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_md_template(template_text: str, values: dict[str, Any]) -> str:
    rendered = template_text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered.strip()


def render_thinking_block(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("<think>") and text.endswith("</think>"):
        return text
    return f"<think>{text}</think>"


def build_full_completion(call: JsonDict) -> str:
    raw_text = str(call.get("raw_text") or "")
    thinking_block = render_thinking_block(call.get("thinking"))
    if thinking_block and raw_text:
        return f"{thinking_block}\n\n{raw_text}"
    return thinking_block or raw_text


def normalize_completion(call: JsonDict, *, mode: str) -> str:
    raw_text = str(call.get("raw_text") or "")
    parsed = call.get("parsed") or {}
    has_thinking = bool(str(call.get("thinking") or "").strip())
    if mode == "raw":
        return raw_text
    if mode == "parsed":
        return canonical_json_text(parsed) if parsed else raw_text
    if mode == "full":
        return build_full_completion(call)
    if has_thinking:
        return build_full_completion(call)
    if parsed:
        return canonical_json_text(parsed)
    return raw_text


def maybe_add_image_tokens(user_prompt: str, image_count: int, mode: str) -> str:
    if image_count <= 0 or mode == "none":
        return user_prompt
    if "<image>" in user_prompt:
        return user_prompt
    prefix = "<image>" * image_count
    if not user_prompt:
        return prefix
    return f"{prefix}\n{user_prompt}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _looks_like_pil_image(value: Any) -> bool:
    return hasattr(value, "save") and hasattr(value, "mode") and hasattr(value, "size")


def _pil_to_data_url(image: Any) -> str:
    fmt = str(getattr(image, "format", "PNG") or "PNG").upper()
    mime = "image/png" if fmt == "PNG" else f"image/{fmt.lower()}"
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def coerce_image_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    if isinstance(value, dict):
        for key in ("path", "url", "image", "file_name", "filename"):
            raw = value.get(key)
            if isinstance(raw, str) and raw:
                return raw
        data = value.get("bytes")
        if isinstance(data, (bytes, bytearray)):
            encoded = base64.b64encode(bytes(data)).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    if _looks_like_pil_image(value):
        return _pil_to_data_url(value)
    return None


def infer_extension_from_mime(mime_type: str) -> str:
    guessed = mimetypes.guess_extension(mime_type, strict=False)
    if guessed == ".jpe":
        return ".jpg"
    if guessed:
        return guessed
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/webp":
        return ".webp"
    return ".bin"


def parse_data_url(value: str) -> tuple[str, bytes] | None:
    if not value.startswith("data:"):
        return None
    header, sep, payload = value.partition(",")
    if not sep:
        return None
    mime_type = header[5:].split(";", 1)[0].strip() or "application/octet-stream"
    if ";base64" not in header:
        return None
    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except Exception:
        return None
    return mime_type, raw_bytes


def maybe_materialize_images(
    *,
    images: list[str],
    storage_mode: str,
    images_output_dir: Path | None,
    output_root: Path,
    image_path_style: str,
    overwrite_images_dir: bool,
    state: dict[str, Any],
) -> tuple[list[str], list[str]]:
    # Returns:
    # - prompt_images: always used for <image> token counting
    # - record_images: what gets written into JSON record
    if not images:
        return [], []
    if storage_mode == "inline":
        return images, images

    if images_output_dir is None:
        images_root = (output_root / "images").resolve()
    else:
        images_root = images_output_dir
    if not state.get("images_dir_ready"):
        if images_root.exists():
            if overwrite_images_dir:
                shutil.rmtree(images_root)
            else:
                raise FileExistsError(f"Images output directory already exists: {images_root}. Use --overwrite-images-dir.")
        images_root.mkdir(parents=True, exist_ok=True)
        state["images_dir_ready"] = True
        state["images_root"] = str(images_root)

    images_root = Path(state["images_root"])
    prompt_images = list(images)
    record_images: list[str] = []
    for value in images:
        parsed = parse_data_url(value)
        if parsed is None:
            # Keep non-data URLs unchanged.
            record_images.append(value)
            continue
        mime_type, raw_bytes = parsed
        sha1 = hashlib.sha1(raw_bytes).hexdigest()
        ext = infer_extension_from_mime(mime_type)
        target = images_root / f"{sha1}{ext}"
        if not target.exists():
            target.write_bytes(raw_bytes)
        if image_path_style == "absolute":
            record_images.append(str(target.resolve()))
        else:
            record_images.append(target.relative_to(output_root).as_posix())
        state["materialized_image_count"] = int(state.get("materialized_image_count", 0)) + 1

    if storage_mode == "detached":
        return prompt_images, []
    return prompt_images, record_images


def load_task_images(
    *,
    task_ids: set[str],
    dataset_name: str | None,
    dataset_split: str | None,
    strict: bool,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if not task_ids or not dataset_name or not dataset_split:
        return {}, {"enabled": False, "reason": "missing_dataset_config"}
    try:
        from datasets import load_dataset
    except Exception as exc:
        if strict:
            raise RuntimeError("datasets package is required to resolve images") from exc
        return {}, {"enabled": False, "reason": f"datasets_import_failed: {exc}"}

    try:
        import os
        if os.path.isfile(dataset_name):
            raw_dataset = load_dataset("json", data_files=dataset_name, split=dataset_split)
        else:
            raw_dataset = load_dataset(dataset_name, split=dataset_split)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Failed to load dataset {dataset_name}:{dataset_split}") from exc
        return {}, {"enabled": False, "reason": f"dataset_load_failed: {exc}"}

    images_by_task: dict[str, list[str]] = {}
    unresolved_count = 0
    for item in raw_dataset:
        task_id = str(item.get("idx", item.get("id", "")))
        if task_id not in task_ids:
            continue
        resolved_images: list[str] = []
        for image_value in list(item.get("images", [])):
            resolved = coerce_image_value(image_value)
            if resolved is None:
                unresolved_count += 1
                continue
            resolved_images.append(resolved)
        images_by_task[task_id] = resolved_images
        if len(images_by_task) == len(task_ids):
            break

    missing_tasks = sorted(task_ids - set(images_by_task))
    if strict and missing_tasks:
        raise RuntimeError(f"Failed to resolve images for task ids: {missing_tasks}")
    return images_by_task, {
        "enabled": True,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "resolved_task_count": len(images_by_task),
        "missing_task_ids": missing_tasks,
        "unresolved_image_items": unresolved_count,
    }


def load_dataset_result_stubs(
    *,
    task_ids: set[str],
    dataset_name: str | None,
    dataset_split: str | None,
) -> dict[str, JsonDict]:
    """
    Map task_id -> fields compatible with result.json (question, options, gold, etc.).

    Used when a run stopped before writing per-sample result.json; the tree in
    nodes.jsonl still supports expected_acc_reward, but template prompt rebuild needs
    dataset text fields.
    """
    if not task_ids or not dataset_name or not dataset_split:
        return {}
    try:
        from datasets import load_dataset
    except Exception:
        return {}
    try:
        import os
        if os.path.isfile(dataset_name):
            raw_dataset = load_dataset("json", data_files=dataset_name, split=dataset_split)
        else:
            raw_dataset = load_dataset(dataset_name, split=dataset_split)
    except Exception:
        return {}
    out: dict[str, JsonDict] = {}
    remaining = set(task_ids)
    for item in raw_dataset:
        tid = str(item.get("idx", item.get("id", "")))
        if tid not in remaining:
            continue
        options = [str(option) for option in item.get("options", [])]
        answer_index = _parse_answer_index(item.get("answer", 0), options)
        gold_letter = ""
        if 0 <= answer_index < 26:
            gold_letter = chr(ord("A") + answer_index)
        out[tid] = {
            "task_id": tid,
            "question": str(item.get("question", "")),
            "options": options,
            "gold_answer_letter": gold_letter,
            "discipline": str(item.get("discipline", "unknown")),
        }
        remaining.discard(tid)
        if not remaining:
            break
    return out


def build_nodes_by_id(nodes: list[JsonDict]) -> dict[str, JsonDict]:
    return {str(node["node_id"]): node for node in nodes}


def parse_main_action(call: JsonDict) -> str | None:
    parsed = call.get("parsed")
    if isinstance(parsed, dict):
        action = parsed.get("action")
        if isinstance(action, str) and action:
            return action
    raw_text = str(call.get("raw_text") or "").strip()
    if not raw_text:
        return None
    try:
        payload = json.loads(raw_text)
    except Exception:
        return None
    action = payload.get("action")
    if isinstance(action, str) and action:
        return action
    return None


def has_nonempty_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def is_informative_delegate_node(node: JsonDict) -> bool:
    if str(node.get("action") or "") != "delegate":
        return False
    if not bool(node.get("delegate_parse_ok")):
        return False
    if node.get("delegate_confidence") is None:
        return False
    return has_nonempty_text(node.get("delegate_answer")) or has_nonempty_text(node.get("delegate_evidence"))


def sort_correct_trajectory_key(trajectory: JsonDict) -> tuple[float, float, float, float]:
    return (
        safe_float(trajectory.get("selection_score"), default=0.0),
        safe_float(trajectory.get("latest_delegate_confidence"), default=-1.0),
        -safe_float(trajectory.get("trajectory_cost"), default=0.0),
        safe_float(trajectory.get("depth"), default=0.0),
    )


def list_correct_trajectories(
    *,
    result: JsonDict,
    latest: JsonDict,
    nodes_by_id: dict[str, JsonDict],
) -> list[JsonDict]:
    trajectories = latest.get("final_trajectories") or []
    correct_trajectories: list[JsonDict] = []
    for trajectory in trajectories:
        leaf_node_id = str(trajectory.get("leaf_node_id") or "")
        leaf_node = nodes_by_id.get(leaf_node_id)
        if leaf_node is None:
            continue
        if not bool(trajectory.get("correct")):
            continue
        if str(leaf_node.get("action") or "") != "submit":
            continue
        if not bool(leaf_node.get("boxed_letter")) or not bool(leaf_node.get("is_correct")):
            continue
        correct_trajectories.append(trajectory)
    if not correct_trajectories:
        return []

    preferred_leaf_id = str(result.get("best_leaf_node_id") or "")
    if preferred_leaf_id:
        preferred: list[JsonDict] = []
        remaining: list[JsonDict] = []
        for trajectory in correct_trajectories:
            if str(trajectory.get("leaf_node_id") or "") == preferred_leaf_id:
                preferred.append(trajectory)
            else:
                remaining.append(trajectory)
        preferred.sort(key=sort_correct_trajectory_key, reverse=True)
        remaining.sort(key=sort_correct_trajectory_key, reverse=True)
        return preferred + remaining

    correct_trajectories.sort(key=sort_correct_trajectory_key, reverse=True)
    return correct_trajectories


def select_best_correct_trajectory(
    *,
    result: JsonDict,
    latest: JsonDict,
    nodes_by_id: dict[str, JsonDict],
) -> JsonDict | None:
    correct_trajectories = list_correct_trajectories(result=result, latest=latest, nodes_by_id=nodes_by_id)
    if not correct_trajectories:
        return None
    return correct_trajectories[0]


def build_strict_sft_selection(
    *,
    result: JsonDict,
    latest: JsonDict,
    nodes_by_id: dict[str, JsonDict],
    calls_by_node_id: dict[str, JsonDict],
    min_delegate_findings: int,
    filter_mode: str,
    max_correct_paths_per_sample: int,
) -> StrictSftSelection:
    correct_trajectories = list_correct_trajectories(result=result, latest=latest, nodes_by_id=nodes_by_id)
    if not correct_trajectories:
        return StrictSftSelection(
            path_node_ids=[],
            keep_node_ids=set(),
            legal_delegate_node_ids=set(),
            legal_submit_node_ids=set(),
            selected_path_count=0,
            skip_reason="no_correct_terminal_path",
        )

    if filter_mode == "strict_orchestrator":
        selected_trajectories = correct_trajectories[:1]
    elif filter_mode == "all_correct_paths_capped":
        selected_trajectories = correct_trajectories[:max(1, max_correct_paths_per_sample)]
    else:
        raise ValueError(f"Unsupported strict SFT filter mode: {filter_mode}")

    path_node_ids: list[str] = []
    keep_node_ids: set[str] = set()
    legal_delegate_node_ids: set[str] = set()
    legal_submit_node_ids: set[str] = set()

    for trajectory in selected_trajectories:
        current_path_node_ids = [str(node_id) for node_id in trajectory.get("node_ids") or [] if str(node_id) in nodes_by_id]
        if not current_path_node_ids:
            continue
        path_node_ids.extend(current_path_node_ids)

        path_delegate_node_ids: list[str] = []
        for node_id in current_path_node_ids:
            node = nodes_by_id[node_id]
            if str(node.get("action") or "") != "delegate":
                continue
            if is_fallback_call(node):
                continue
            call = calls_by_node_id.get(node_id)
            if call is None or parse_main_action(call) != "delegate_task":
                continue
            if is_informative_delegate_node(node):
                path_delegate_node_ids.append(node_id)
                legal_delegate_node_ids.add(node_id)
                keep_node_ids.add(node_id)

        submit_node_id = str(trajectory.get("leaf_node_id") or "")
        submit_node = nodes_by_id.get(submit_node_id)
        submit_call = calls_by_node_id.get(submit_node_id)
        if (
            submit_node is not None
            and str(submit_node.get("action") or "") == "submit"
            and bool(submit_node.get("boxed_letter"))
            and bool(submit_node.get("is_correct"))
            and submit_call is not None
            and parse_main_action(submit_call) == "submit"
            and len(path_delegate_node_ids) >= min_delegate_findings
        ):
            legal_submit_node_ids.add(submit_node_id)
            keep_node_ids.add(submit_node_id)

    if not path_node_ids:
        return StrictSftSelection(
            path_node_ids=[],
            keep_node_ids=set(),
            legal_delegate_node_ids=set(),
            legal_submit_node_ids=set(),
            selected_path_count=0,
            skip_reason="empty_selected_path",
        )

    skip_reason = None if keep_node_ids else "path_has_no_legal_sft_nodes"
    return StrictSftSelection(
        path_node_ids=list(dict.fromkeys(path_node_ids)),
        keep_node_ids=keep_node_ids,
        legal_delegate_node_ids=legal_delegate_node_ids,
        legal_submit_node_ids=legal_submit_node_ids,
        selected_path_count=len(selected_trajectories),
        skip_reason=skip_reason,
    )


def should_keep_call_for_strict_sft(
    *,
    call: JsonDict,
    node: JsonDict,
    selection: StrictSftSelection | None,
) -> bool:
    if selection is None:
        return True
    node_id = str(node.get("node_id") or "")
    if node_id not in selection.keep_node_ids:
        return False
    action = str(node.get("action") or "")
    if action == "delegate":
        return node_id in selection.legal_delegate_node_ids
    if action == "submit":
        return node_id in selection.legal_submit_node_ids
    return False


def gold_letter_to_index(letter: str | None) -> int:
    if not letter:
        return 0
    value = str(letter).strip().upper()
    if len(value) != 1 or not ("A" <= value <= "Z"):
        return 0
    return ord(value) - ord("A")


def reconstruct_path(node_id: str | None, nodes_by_id: dict[str, JsonDict]) -> list[JsonDict]:
    if not node_id:
        return []
    path: list[JsonDict] = []
    current = nodes_by_id.get(node_id)
    while current is not None:
        path.append(current)
        parent_id = current.get("parent_id")
        if parent_id is None:
            break
        current = nodes_by_id.get(str(parent_id))
    path.reverse()
    return path


def build_memory_from_path(path: list[JsonDict]) -> MainMemory:
    memory = MainMemory()
    for node in path[1:]:
        if node.get("action") != "delegate":
            continue
        delegate_result = DelegateResult(
            raw_answer_text="",
            answer=node.get("delegate_answer"),
            confidence=node.get("delegate_confidence"),
            reasoning_summary=str(node.get("delegate_evidence") or ""),
            parse_ok=bool(node.get("delegate_parse_ok")),
            error=node.get("error"),
            cost=0.0,
            input_tokens=0,
            output_tokens=0,
        )
        memory.add_attempt(
            AttemptRecord(
                attempt_index=int(node.get("depth") or len(memory.attempts) + 1),
                model=str(node.get("chosen_model") or "-"),
                instruction=str(node.get("focus_question") or node.get("instruction") or ""),
                delegate_result=delegate_result,
                main_reasoning=str(node.get("orchestra_reasoning") or ""),
            )
        )
    return memory


def build_base_sample(*, result: JsonDict, view: JsonDict | None, images: list[str]) -> ReasoningSample:
    reference_steps = list((view or {}).get("reference", {}).get("steps", []) or [])
    return ReasoningSample(
        task_id=str(result.get("task_id") or ""),
        question=str(result.get("question") or ""),
        options=[str(option) for option in list(result.get("options", []) or [])],
        answer_index=gold_letter_to_index(result.get("gold_answer_letter")),
        steps=[str(step) for step in reference_steps],
        discipline=str(result.get("discipline") or "unknown"),
        images=images,
    )


def build_node_sample(
    *,
    sample: ReasoningSample,
    parent_path: list[JsonDict],
    node: JsonDict,
    orchestra_model: str,
) -> ReasoningSample:
    del parent_path, node, orchestra_model
    question = sample.question
    return ReasoningSample(
        task_id=sample.task_id,
        question=question,
        options=sample.options,
        answer_index=sample.answer_index,
        steps=sample.steps,
        discipline=sample.discipline,
        images=sample.images,
    )


def format_options_numbered(options: list[str]) -> str:
    return "\n".join(f"{idx + 1}. {option}" for idx, option in enumerate(options))


def build_main_template_values(
    *,
    base_sample: ReasoningSample,
    node_sample: ReasoningSample,
    memory: MainMemory,
    node: JsonDict,
    run_config: JsonDict,
    result: JsonDict,
    step_index: int,
    max_steps: int,
    force_submit: bool,
    model_pool: list[str],
) -> dict[str, Any]:
    remaining_steps = max_steps - step_index + 1
    action_line = "You MUST output action=submit on this turn." if force_submit else "You may choose delegate_task or submit."
    if remaining_steps <= 1:
        urgency = "FINAL STEP. You MUST submit now."
    elif remaining_steps <= 2:
        urgency = "URGENT: Submit now. You have almost no steps left."
    elif remaining_steps <= 3:
        urgency = "Running low on steps. Submit if you have any informative finding."
    else:
        urgency = ""
    return {
        "task_id": base_sample.task_id,
        "node_id": node.get("node_id") or "",
        "parent_id": node.get("parent_id") or "",
        "round_index": node.get("round_index") or "",
        "depth": node.get("depth") or "",
        "question": base_sample.question,
        "question_with_branch_context": node_sample.question,
        "options_text": format_options_with_letters(base_sample.options),
        "step_history_text": memory.as_brief_text(),
        "step_index": step_index,
        "max_steps": max_steps,
        "remaining_steps": remaining_steps,
        "force_submit": str(force_submit).lower(),
        "action_line": action_line,
        "urgency": urgency,
        "urgency_suffix": f" {urgency}" if urgency else "",
        "model_guidance_text": build_model_guidance_text(model_pool),
        "model_pool_json": canonical_json_text(model_pool),
        "orchestra_model": result.get("orchestra_model") or run_config.get("orchestra_model") or "",
        "branch_instruction": node.get("instruction") or "",
    }


def build_sub_template_values(
    *,
    base_sample: ReasoningSample,
    memory: MainMemory,
    node: JsonDict,
) -> dict[str, Any]:
    return {
        "task_id": base_sample.task_id,
        "node_id": node.get("node_id") or "",
        "parent_id": node.get("parent_id") or "",
        "round_index": node.get("round_index") or "",
        "depth": node.get("depth") or "",
        "question": base_sample.question,
        "instruction": node.get("focus_question") or node.get("instruction") or "",
        "options_text": format_options_numbered(base_sample.options),
        "prior_steps_text": memory.delegate_context_summary(),
    }


def render_prompts(
    *,
    prompt_source: str,
    main_template_path: Path | None,
    sub_template_path: Path | None,
    prompt_config: JsonDict,
    call: JsonDict,
    node: JsonDict,
    nodes_by_id: dict[str, JsonDict],
    result: JsonDict,
    view: JsonDict | None,
    run_config: JsonDict,
    images: list[str],
) -> tuple[str, str]:
    if prompt_source == "logged":
        return str(call.get("system_prompt") or ""), str(call.get("user_prompt") or "")

    actor = str(call.get("actor") or "")
    if actor not in {"main", "delegate"}:
        return str(call.get("system_prompt") or ""), str(call.get("user_prompt") or "")

    parent_path = reconstruct_path(str(node.get("parent_id")) if node.get("parent_id") is not None else None, nodes_by_id)
    memory = build_memory_from_path(parent_path)
    base_sample = build_base_sample(result=result, view=view, images=images)

    if actor == "main":
        node_sample = build_node_sample(
            sample=base_sample,
            parent_path=parent_path,
            node=node,
            orchestra_model=str(result.get("orchestra_model") or run_config.get("orchestra_model") or ""),
        )
        step_index = len(memory.attempts) + 1
        config_max_steps = prompt_config.get("max_steps")
        max_steps = int(config_max_steps or run_config.get("node_max_steps") or result.get("path_max_steps") or max(1, step_index))
        force_submit = step_index == max_steps and bool(memory.attempts)
        config_sub_models = prompt_config.get("sub_models")
        if isinstance(config_sub_models, list) and config_sub_models:
            model_pool = [str(model) for model in config_sub_models]
        else:
            model_pool = [str(model) for model in list(node.get("model_pool", []) or [])]
        allow_self_delegation = bool(
            prompt_config.get("allow_main_model_delegation", run_config.get("allow_orchestra_model_delegation", False))
        )
        user_prompt = build_main_prompt(
            sample=node_sample,
            step_history_text=memory.as_brief_text(),
            step_index=step_index,
            max_steps=max_steps,
            sub_models=model_pool,
            allow_self_delegation=allow_self_delegation,
            force_submit=force_submit,
        )
        if main_template_path is not None:
            user_prompt = render_md_template(
                load_text(main_template_path),
                build_main_template_values(
                    base_sample=base_sample,
                    node_sample=node_sample,
                    memory=memory,
                    node=node,
                    run_config=run_config,
                    result=result,
                    step_index=step_index,
                    max_steps=max_steps,
                    force_submit=force_submit,
                    model_pool=model_pool,
                ),
            )
        return "You are a strict orchestration controller. Output JSON only.", user_prompt

    instruction = str(node.get("focus_question") or node.get("instruction") or "")
    user_prompt = build_sub_prompt(
        question=base_sample.question,
        options=base_sample.options,
        instruction=instruction,
        prior_attempts=memory.attempts,
    )
    if sub_template_path is not None:
        user_prompt = render_md_template(
            load_text(sub_template_path),
            build_sub_template_values(
                base_sample=base_sample,
                memory=memory,
                node=node,
            ),
        )
    return "You are a rigorous multimodal scientific reasoning assistant.", user_prompt


def compute_descendant_reward_stats(
    *,
    node_id: str,
    nodes_by_id: dict[str, JsonDict],
    terminal_scope: str,
    empty_reward: str,
    memo: dict[str, RewardStats],
) -> RewardStats:
    cached = memo.get(node_id)
    if cached is not None:
        return cached

    node = nodes_by_id[node_id]
    children = [nodes_by_id[child_id] for child_id in node.get("children_ids", []) if child_id in nodes_by_id]

    if not children:
        if bool(node.get("is_terminal")):
            is_submit = node.get("action") == "submit" and bool(node.get("boxed_letter"))
            terminal_count = 1 if terminal_scope == "all_terminal" or is_submit else 0
            submit_count = 1 if is_submit else 0
            correct_count = 1 if is_submit and bool(node.get("is_correct")) else 0
            reward = (
                correct_count / terminal_count
                if terminal_count > 0
                else (math.nan if empty_reward == "nan" else 0.0)
            )
            stats = RewardStats(
                reward=reward,
                terminal_count=terminal_count,
                submit_count=submit_count,
                correct_count=correct_count,
                has_open_descendant=False,
            )
            memo[node_id] = stats
            return stats
        stats = RewardStats(
            reward=(math.nan if empty_reward == "nan" else 0.0),
            terminal_count=0,
            submit_count=0,
            correct_count=0,
            has_open_descendant=True,
        )
        memo[node_id] = stats
        return stats

    terminal_count = 0
    submit_count = 0
    correct_count = 0
    has_open_descendant = False
    for child in children:
        child_stats = compute_descendant_reward_stats(
            node_id=str(child["node_id"]),
            nodes_by_id=nodes_by_id,
            terminal_scope=terminal_scope,
            empty_reward=empty_reward,
            memo=memo,
        )
        terminal_count += child_stats.terminal_count
        submit_count += child_stats.submit_count
        correct_count += child_stats.correct_count
        has_open_descendant = has_open_descendant or child_stats.has_open_descendant

    denominator = terminal_count if terminal_scope == "all_terminal" else submit_count
    reward = correct_count / denominator if denominator > 0 else (math.nan if empty_reward == "nan" else 0.0)
    stats = RewardStats(
        reward=reward,
        terminal_count=terminal_count,
        submit_count=submit_count,
        correct_count=correct_count,
        has_open_descendant=has_open_descendant,
    )
    memo[node_id] = stats
    return stats


def select_calls(calls: list[JsonDict], actor: str) -> list[JsonDict]:
    if actor == "all":
        return list(calls)
    return [call for call in calls if str(call.get("actor")) == actor]


def include_self_delegate_calls(selected_calls: list[JsonDict], all_calls: list[JsonDict]) -> list[JsonDict]:
    """Append delegate calls with model=self, preserving original call order."""
    selected_call_ids = {str(call.get("call_id") or "") for call in selected_calls}
    merged = list(selected_calls)
    for call in all_calls:
        if str(call.get("actor") or "") != "delegate":
            continue
        if str(call.get("model") or "") != "self":
            continue
        call_id = str(call.get("call_id") or "")
        if call_id and call_id in selected_call_ids:
            continue
        merged.append(call)
        if call_id:
            selected_call_ids.add(call_id)
    merged.sort(key=lambda item: str(item.get("call_id") or ""))
    return merged


def build_default_record(
    *,
    export_mode: str,
    record_style: str,
    prompt_source: str,
    call: JsonDict,
    node: JsonDict,
    result: JsonDict,
    view: JsonDict | None,
    latest: JsonDict,
    run_config: JsonDict,
    system_prompt_used: str,
    user_prompt_used: str,
    messages: list[JsonDict],
    images: list[str],
    assistant_content: str,
    reward_stats: RewardStats,
) -> JsonDict:
    minimal_record: JsonDict = {
        "messages": messages,
        "expected_acc_reward": reward_stats.reward,
        "answer": assistant_content,
        "task_id": result.get("task_id"),
        "node_id": node.get("node_id"),
        "parent_id": node.get("parent_id"),
    }
    if images:
        minimal_record["images"] = images
    if record_style == "minimal":
        return minimal_record

    standard_record: JsonDict = {
        **minimal_record,
        "export_mode": export_mode,
        "depth": node.get("depth"),
        "round_index": node.get("round_index"),
        "action": node.get("action"),
        "gold_answer_letter": result.get("gold_answer_letter"),
        "descendant_terminal_count": reward_stats.terminal_count,
        "descendant_submit_count": reward_stats.submit_count,
        "descendant_correct_count": reward_stats.correct_count,
    }
    if view is not None:
        standard_record["reference_steps"] = list(view.get("reference", {}).get("steps", []))
    if export_mode == "sft":
        standard_record["logged_completion"] = assistant_content
    if record_style == "standard":
        return standard_record

    full_record: JsonDict = {
        **standard_record,
        "actor": call.get("actor"),
        "status": node.get("status"),
        "chosen_model": node.get("chosen_model"),
        "orchestra_model": result.get("orchestra_model"),
        "discipline": result.get("discipline"),
        "subtree_has_open_descendant": reward_stats.has_open_descendant,
        "node_is_terminal": bool(node.get("is_terminal")),
        "node_is_correct": bool(node.get("is_correct")),
        "boxed_letter": node.get("boxed_letter"),
        "focus_question": node.get("focus_question"),
        "delegate_answer": node.get("delegate_answer"),
        "delegate_confidence": node.get("delegate_confidence"),
        "submit_reason": node.get("submit_reason"),
        "prompt_source": prompt_source,
        "system_prompt_used": system_prompt_used,
        "user_prompt_used": user_prompt_used,
        "logged_system_prompt": call.get("system_prompt"),
        "logged_user_prompt": call.get("user_prompt"),
        "logged_completion": assistant_content,
        "logged_completion_raw": call.get("raw_text"),
        "logged_thinking": call.get("thinking") or "",
        "logged_completion_full": build_full_completion(call),
        "logged_parsed_json": canonical_json_text(call.get("parsed") or {}),
        "input_tokens": call.get("input_tokens"),
        "output_tokens": call.get("output_tokens"),
        "call_cost": call.get("cost"),
        "sample_success": result.get("success"),
        "sample_any_correct_leaf": result.get("any_correct_leaf"),
        "sample_best_leaf_correct": result.get("best_leaf_correct"),
        "sample_majority_correct": result.get("majority_correct"),
        "sample_correct_leaf_count": result.get("correct_leaf_count"),
        "sample_final_leaf_count": result.get("final_leaf_count"),
        "tree_budget_limit": result.get("budget_limit"),
        "tree_budget_spent": result.get("budget_spent"),
        "tree_stop_reason": result.get("stop_reason"),
        "tree_target_leaf_trajectories": result.get("target_leaf_trajectories"),
        "tree_branching_factor": result.get("branching_factor"),
        "tree_path_max_steps": result.get("path_max_steps"),
    }
    return full_record


def build_messages(
    *,
    system_prompt: str | None,
    user_prompt: str | None,
    assistant_content: str | None,
    images: list[str],
    image_token_mode: str,
) -> list[JsonDict]:
    messages: list[JsonDict] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    user_content = maybe_add_image_tokens(str(user_prompt or ""), len(images), image_token_mode)
    messages.append({"role": "user", "content": user_content})
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def normalize_submit_only_action_line(user_prompt: str) -> str:
    for old_line in [
        "This turn: action MUST be submit. Delegation is unavailable.",
        "This turn: output submit only (delegation disabled).",
        "Last orchestration turn: you must output submit with a letter.",
    ]:
        user_prompt = user_prompt.replace(
            old_line,
            "Default: delegate when a specific factual gap remains. "
            "If the question requires multi-step reasoning (e.g. read a figure then compute), "
            "delegate each step separately \u2014 do NOT reason on your own. "
            "Do NOT delegate just to raise confidence \u2014 submit when all reasoning steps have been answered by delegates.",
        )
    return user_prompt


def iter_sample_dirs(samples_dir: Path, task_ids: set[str] | None = None) -> list[Path]:
    paths: list[Path] = []
    for child in sorted(samples_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        # If this directory has nodes.jsonl, it's a flat sample (GPQA / SGI).
        if (child / "nodes.jsonl").exists():
            paths.append(child)
        else:
            # Otherwise check one level deeper for nested samples (SFE: A011/0016).
            subs = sorted(
                (s for s in child.iterdir() if s.is_dir() and (s / "nodes.jsonl").exists()),
                key=lambda p: p.name,
            )
            paths.extend(subs)
    if task_ids:
        paths = [p for p in paths if sample_dir_to_task_id(p, samples_dir) in task_ids]
    return paths


def sample_dir_to_task_id(sample_dir: Path, samples_dir: Path) -> str:
    """Return the task_id derived from a sample directory path.

    For flat samples (``samples/abc123/``) this is just the directory name.
    For nested SFE samples (``samples/A011/0016/``) it is ``A011/0016``.
    """
    rel = sample_dir.relative_to(samples_dir)
    return str(rel)


def resolve_calls_jsonl_path(sample_dir: Path) -> tuple[list[str], Path | None]:
    """
    Require nodes.jsonl + latest.json. Accept calls.jsonl or calls.partial.jsonl.

    Returns (missing_filenames, calls_path). calls_path is None when no calls file exists.
    """
    missing: list[str] = []
    if not (sample_dir / "nodes.jsonl").exists():
        missing.append("nodes.jsonl")
    if not (sample_dir / "latest.json").exists():
        missing.append("latest.json")
    primary = sample_dir / "calls.jsonl"
    partial = sample_dir / "calls.partial.jsonl"
    if primary.exists():
        return missing, primary
    if partial.exists():
        return missing, partial
    missing.append("calls.jsonl")
    return missing, None


def is_fallback_call(node: JsonDict) -> bool:
    """Return True when the node was produced by a guardrail fallback override."""
    return str(node.get("orchestra_reasoning") or "") in FALLBACK_ORCHESTRA_REASONS


def is_invalid_submit_node(node: JsonDict) -> bool:
    """Return True when a submit terminal has no valid boxed option."""
    if str(node.get("action") or "") != "submit":
        return False
    boxed_letter = str(node.get("boxed_letter") or "").strip()
    return not bool(boxed_letter)


def load_image_counts_from_selected_tasks(run_dir: Path) -> dict[str, int]:
    """Read per-task image counts stored in selected_tasks.json at run time."""
    path = run_dir / "selected_tasks.json"
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        counts: dict[str, int] = {}
        for sample_info in data.get("selected_samples", []):
            tid = str(sample_info.get("task_id", ""))
            count = sample_info.get("image_count")
            if tid and count is not None:
                counts[tid] = int(count)
        return counts
    except Exception:
        return {}


def export_run(
    *,
    input_path: str | Path,
    output_dir: str | Path | None,
    actor: str,
    dataset_type: str,
    record_style: str,
    prompt_source: str,
    main_template_path: Path | None,
    sub_template_path: Path | None,
    prompt_config_path: Path | None,
    completion_mode: str,
    image_mode: str,
    resolve_images: str,
    terminal_scope: str,
    empty_reward: str,
    image_storage_mode: str,
    images_output_dir: str | Path | None,
    image_path_style: str,
    overwrite_images_dir: bool,
    formatter: FormatterFn | None,
    task_ids: set[str] | None,
    filter_fallback: bool = False,
    sft_filter_mode: str = "none",
    sft_min_delegate_findings: int = 2,
    sft_max_correct_paths_per_sample: int = 5,
    include_self_delegate_with_main: bool = False,
    max_completion_chars: int | None = None,
    max_messages_chars: int | None = None,
    drop_invalid_completion_json: bool = True,
    normalize_submit_action_line: bool = False,
    dataset_name_override: str | None = None,
    dataset_split_override: str | None = None,
) -> dict[str, Any]:
    if sft_filter_mode != "none" and actor != "main":
        raise ValueError("Strict SFT filtering currently supports only --actor main.")
    run_dir, samples_dir = resolve_run_and_samples_dir(input_path)
    run_config = load_json(run_dir / "config.snapshot.json") if (run_dir / "config.snapshot.json").exists() else {}
    prompt_config = load_prompt_config(prompt_config_path)

    sample_dirs = iter_sample_dirs(samples_dir, task_ids=task_ids)
    if not sample_dirs:
        raise ValueError(f"No sample directories found under {samples_dir}")

    ds_name_resolved = dataset_name_override or run_config.get("dataset_name")
    ds_split_resolved = dataset_split_override or run_config.get("dataset_split")
    # Text stubs for incomplete samples (no result.json): match image resolution defaults.
    stub_dataset_name = ds_name_resolved if isinstance(ds_name_resolved, str) else "InternScience/SGI-Reasoning"
    stub_dataset_split = ds_split_resolved if isinstance(ds_split_resolved, str) else "test"

    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (run_dir / "msswift_export").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    materialize_state: dict[str, Any] = {
        "images_dir_ready": False,
        "materialized_image_count": 0,
    }

    images_by_task: dict[str, list[str]] = {}
    image_resolution_meta: dict[str, Any] = {"enabled": False, "reason": "disabled"}
    if resolve_images != "never":
        target_task_ids = {sample_dir_to_task_id(sample_dir, samples_dir) for sample_dir in sample_dirs}
        images_by_task, image_resolution_meta = load_task_images(
            task_ids=target_task_ids,
            dataset_name=ds_name_resolved if isinstance(ds_name_resolved, str) else None,
            dataset_split=ds_split_resolved if isinstance(ds_split_resolved, str) else None,
            strict=(resolve_images == "always"),
        )

    # When resolve_images=="never", read image counts from selected_tasks.json so
    # that the correct number of <image> tokens can still be inserted into the
    # prompt without embedding actual image data in the output record.
    image_counts_by_task: dict[str, int] = {}
    if resolve_images == "never":
        image_counts_by_task = load_image_counts_from_selected_tasks(run_dir)
        if image_counts_by_task:
            image_resolution_meta = {
                "enabled": False,
                "reason": "never_embed_use_local_counts",
                "tasks_with_image_count": len(image_counts_by_task),
            }

    export_modes = ["ppo", "sft"] if dataset_type == "both" else [dataset_type]
    rows_by_mode: dict[str, list[JsonDict]] = {mode: [] for mode in export_modes}
    skipped_samples: list[dict[str, Any]] = []
    skipped_records = 0
    skipped_invalid_submit_records = 0
    skipped_overlong_records = 0
    skipped_invalid_completion_json_records = 0
    strict_sft_stats = {
        "enabled": sft_filter_mode != "none",
        "filter_mode": sft_filter_mode,
        "min_delegate_findings": sft_min_delegate_findings,
        "max_correct_paths_per_sample": sft_max_correct_paths_per_sample,
        "samples_with_selected_path": 0,
        "samples_without_selected_path": 0,
        "selected_path_count": 0,
        "kept_delegate_rows": 0,
        "kept_submit_rows": 0,
    }

    stub_task_ids = {sample_dir_to_task_id(sd, samples_dir) for sd in sample_dirs if not (sd / "result.json").exists()}
    dataset_result_stubs = load_dataset_result_stubs(
        task_ids=stub_task_ids,
        dataset_name=stub_dataset_name,
        dataset_split=stub_dataset_split,
    )
    synthetic_result_from_dataset = 0
    synthetic_result_minimal_fallback = 0

    for sample_dir in sample_dirs:
        missing_req, calls_path = resolve_calls_jsonl_path(sample_dir)
        if missing_req or calls_path is None:
            skipped_samples.append({"task_id": sample_dir_to_task_id(sample_dir, samples_dir), "missing_files": missing_req})
            continue

        latest = load_json(sample_dir / "latest.json")
        if (sample_dir / "result.json").exists():
            result = load_json(sample_dir / "result.json")
        else:
            _sd_tid = sample_dir_to_task_id(sample_dir, samples_dir)
            tid = str(latest.get("task_id") or _sd_tid)
            stub = dataset_result_stubs.get(tid) or dataset_result_stubs.get(_sd_tid)
            if stub is not None:
                result = {**stub, "status": "snapshot"}
                synthetic_result_from_dataset += 1
            else:
                result = {
                    "task_id": tid,
                    "question": "",
                    "options": [],
                    "gold_answer_letter": "",
                    "discipline": "unknown",
                    "status": "snapshot",
                }
                synthetic_result_minimal_fallback += 1
        view = load_json(sample_dir / "view.json") if (sample_dir / "view.json").exists() else None
        nodes = load_jsonl(sample_dir / "nodes.jsonl")
        calls = load_jsonl(calls_path)
        nodes_by_id = build_nodes_by_id(nodes)
        selected_calls = select_calls(calls, actor=actor)
        if include_self_delegate_with_main and actor == "main":
            selected_calls = include_self_delegate_calls(selected_calls, calls)
        main_calls = [call for call in calls if str(call.get("actor") or "") == "main"]
        calls_by_node_id = {
            str(call.get("node_id")): call
            for call in main_calls
            if str(call.get("node_id") or "") in nodes_by_id
        }
        strict_sft_selection: StrictSftSelection | None = None
        if sft_filter_mode in {"strict_orchestrator", "all_correct_paths_capped"}:
            strict_sft_selection = build_strict_sft_selection(
                result=result,
                latest=latest,
                nodes_by_id=nodes_by_id,
                calls_by_node_id=calls_by_node_id,
                min_delegate_findings=sft_min_delegate_findings,
                filter_mode=sft_filter_mode,
                max_correct_paths_per_sample=sft_max_correct_paths_per_sample,
            )
            if strict_sft_selection.skip_reason is None:
                strict_sft_stats["samples_with_selected_path"] += 1
                strict_sft_stats["selected_path_count"] += strict_sft_selection.selected_path_count
            else:
                strict_sft_stats["samples_without_selected_path"] += 1
        reward_memo: dict[str, RewardStats] = {}
        task_id_str = str(result.get("task_id") or "")
        images = images_by_task.get(task_id_str, [])
        # For never mode: use placeholder list of correct length so that
        # <image> tokens are inserted into the prompt, but the placeholders
        # are NOT written into the output record.
        if resolve_images == "never" and not images:
            count = image_counts_by_task.get(task_id_str, 0)
            prompt_images = [""] * count
            record_images: list[str] = []
        else:
            prompt_images, record_images = maybe_materialize_images(
                images=images,
                storage_mode=image_storage_mode,
                images_output_dir=(Path(images_output_dir).expanduser().resolve() if images_output_dir else None),
                output_root=output_root,
                image_path_style=image_path_style,
                overwrite_images_dir=overwrite_images_dir,
                state=materialize_state,
            )

        for call in selected_calls:
            node_id = str(call.get("node_id"))
            node = nodes_by_id.get(node_id)
            if node is None:
                skipped_records += 1
                continue

            if filter_fallback and is_fallback_call(node):
                skipped_records += 1
                continue
            if filter_fallback and is_invalid_submit_node(node):
                skipped_records += 1
                skipped_invalid_submit_records += 1
                continue

            reward_stats = compute_descendant_reward_stats(
                node_id=node_id,
                nodes_by_id=nodes_by_id,
                terminal_scope=terminal_scope,
                empty_reward=empty_reward,
                memo=reward_memo,
            )
            normalized_completion = normalize_completion(call, mode=completion_mode)
            if max_completion_chars is not None and max_completion_chars > 0:
                if len(normalized_completion) > max_completion_chars:
                    skipped_records += 1
                    skipped_overlong_records += 1
                    continue
            if drop_invalid_completion_json:
                try:
                    parsed_completion = json.loads(normalized_completion)
                except Exception:
                    skipped_records += 1
                    skipped_invalid_completion_json_records += 1
                    continue
                if not isinstance(parsed_completion, dict):
                    skipped_records += 1
                    skipped_invalid_completion_json_records += 1
                    continue
            rendered_system_prompt, rendered_user_prompt = render_prompts(
                prompt_source=prompt_source,
                main_template_path=main_template_path,
                sub_template_path=sub_template_path,
                prompt_config=prompt_config,
                call=call,
                node=node,
                nodes_by_id=nodes_by_id,
                result=result,
                view=view,
                run_config=run_config,
                images=prompt_images,
            )
            if normalize_submit_action_line:
                rendered_user_prompt = normalize_submit_only_action_line(rendered_user_prompt)

            for export_mode in export_modes:
                if export_mode == "sft" and sft_filter_mode != "none":
                    if not should_keep_call_for_strict_sft(
                        call=call,
                        node=node,
                        selection=strict_sft_selection,
                    ):
                        skipped_records += 1
                        continue
                assistant_content = normalized_completion if export_mode == "sft" else None
                messages = build_messages(
                    system_prompt=rendered_system_prompt,
                    user_prompt=rendered_user_prompt,
                    assistant_content=assistant_content,
                    images=prompt_images,
                    image_token_mode=image_mode,
                )
                context = {
                    "export_mode": export_mode,
                    "prompt_source": prompt_source,
                    "call": call,
                    "node": node,
                    "result": result,
                    "view": view,
                    "latest": latest,
                    "run_config": run_config,
                    "system_prompt_used": rendered_system_prompt,
                    "user_prompt_used": rendered_user_prompt,
                    "messages": messages,
                    "images": record_images,
                    "prompt_images": prompt_images,
                    "assistant_content": normalized_completion,
                    "reward_stats": {
                        "reward": reward_stats.reward,
                        "terminal_count": reward_stats.terminal_count,
                        "submit_count": reward_stats.submit_count,
                        "correct_count": reward_stats.correct_count,
                        "has_open_descendant": reward_stats.has_open_descendant,
                    },
                    "sample_dir": str(sample_dir),
                    "run_dir": str(run_dir),
                }
                if formatter is not None:
                    record = formatter(context)
                else:
                    record = build_default_record(
                        export_mode=export_mode,
                        record_style=record_style,
                        prompt_source=prompt_source,
                        call=call,
                        node=node,
                        result=result,
                        view=view,
                        latest=latest,
                        run_config=run_config,
                        system_prompt_used=rendered_system_prompt,
                        user_prompt_used=rendered_user_prompt,
                        messages=messages,
                        images=record_images,
                        assistant_content=normalized_completion,
                        reward_stats=reward_stats,
                    )
                if record is None:
                    skipped_records += 1
                    continue
                if max_messages_chars is not None and max_messages_chars > 0:
                    total_chars = sum(len(m.get("content", "")) for m in record.get("messages", []))
                    if total_chars > max_messages_chars:
                        skipped_records += 1
                        skipped_overlong_records += 1
                        continue
                rows_by_mode[export_mode].append(record)
                if export_mode == "sft" and sft_filter_mode != "none":
                    action = str(node.get("action") or "")
                    if action == "delegate":
                        strict_sft_stats["kept_delegate_rows"] += 1
                    elif action == "submit":
                        strict_sft_stats["kept_submit_rows"] += 1

    output_files: dict[str, str] = {}
    for mode, rows in rows_by_mode.items():
        file_path = output_root / f"msswift_{mode}.jsonl"
        dump_jsonl(file_path, rows)
        output_files[mode] = str(file_path)

    summary = {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "run_dir": str(run_dir),
        "samples_dir": str(samples_dir),
        "output_dir": str(output_root),
        "output_files": output_files,
        "dataset_type": dataset_type,
        "record_style": record_style,
        "prompt_source": prompt_source,
        "main_template_path": str(main_template_path) if main_template_path is not None else None,
        "sub_template_path": str(sub_template_path) if sub_template_path is not None else None,
        "prompt_config_path": str(prompt_config_path) if prompt_config_path is not None else None,
        "actor": actor,
        "completion_mode": completion_mode,
        "image_mode": image_mode,
        "resolve_images": resolve_images,
        "terminal_scope": terminal_scope,
        "empty_reward": empty_reward,
        "image_storage_mode": image_storage_mode,
        "images_output_dir": str((Path(images_output_dir).expanduser().resolve() if images_output_dir else (output_root / "images").resolve()))
        if image_storage_mode != "inline"
        else None,
        "image_path_style": image_path_style,
        "filter_fallback": filter_fallback,
        "include_self_delegate_with_main": include_self_delegate_with_main,
        "max_completion_chars": max_completion_chars,
        "max_messages_chars": max_messages_chars,
        "drop_invalid_completion_json": drop_invalid_completion_json,
        "normalize_submit_action_line": normalize_submit_action_line,
        "sft_filter_mode": sft_filter_mode,
        "sft_min_delegate_findings": sft_min_delegate_findings,
        "sft_max_correct_paths_per_sample": sft_max_correct_paths_per_sample,
        "sample_dir_count": len(sample_dirs),
        "skipped_samples": skipped_samples,
        "synthetic_result_from_dataset_count": synthetic_result_from_dataset,
        "synthetic_result_minimal_fallback_count": synthetic_result_minimal_fallback,
        "skipped_record_count": skipped_records,
        "skipped_overlong_record_count": skipped_overlong_records,
        "skipped_invalid_completion_json_record_count": skipped_invalid_completion_json_records,
        "rows_per_mode": {mode: len(rows) for mode, rows in rows_by_mode.items()},
        "skipped_invalid_submit_record_count": skipped_invalid_submit_records,
        "strict_sft_stats": strict_sft_stats,
        "image_resolution": image_resolution_meta,
        "notes": {
            "ms_swift_messages_required": True,
            "ppo_prompt_only": "ppo export ends with the user turn only; assistant targets are omitted",
            "reward_definition": (
                "expected_acc_reward = correct_terminal_descendant_count / descendant_terminal_count "
                "or / descendant_submit_count depending on terminal_scope"
            ),
            "full_completion_mode": "completion_mode=full rebuilds <think>...</think> plus the final assistant content from logged thinking/raw_text",
            "prompt_source": "logged uses stored prompts; template reconstructs prompts from fixed repo templates plus node state",
            "prompt_config": "Pass --prompt-config to override prompt reconstruction with a current runtime config (for example configs/reasoning.yaml), including sub_models and max_steps.",
            "resolve_images_never": (
                "resolve_images=never reads image_count from selected_tasks.json to add <image> tokens "
                "to the prompt without fetching images from HuggingFace or embedding them in the record; "
                "image_count is written by the runner when it saves selected_tasks.json"
            ),
            "image_storage_mode": (
                "inline keeps image payloads in JSON records; file_paths writes data-url images to files and stores file paths; "
                "detached writes files but omits images from JSON records while preserving <image> token insertion."
            ),
            "external_md_templates": "Pass --main-template-md and/or --sub-template-md to override reconstructed prompts with Markdown templates and placeholder filling",
            "formatter_hook": "Pass --formatter path/to/file.py[:func_name] to override each exported record",
            "filter_fallback": (
                "When --filter-fallback is set, calls where the node's orchestra_reasoning matches a known "
                "guardrail fallback phrase are skipped; submit nodes with empty boxed_letter are also skipped. "
                "Both are counted in skipped_record_count."
            ),
            "strict_sft_filter": (
                "sft_filter_mode=strict_orchestrator keeps one best correct terminal path per sample; "
                "sft_filter_mode=all_correct_paths_capped unions up to sft_max_correct_paths_per_sample correct paths "
                "per sample. Both modes retain legal delegate_task nodes on selected correct paths and keep correct "
                "submit nodes only when they follow at least sft_min_delegate_findings informative delegates."
            ),
            "incomplete_run_export": (
                "Samples without result.json still export when nodes.jsonl, latest.json, and calls.jsonl or "
                "calls.partial.jsonl exist. Missing result.json fields are filled from the Hugging Face dataset "
                "(dataset_name/split from the run config, or InternScience/SGI-Reasoning / test when absent); "
                "if a task row is still not found, a minimal stub is used."
            ),
        },
    }
    dump_json(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export SciOrch MCTS v2 runs into ms-swift-compatible JSONL datasets. "
            "Default reward is the expected correctness over all terminal descendants of each node."
        )
    )
    parser.add_argument("input_path", help="Path to an MCTS run dir or its samples dir.")
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to <run_dir>/msswift_export.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["ppo", "sft", "both"],
        default="ppo",
        help="Which ms-swift dataset shape to export.",
    )
    parser.add_argument(
        "--record-style",
        choices=["minimal", "standard", "full"],
        default="minimal",
        help="How many extra metadata fields to keep in each exported record.",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["template", "logged"],
        default="template",
        help="Use fixed repo templates plus reconstructed state, or reuse the exact logged prompts.",
    )
    parser.add_argument(
        "--main-template-md",
        help=(
            "Optional Markdown file to override the reconstructed main-agent user prompt. "
            "If omitted in --prompt-source template mode, uses sciorch.prompts.main_reasoning.build_main_prompt. "
            "Supported placeholders include {{question}}, {{question_with_branch_context}}, "
            "{{options_text}}, {{step_history_text}}, {{step_index}}, {{max_steps}}, "
            "{{remaining_steps}}, {{action_line}}, {{urgency}}, {{urgency_suffix}}, "
            "{{model_guidance_text}}, "
            "{{model_pool_json}}, {{task_id}}, {{node_id}}, {{round_index}}."
        ),
    )
    parser.add_argument(
        "--prompt-config",
        help=(
            "Optional runtime config YAML (for example configs/reasoning.yaml). "
            "When set with --prompt-source template, uses this config's sub_models/max_steps/"
            "allow_main_model_delegation to reconstruct prompts."
        ),
    )
    parser.add_argument(
        "--sub-template-md",
        help=(
            "Optional Markdown file to override the reconstructed sub-agent user prompt. "
            "Supported placeholders include {{question}}, {{instruction}}, {{options_text}}, "
            "{{prior_steps_text}}, {{task_id}}, {{node_id}}, {{round_index}}."
        ),
    )
    parser.add_argument(
        "--actor",
        choices=["main", "delegate", "all"],
        default="main",
        help="Which call actor(s) to export.",
    )
    parser.add_argument(
        "--completion-mode",
        choices=["auto", "raw", "parsed", "full"],
        default="auto",
        help=(
            "How to serialize the logged assistant completion. "
            "'auto' prefers <think>...</think> plus final content when thinking is stored; "
            "'full' forces that reconstruction."
        ),
    )
    parser.add_argument(
        "--image-mode",
        choices=["prefix", "none"],
        default="prefix",
        help="How to inject <image> tokens into the user prompt when images are resolved.",
    )
    parser.add_argument(
        "--resolve-images",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "How to handle images. "
            "'auto': load from HuggingFace dataset and embed in output record. "
            "'always': same as auto but fail hard if images cannot be resolved. "
            "'never': do NOT fetch from HuggingFace and do NOT embed images in the output record. "
            "Instead, read the per-task image_count stored in selected_tasks.json (written by the "
            "runner) and use it to insert the correct number of <image> tokens into the prompt. "
            "This lets you produce correctly-shaped prompts offline without large image blobs in "
            "the JSONL file. Requires the run to have been done with a runner version that saves "
            "image_count in selected_tasks.json."
        ),
    )
    parser.add_argument(
        "--image-storage-mode",
        choices=["inline", "file_paths", "detached"],
        default="inline",
        help=(
            "How resolved images are written in exported JSON records. "
            "'inline' keeps original image values (default). "
            "'file_paths' writes data-url images to files and stores image paths in JSON. "
            "'detached' writes files but omits the images field from JSON records."
        ),
    )
    parser.add_argument(
        "--images-output-dir",
        help="Directory for image files when --image-storage-mode is file_paths or detached. Defaults to <output-dir>/images.",
    )
    parser.add_argument(
        "--image-path-style",
        choices=["relative", "absolute"],
        default="relative",
        help="Path style used when --image-storage-mode=file_paths.",
    )
    parser.add_argument(
        "--overwrite-images-dir",
        action="store_true",
        default=False,
        help="Overwrite --images-output-dir if it already exists.",
    )
    parser.add_argument(
        "--terminal-scope",
        choices=["all_terminal", "submit_only"],
        default="all_terminal",
        help=(
            "Whether reward denominator counts all terminal descendants "
            "(failed terminals count as 0) or only submit leaves."
        ),
    )
    parser.add_argument(
        "--empty-reward",
        choices=["zero", "nan"],
        default="zero",
        help="Reward to emit when a node has no descendant terminal estimate in the stored tree.",
    )
    parser.add_argument(
        "--formatter",
        help=(
            "Optional Python formatter hook in the form path/to/file.py[:func_name]. "
            "The function receives a context dict and should return one record dict or None."
        ),
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Optional task id filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Override Hugging Face dataset id for resolving images (e.g. InternScience/SGI-Reasoning). "
            "Defaults to dataset_name in the run dir's config.snapshot.json."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help=(
            "Override dataset split when resolving images (e.g. test). "
            "Defaults to dataset_split in config.snapshot.json."
        ),
    )
    parser.add_argument(
        "--filter-fallback",
        action="store_true",
        default=False,
        help=(
            "Skip records where the node was produced by a guardrail fallback override "
            "(e.g. malformed action, unparseable final answer). "
            "These nodes have boilerplate orchestra_reasoning and often empty raw_text."
        ),
    )
    parser.add_argument(
        "--include-self-delegate-with-main",
        action="store_true",
        default=False,
        help=(
            "When --actor main is used, also export delegate calls where model is exactly 'self'. "
            "Reward computation remains unchanged (same descendant expected accuracy as other rows)."
        ),
    )
    parser.add_argument(
        "--max-completion-chars",
        type=int,
        default=None,
        help=(
            "Optional completion-length filter. Records whose normalized assistant completion length "
            "exceeds this threshold are dropped."
        ),
    )
    parser.add_argument(
        "--max-messages-chars",
        type=int,
        default=None,
        help=(
            "Optional total-messages-length filter. Records whose combined messages character count "
            "exceeds this threshold are dropped."
        ),
    )
    parser.add_argument(
        "--drop-invalid-completion-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to drop records whose normalized completion is not a single valid JSON object "
            "(e.g. concatenated retry payloads that fail json parsing)."
        ),
    )
    parser.add_argument(
        "--normalize-submit-action-line",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, normalize the prompt line "
            "'This turn: action MUST be submit. Delegation is unavailable.' "
            "to 'This turn: choose delegate_task or submit.' in exported user prompts."
        ),
    )
    parser.add_argument(
        "--sft-filter-mode",
        choices=["none", "strict_orchestrator", "all_correct_paths_capped"],
        default="none",
        help=(
            "Optional SFT-only filtering policy. "
            "'strict_orchestrator' keeps only one best correct terminal path per sample, "
            "'all_correct_paths_capped' keeps multiple correct terminal paths per sample up to a cap, "
            "and both modes drop obvious policy violations such as fallback delegates and early submit nodes."
        ),
    )
    parser.add_argument(
        "--sft-min-delegate-findings",
        type=int,
        default=2,
        help="Minimum number of informative delegate findings required before keeping a submit node in strict SFT mode.",
    )
    parser.add_argument(
        "--sft-max-correct-paths-per-sample",
        type=int,
        default=5,
        help="Maximum number of correct terminal paths to keep per sample when --sft-filter-mode=all_correct_paths_capped.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    formatter = parse_formatter(args.formatter)
    main_template_path = parse_template_path(args.main_template_md)
    sub_template_path = parse_template_path(args.sub_template_md)
    prompt_config_path = parse_prompt_config_path(args.prompt_config)
    summary = export_run(
        input_path=args.input_path,
        output_dir=args.output_dir,
        actor=args.actor,
        dataset_type=args.dataset_type,
        record_style=args.record_style,
        prompt_source=args.prompt_source,
        main_template_path=main_template_path,
        sub_template_path=sub_template_path,
        prompt_config_path=prompt_config_path,
        completion_mode=args.completion_mode,
        image_mode=args.image_mode,
        resolve_images=args.resolve_images,
        terminal_scope=args.terminal_scope,
        empty_reward=args.empty_reward,
        image_storage_mode=args.image_storage_mode,
        images_output_dir=args.images_output_dir,
        image_path_style=args.image_path_style,
        overwrite_images_dir=args.overwrite_images_dir,
        formatter=formatter,
        task_ids=set(args.task_ids or []),
        filter_fallback=args.filter_fallback,
        sft_filter_mode=args.sft_filter_mode,
        sft_min_delegate_findings=args.sft_min_delegate_findings,
        sft_max_correct_paths_per_sample=args.sft_max_correct_paths_per_sample,
        include_self_delegate_with_main=args.include_self_delegate_with_main,
        max_completion_chars=args.max_completion_chars,
        max_messages_chars=getattr(args, "max_messages_chars", None),
        drop_invalid_completion_json=args.drop_invalid_completion_json,
        normalize_submit_action_line=args.normalize_submit_action_line,
        dataset_name_override=args.dataset_name,
        dataset_split_override=args.dataset_split,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
