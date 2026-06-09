#!/usr/bin/env python3
"""
Train-generate loop orchestrator.

Each iteration:
  1. Start vLLM (tmux session) with the model from the previous iteration.
  2. Run MCTS data generation  (maso conda env).
  3. Kill vLLM and wait for VRAM to drain.
  4. Run offline-REINFORCE training  (swift conda env). From iteration 2 onward,
     the first training run resumes trainer/optimizer state from the previous
     iteration's latest checkpoint (see train.resume_from_previous_iteration).
  5. Apply tokenizer_config.json fix on the saved checkpoint.
  6. Write done.marker + latest_checkpoint.txt.

Usage:
  cd <repo root>
  python scripts/train_loop.py --config configs/train_loop/loop.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[loop {ts}] {msg}", flush=True)


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return raw


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Directory / marker helpers
# ---------------------------------------------------------------------------

def iter_dir(cfg: dict, n: int) -> Path:
    root = Path(cfg["loop"]["output_root"])
    version = cfg["loop"]["version"]
    prefix = cfg["loop"]["iteration_prefix"]
    return root / version / f"{prefix}{n}"


def data_done(idir: Path) -> bool:
    """True when the msswift JSONL export exists and is non-empty."""
    jsonl = idir / "msswift_export" / "msswift_ppo.jsonl"
    return jsonl.exists() and jsonl.stat().st_size > 0


def prune_done(idir: Path) -> bool:
    """True when prune output exists and is non-empty."""
    pruned_dir = idir / "pruned"
    return (pruned_dir / "samples").is_dir() or (pruned_dir / "scored.json").exists()


def train_done(idir: Path) -> bool:
    if (idir / "training" / "done.marker").exists():
        return True
    ckpt = find_latest_partial_ckpt(idir / "training")
    if ckpt is not None:
        write_train_markers(idir / "training", ckpt)
        return True
    return False


def find_latest_partial_ckpt(train_dir: Path) -> Path | None:
    """Return the checkpoint dir with the highest step number, or None.

    Supports both:
    - <train_dir>/checkpoint-*
    - <train_dir>/<run_name>/checkpoint-*
    """
    if not train_dir.exists():
        return None
    candidates: list[Path] = []
    for d in train_dir.rglob("checkpoint-*"):
        if not d.is_dir():
            continue
        suffix = d.name.split("-", 1)[1] if "-" in d.name else ""
        if suffix.isdigit() and (d / "trainer_state.json").exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda d: (int(d.name.split("-", 1)[1]), d.stat().st_mtime),
    )


def latest_checkpoint(train_dir: Path) -> Path | None:
    """Return checkpoint recorded in latest_checkpoint.txt, or detect it."""
    marker = train_dir / "latest_checkpoint.txt"
    if marker.exists():
        p = Path(marker.read_text().strip())
        if p.exists():
            return p
    return find_latest_partial_ckpt(train_dir)


def write_train_markers(train_dir: Path, ckpt: Path) -> None:
    (train_dir / "latest_checkpoint.txt").write_text(str(ckpt), encoding="utf-8")
    (train_dir / "done.marker").write_text("done\n", encoding="utf-8")
    log(f"  wrote done.marker + latest_checkpoint.txt -> {ckpt}")


def pick_serving_model(cfg: dict, n: int, *, dry_run: bool = False) -> str:
    """Return absolute model path: base_model for iter 1, prev ckpt otherwise."""
    if n == 1:
        return cfg["loop"]["base_model"]
    prev_train_dir = iter_dir(cfg, n - 1) / "training"
    ckpt = latest_checkpoint(prev_train_dir)
    if ckpt is None:
        if dry_run:
            placeholder = f"<checkpoint-from-iteration-{n-1}>"
            log(f"  [dry-run] No checkpoint found in {prev_train_dir}; using placeholder: {placeholder}")
            return placeholder
        raise RuntimeError(
            f"Cannot start iteration {n}: no checkpoint found in {prev_train_dir}. "
            "Ensure iteration " + str(n - 1) + " training completed successfully."
        )
    return str(ckpt)


# ---------------------------------------------------------------------------
# vLLM lifecycle (dedicated tmux session, spawned as child of current terminal)
# ---------------------------------------------------------------------------

def _vllm_base_url(cfg: dict) -> str:
    vcfg = cfg["vllm"]
    probe_host = vcfg.get("probe_host", "127.0.0.1")
    return f"http://{probe_host}:{vcfg['port']}"


def _vllm_probe_paths(cfg: dict) -> list[str]:
    paths = cfg["vllm"].get("probe_paths", ["/health", "/v1/models"])
    if not isinstance(paths, list) or not paths:
        return ["/health", "/v1/models"]
    out: list[str] = []
    for p in paths:
        s = str(p).strip()
        if not s:
            continue
        if not s.startswith("/"):
            s = "/" + s
        out.append(s)
    return out or ["/health", "/v1/models"]


def vllm_alive(cfg: dict, *, timeout_s: float | None = None) -> bool:
    vcfg = cfg["vllm"]
    probe_timeout = float(timeout_s if timeout_s is not None else vcfg.get("probe_timeout_s", 10))
    base = _vllm_base_url(cfg)
    for path in _vllm_probe_paths(cfg):
        try:
            with urllib.request.urlopen(f"{base}{path}", timeout=probe_timeout) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def _tmux_session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    )
    return result.returncode == 0


def ensure_vllm(cfg: dict, model_path: str, *, dry_run: bool) -> None:
    vcfg = cfg["vllm"]
    session = vcfg["tmux_session"]

    if vllm_alive(cfg):
        log(f"vLLM already alive at port {vcfg['port']}, skipping start.")
        return

    if _tmux_session_exists(session):
        log(f"Stale tmux session '{session}' found (vLLM not responding). Killing it first.")
        stop_vllm(cfg, dry_run=dry_run)

    serve_cmd = (
        f"source /root/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {vcfg['conda_env']} && "
        f"CUDA_VISIBLE_DEVICES={vcfg['cuda_visible_devices']} "
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model_path} "
        f"--host {vcfg['host']} "
        f"--port {vcfg['port']} "
        f"--served-model-name {vcfg['served_model_name']} "
        f"--tensor-parallel-size {vcfg['tensor_parallel_size']} "
        f"--max-model-len {vcfg['max_model_len']} "
        f"--gpu-memory-utilization {vcfg['gpu_memory_utilization']} "
        f"--dtype auto --trust-remote-code"
    )

    log(f"Starting vLLM in new tmux session '{session}' with model: {model_path}")
    if dry_run:
        log(f"  [dry-run] tmux new-session -d -s {session} bash -lc '<vllm cmd>'")
        return

    # Spawn a detached tmux session; it inherits the GPU lease from this terminal.
    subprocess.check_call([
        "tmux", "new-session", "-d", "-s", session, "bash", "-lc", serve_cmd,
    ])

    poll_interval = float(vcfg.get("ready_poll_interval_s", 5))
    deadline_probe_timeout = float(vcfg.get("deadline_probe_timeout_s", 20))
    log(f"  Waiting for vLLM to become ready (timeout {vcfg['ready_timeout_s']}s)...")
    deadline = time.time() + vcfg["ready_timeout_s"]
    while time.time() < deadline:
        if vllm_alive(cfg):
            log("  vLLM is ready.")
            return
        if not _tmux_session_exists(session):
            raise RuntimeError(
                f"vLLM tmux session '{session}' exited before ready. "
                "Inspect startup logs in that session."
            )
        time.sleep(poll_interval)

    # One final probe with a looser HTTP timeout to reduce false negatives
    # when the first successful response is slow.
    if vllm_alive(cfg, timeout_s=deadline_probe_timeout):
        log("  vLLM became ready on final probe.")
        return
    raise TimeoutError(
        f"vLLM did not become ready within {vcfg['ready_timeout_s']}s. "
        f"Check tmux session '{session}' for errors, or increase "
        "'vllm.ready_timeout_s' / 'vllm.probe_timeout_s'."
    )


def stop_vllm(cfg: dict, *, dry_run: bool) -> None:
    vcfg = cfg["vllm"]
    session = vcfg["tmux_session"]

    if not _tmux_session_exists(session):
        log(f"vLLM session '{session}' not found, nothing to stop.")
        return

    log(f"Stopping vLLM (killing tmux session '{session}')...")
    if not dry_run:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)

    grace = vcfg.get("shutdown_grace_s", 60)
    log(f"  Waiting up to {grace}s for VRAM to drain...")
    if dry_run:
        log("  [dry-run] skip VRAM wait")
        return

    deadline = time.time() + grace
    while time.time() < deadline:
        time.sleep(5)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            used_mb = [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip().isdigit()]
            log(f"    VRAM used: {used_mb} MiB")
            if used_mb and max(used_mb) < 2000:
                log("  VRAM drained.")
                return
    log("  Warning: VRAM may not have fully drained; proceeding anyway.")


class VllmWatchdog:
    """Background thread that monitors vLLM health and restarts it if it dies.

    Usage::

        with VllmWatchdog(cfg, model_path, check_interval=30):
            run_generate(...)
    """

    def __init__(
        self,
        cfg: dict,
        model_path: str,
        *,
        check_interval: float = 30,
        max_consecutive_failures: int = 3,
    ):
        self._cfg = cfg
        self._model_path = model_path
        self._interval = check_interval
        self._max_failures = max_consecutive_failures
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VllmWatchdog":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="vllm-watchdog")
        self._thread.start()
        log(f"vLLM watchdog started (check every {self._interval}s)")
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        log("vLLM watchdog stopped.")

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if self._stop.is_set():
                break
            if vllm_alive(self._cfg, timeout_s=10):
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            log(
                f"  [watchdog] vLLM health check failed "
                f"({consecutive_failures}/{self._max_failures})"
            )
            if consecutive_failures >= self._max_failures:
                log("  [watchdog] vLLM appears down, restarting...")
                try:
                    ensure_vllm(self._cfg, self._model_path, dry_run=False)
                    log("  [watchdog] vLLM restarted successfully.")
                except Exception as e:
                    log(f"  [watchdog] Failed to restart vLLM: {e}")
                consecutive_failures = 0


# ---------------------------------------------------------------------------
# MCTS config rendering
# ---------------------------------------------------------------------------

def render_mcts_yaml(
    cfg: dict,
    idir: Path,
    n: int,
    *,
    dry_run: bool,
    force_export_resume: bool = False,
) -> Path:
    template_path = Path(cfg["generate"]["mcts_config_template"])
    template = load_yaml(template_path)

    out = copy.deepcopy(template)
    vcfg = cfg["vllm"]
    loop_cfg = cfg["loop"]

    # Run overrides
    out.setdefault("run", {})
    out["run"]["iteration_name"] = f"{loop_cfg['iteration_prefix']}{n}"
    out["run"]["resume_iteration"] = f"{loop_cfg['version']}/{loop_cfg['iteration_prefix']}{n}"

    # Reasoning overrides: point to our vLLM instance
    out.setdefault("reasoning", {})
    out["reasoning"]["orchestra_local_base_url"] = f"http://127.0.0.1:{vcfg['port']}/v1"
    out["reasoning"]["orchestra_model"] = vcfg["served_model_name"]

    # Override task_ids from fold_splits.json: iteration N uses fold_N.
    fold_splits_path = cfg["generate"].get("fold_splits")
    if fold_splits_path:
        fold_splits_path = Path(fold_splits_path)
        if fold_splits_path.exists():
            with open(fold_splits_path) as _f:
                fold_splits = json.load(_f)
            num_folds = len(fold_splits)
            fold_idx = ((n - 1) % num_folds) + 1
            fold_key = f"fold_{fold_idx}"
            if fold_key in fold_splits:
                out.setdefault("reasoning", {})
                out["reasoning"]["task_ids"] = fold_splits[fold_key]
                log(f"  Using {fold_key} task_ids ({len(fold_splits[fold_key])} tasks) from {fold_splits_path}")
            else:
                log(f"  WARNING: {fold_key} not found in {fold_splits_path}, keeping template task_ids")
        else:
            log(f"  WARNING: fold_splits file not found: {fold_splits_path}")

    # Default: auto resume so generate_mcts.py can pick up partials.
    # Special case: if prune is already done but export artifact is missing,
    # force start_from=export to avoid the auto-resume "already finished" skip.
    out.setdefault("steps", {})
    out["steps"]["start_from"] = "export" if force_export_resume else "auto"

    dest = idir / "_loop" / f"mcts_generate.iter{n}.yaml"
    if not dry_run:
        write_yaml(dest, out)
        log(f"  Wrote per-iteration mcts config: {dest}")
    else:
        log(f"  [dry-run] Would write per-iteration mcts config: {dest}")

    return dest


# ---------------------------------------------------------------------------
# Generate step
# ---------------------------------------------------------------------------

def run_generate(
    cfg: dict,
    idir: Path,
    n: int,
    *,
    dry_run: bool,
    force_export_resume: bool = False,
) -> None:
    yaml_path = render_mcts_yaml(
        cfg,
        idir,
        n,
        dry_run=dry_run,
        force_export_resume=force_export_resume,
    )
    gcfg = cfg["generate"]

    cmd_str = (
        f"source /root/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {gcfg['conda_env']} && "
        f"cd {gcfg['cwd']} && "
        f"python scripts/generate_mcts.py --config {yaml_path}"
    )

    log(f"Running data generation (env={gcfg['conda_env']})...")
    log(f"  {cmd_str}")
    if dry_run:
        log("  [dry-run] skipping execution")
        return

    subprocess.check_call(["bash", "-lc", cmd_str])
    log("Data generation finished.")


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

def run_train(cfg: dict, idir: Path, n: int, model_path: str, *, dry_run: bool) -> None:
    tcfg = cfg["train"]
    train_out = idir / "training"
    dataset = idir / "msswift_export" / "msswift_ppo.jsonl"

    # Build env-var string from config overrides
    env_parts: list[str] = []
    for k, v in (tcfg.get("env_overrides") or {}).items():
        env_parts.append(f"{k}={shlex.quote(str(v))}")
    override_keys = {str(k) for k in (tcfg.get("env_overrides") or {}).keys()}

    # Cross-iteration resume: use MAX_STEPS for precise control.
    # NUM_EPOCHS is a total target in HuggingFace, but the epoch field in the old
    # checkpoint maps to a different step count when the new dataset has a different
    # number of steps per epoch. Using MAX_STEPS avoids this mismatch entirely.
    if "NUM_EPOCHS" in override_keys and n > 1:
        import json as _json
        import math
        per_iter_epochs = int((tcfg.get("env_overrides") or {})["NUM_EPOCHS"])
        ckpt_path = Path(model_path)
        ts_path = ckpt_path / "trainer_state.json"
        global_step = 0
        if ts_path.exists():
            ts = _json.loads(ts_path.read_text())
            global_step = int(ts.get("global_step", 0))
            old_epoch = float(ts.get("epoch", 0.0))
            log(f"  Checkpoint: global_step={global_step}, epoch={old_epoch:.2f}")

        # Compute steps_per_epoch for the NEW dataset.
        # steps_per_epoch = ceil(dataset_lines / effective_batch_size)
        # effective_batch_size = per_device_batch * grad_accum * num_gpus
        #   (Note: Swift does NOT auto-double grad_accum for DDP — GRAD_ACCUM in
        #    the config already accounts for the multi-GPU setup.)
        dataset_lines = 0
        if dataset.exists():
            with open(dataset) as _f:
                dataset_lines = sum(1 for _ in _f)
        if dataset_lines == 0:
            log(f"  WARNING: dataset {dataset} is empty or missing, falling back to NUM_EPOCHS")
        else:
            overrides = tcfg.get("env_overrides") or {}
            batch_size = int(overrides.get("BATCH_SIZE", 1))
            grad_accum = int(overrides.get("GRAD_ACCUM", 64))
            num_gpus = len(tcfg.get("cuda_visible_devices", "0").split(","))
            effective_batch = batch_size * grad_accum * num_gpus
            new_steps_per_epoch = math.ceil(dataset_lines / effective_batch)
            additional_steps = per_iter_epochs * new_steps_per_epoch
            max_steps = global_step + additional_steps
            log(f"  New dataset: {dataset_lines} rows, effective_batch={effective_batch}, "
                f"steps_per_epoch={new_steps_per_epoch}")
            log(f"  MAX_STEPS={max_steps} (global_step {global_step} + {per_iter_epochs} epochs * "
                f"{new_steps_per_epoch} steps/epoch = +{additional_steps} steps)")
            # Use MAX_STEPS instead of NUM_EPOCHS — HF Trainer treats max_steps > 0
            # as taking priority over num_train_epochs.
            env_parts.append(f"MAX_STEPS={max_steps}")

    # Mandatory overrides that change every iteration
    env_parts.append(f"MODEL={shlex.quote(model_path)}")
    env_parts.append(f"DATASET={shlex.quote(str(dataset))}")
    env_parts.append(f"OUTPUT_DIR={shlex.quote(str(train_out))}")
    env_parts.append(f"CUDA_VISIBLE_DEVICES={shlex.quote(tcfg['cuda_visible_devices'])}")

    # W&B policy: one loop version => one run id (e.g. offline_loop_v0, offline_loop_v1).
    # This allows all iterations under the same version to append into one run.
    run_version = str(cfg["loop"]["version"])
    run_id_prefix = str(tcfg.get("wandb_run_id_prefix", "offline_loop"))
    wandb_run_id = f"{run_id_prefix}_{run_version}"
    if "WANDB_RUN_ID" not in override_keys:
        env_parts.append(f"WANDB_RUN_ID={shlex.quote(wandb_run_id)}")
    if "WANDB_RESUME" not in override_keys:
        env_parts.append("WANDB_RESUME=allow")
    # Keep one readable run name per version unless user explicitly overrides it.
    if "WANDB_RUN_NAME" not in override_keys:
        env_parts.append(f"WANDB_RUN_NAME={shlex.quote(run_version)}")

    # Resume priority:
    # 1) Latest checkpoint under *this* iteration's OUTPUT_DIR (crash mid-run).
    # 2) Else, iteration>=2: resume trainer/optimizer state from previous iteration's
    #    latest checkpoint (requires SAVE_ONLY_MODEL false on prior run).
    partial_ckpt = find_latest_partial_ckpt(train_out)
    if partial_ckpt is not None:
        env_parts.append(f"RESUME_FROM_CHECKPOINT={shlex.quote(str(partial_ckpt))}")
        log(f"  Resuming training from checkpoint: {partial_ckpt}")
    elif n > 1 and bool(tcfg.get("resume_from_previous_iteration", True)):
        prev_train_dir = iter_dir(cfg, n - 1) / "training"
        prev_ckpt = latest_checkpoint(prev_train_dir)
        if prev_ckpt is not None:
            env_parts.append(f"RESUME_FROM_CHECKPOINT={shlex.quote(str(prev_ckpt))}")
            log(f"  Resuming trainer state from previous iteration: {prev_ckpt}")
            # Bump SAVE_TOTAL_LIMIT by 1 so the last checkpoint in the current
            # iteration is not deleted by HF Trainer (the best checkpoint from a
            # previous iteration is counted against the limit).
            overrides = tcfg.get("env_overrides") or {}
            orig_limit = int(overrides.get("SAVE_TOTAL_LIMIT", 1))
            bumped = orig_limit + 1
            env_parts.append(f"SAVE_TOTAL_LIMIT={bumped}")
            log(f"  Bumped SAVE_TOTAL_LIMIT {orig_limit} -> {bumped} (cross-iteration resume)")

    env_str = " ".join(env_parts)
    cmd_str = (
        f"source /root/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {tcfg['conda_env']} && "
        f"cd {tcfg['cwd']} && "
        f"{env_str} bash {tcfg['script']}"
    )

    log(f"Running training (env={tcfg['conda_env']}, model={model_path})...")
    log(f"  output_dir: {train_out}")
    if dry_run:
        log(f"  [dry-run] cmd: {cmd_str}")
        return

    train_out.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["bash", "-lc", cmd_str])
    log("Training finished.")


# ---------------------------------------------------------------------------
# Tokenizer fix
# ---------------------------------------------------------------------------

def apply_tokenizer_fix(cfg: dict, ckpt: Path, model_path: str, *, dry_run: bool) -> None:
    fxcfg = cfg.get("tokenizer_fix", {})
    if not fxcfg.get("apply_after_each_train", True):
        return

    source_dir = Path(model_path)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]

    log(f"  Syncing tokenizer assets from {source_dir} -> {ckpt}")
    if dry_run:
        log("  [dry-run] skip tokenizer sync")
        return

    for name in tokenizer_files:
        src = source_dir / name
        dst = ckpt / name
        if src.exists():
            shutil.copy2(src, dst)
        elif dst.exists():
            dst.unlink()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train-generate loop orchestrator.")
    parser.add_argument("--config", required=True, help="Path to train_loop.yaml")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without executing any subprocesses.",
    )
    parser.add_argument(
        "--start-iteration", type=int, default=1,
        help="Force start from this iteration number (default: auto-detect from state).",
    )
    args = parser.parse_args()
    dry_run: bool = args.dry_run

    cfg = load_yaml(Path(args.config).expanduser().resolve())
    num_iter = int(cfg["loop"]["num_iterations"])

    # Ctrl-C handler: cleanly kill vLLM session before exit
    def _sigint_handler(sig, frame):
        log("Received SIGINT. Tearing down vLLM session before exit...")
        stop_vllm(cfg, dry_run=False)
        sys.exit(1)
    signal.signal(signal.SIGINT, _sigint_handler)

    log(f"Starting train-generate loop: {num_iter} iteration(s), version={cfg['loop']['version']}")
    if dry_run:
        log("DRY-RUN mode enabled — no subprocesses will be executed.")

    for n in range(1, num_iter + 1):
        idir = iter_dir(cfg, n)
        idir.mkdir(parents=True, exist_ok=True)
        log(f"=== Iteration {n}/{num_iter} -> {idir} ===")

        # --- Data generation phase ---
        if data_done(idir):
            log(f"Data already done for iteration {n}, skipping generation.")
        else:
            export_only_resume = prune_done(idir)
            if export_only_resume:
                log(
                    "Prune output exists but msswift export is missing; "
                    "forcing generate step to resume from export."
                )
                run_generate(cfg, idir, n, dry_run=dry_run, force_export_resume=True)
            else:
                model_path = pick_serving_model(cfg, n, dry_run=dry_run)
                log(f"Serving model: {model_path}")
                ensure_vllm(cfg, model_path, dry_run=dry_run)
                try:
                    with VllmWatchdog(cfg, model_path):
                        run_generate(cfg, idir, n, dry_run=dry_run)
                finally:
                    # Always stop vLLM after generation to free GPU for training
                    stop_vllm(cfg, dry_run=dry_run)

            if not dry_run and not data_done(idir):
                raise RuntimeError(
                    f"Data generation finished but msswift_ppo.jsonl not found in "
                    f"{idir / 'msswift_export'}. Check generation logs."
                )

        # Make sure vLLM is stopped before training (even if data was already done)
        if not dry_run:
            stop_vllm(cfg, dry_run=False)

        # --- Training phase ---
        if train_done(idir):
            log(f"Training already done for iteration {n}, skipping.")
        else:
            # Use base_model if n==1 and no previous checkpoint; otherwise use iter n model
            model_path = pick_serving_model(cfg, n, dry_run=dry_run)
            run_train(cfg, idir, n, model_path, dry_run=dry_run)

            # Find the checkpoint produced by this training run
            train_dir = idir / "training"
            ckpt = find_latest_partial_ckpt(train_dir)
            if ckpt is None and n > 1:
                # save_total_limit may have deleted the current checkpoint because
                # the best checkpoint lives in a previous iteration directory.
                # Fall back to that previous best checkpoint.
                prev_ckpt = latest_checkpoint(iter_dir(cfg, n - 1) / "training")
                if prev_ckpt is not None:
                    log(f"  No checkpoint in {train_dir}; falling back to previous "
                        f"iteration best: {prev_ckpt}")
                    ckpt = prev_ckpt
            if ckpt is None and not dry_run:
                raise RuntimeError(
                    f"Training finished but no checkpoint found in {train_dir}."
                )

            if ckpt is not None:
                apply_tokenizer_fix(cfg, ckpt, model_path, dry_run=dry_run)
                if not dry_run:
                    write_train_markers(train_dir, ckpt)
            else:
                log("  [dry-run] No checkpoint to write markers for.")

        log(f"=== Iteration {n} complete ===\n")

    log(f"All {num_iter} iteration(s) finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
