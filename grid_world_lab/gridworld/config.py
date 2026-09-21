"""JSON configuration, recursive defaults, and typed dotted-path overrides."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "rows": 10, "cols": 10, "mode": "union", "map_samples": None,
        "train_samples": 20000,
        "min_length": 1, "max_length": 32,
        "heldout_min_length": None, "heldout_max_length": None,
        "validation_samples": 1000, "probe_train_samples": 2000,
        "probe_validation_samples": 500, "test_samples_per_cohort": 500,
        "seed": 42, "max_attempts": 200000,
    },
    "model": {
        "layers": 12, "dim": 768, "heads": 12, "dropout": 0.1,
        "mlp_ratio": 4, "context_length": None,
    },
    "train": {
        "epochs": 10, "batch_size": 32, "learning_rate": 0.0003,
        "weight_decay": 0.01, "gradient_accumulation": 1, "grad_clip": 1.0,
        "device": "auto", "precision": "auto", "seed": 42,
        "warmup_fraction": 0.05, "min_lr_ratio": 0.1, "num_workers": 0,
        "eval_every_epochs": 1, "max_steps": None, "compile": False,
        "gradient_checkpointing": False, "loss_on_prompt": True,
        "log_every_steps": 50, "cpu_threads": 4,
    },
    "probe": {
        "layer": -1, "epochs": 30, "batch_size": 2048,
        "extraction_batch_size": 64, "learning_rate": 0.01,
        "weight_decay": 0.0, "seed": 43, "max_train_positions": None,
        "class_balance": False, "calibration_bins": 10, "top_k": 5,
        "cache_device": "auto", "cache_gpu_fraction": 0.25,
    },
    "generation": {
        "max_new_tokens": 64, "batch_size": 64, "temperature": 1.0,
        "top_k": 0, "syntax_only": False, "seed": 44,
        "save_full_probe_probabilities": False,
    },
    "metrics": {
        "rarity_thresholds": [1, 5, 20], "topology_max_nodes": 250,
        "compute_topology_distortion": True,
    },
    "output": {"directory": "runs/default"},
}


def _merge(defaults: dict[str, Any], patch: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = copy.deepcopy(defaults)
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {path}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section {path} must be a JSON object.")
            result[key] = _merge(defaults[key], value, path)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set an existing config leaf; useful for command-line and parameter sweeps."""
    keys = dotted_key.split(".")
    if not keys or any(not key for key in keys):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    target = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise ValueError(f"Unknown configuration key: {dotted_key}")
        target = target[key]
    if keys[-1] not in target or isinstance(target[keys[-1]], dict):
        raise ValueError(f"Override must name an existing configuration leaf: {dotted_key}")
    target[keys[-1]] = copy.deepcopy(value)


def validate_config(config: dict[str, Any]) -> None:
    def get(path: str) -> Any:
        item: Any = config
        for key in path.split("."):
            item = item[key]
        return item

    def integer(path: str, minimum: int = 1, optional: bool = False) -> None:
        value = get(path)
        if optional and value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{path} must be an integer >= {minimum}{' or null' if optional else ''}.")

    def number(path: str, minimum: float = 0, maximum: float | None = None, strict: bool = False) -> None:
        value = get(path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be numeric.")
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite; NaN and infinity are not supported.")
        if (value <= minimum if strict else value < minimum) or (maximum is not None and value > maximum):
            raise ValueError(f"{path} is outside its allowed range.")

    for path in (
        "data.rows", "data.cols", "data.train_samples", "data.min_length", "data.max_length",
        "data.max_attempts", "model.layers", "model.dim", "model.heads", "model.mlp_ratio",
        "train.epochs", "train.batch_size", "train.gradient_accumulation", "train.eval_every_epochs",
        "train.log_every_steps", "train.cpu_threads", "probe.epochs", "probe.batch_size", "probe.extraction_batch_size",
        "probe.calibration_bins", "probe.top_k", "generation.max_new_tokens", "generation.batch_size",
        "metrics.topology_max_nodes",
    ):
        integer(path)
    for path in (
        "data.validation_samples", "data.probe_train_samples", "data.probe_validation_samples",
        "data.test_samples_per_cohort", "train.num_workers", "generation.top_k",
        "data.seed", "train.seed", "probe.seed", "generation.seed",
    ):
        integer(path, 0)
    for path in ("data.map_samples", "data.heldout_min_length", "data.heldout_max_length", "model.context_length", "train.max_steps", "probe.max_train_positions"):
        integer(path, optional=True)
    integer("probe.layer", -1)
    if config["data"]["rows"] * config["data"]["cols"] < 2:
        raise ValueError("The grid must contain at least two nodes for positive-length walks.")
    if config["data"]["min_length"] > config["data"]["max_length"]:
        raise ValueError("data.min_length exceeds data.max_length.")
    if config["data"]["mode"] not in {"union", "frozen_map"}:
        raise ValueError("data.mode must be 'union' or 'frozen_map'.")
    map_samples = config["data"]["map_samples"]
    if config["data"]["mode"] == "frozen_map":
        if map_samples is None:
            raise ValueError("data.map_samples is required when data.mode='frozen_map'.")
        if map_samples > config["data"]["train_samples"]:
            raise ValueError("data.map_samples must not exceed data.train_samples.")
    heldout_min = config["data"]["heldout_min_length"] or config["data"]["min_length"]
    heldout_max = config["data"]["heldout_max_length"] or config["data"]["max_length"]
    if heldout_min > heldout_max:
        raise ValueError("The held-out minimum length exceeds its maximum length.")
    if config["model"]["dim"] % config["model"]["heads"]:
        raise ValueError("model.dim must be divisible by model.heads.")
    if config["probe"]["layer"] >= config["model"]["layers"]:
        raise ValueError("probe.layer must be -1 (final normalization) or a zero-based transformer layer index.")
    number("model.dropout", 0, 1)
    for path in ("train.learning_rate", "probe.learning_rate"):
        number(path, strict=True)
    for path in ("train.weight_decay", "probe.weight_decay", "train.grad_clip", "generation.temperature"):
        number(path)
    for path in ("train.warmup_fraction", "train.min_lr_ratio", "probe.cache_gpu_fraction"):
        number(path, 0, 1)
    device = config["train"]["device"]
    if not isinstance(device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?|mps", device):
        raise ValueError("train.device must be auto, cpu, cuda, cuda:<index>, or mps.")
    if config["train"]["precision"] not in ("auto", "fp32", "fp16", "bf16"):
        raise ValueError("train.precision must be auto, fp32, fp16, or bf16.")
    if config["probe"]["cache_device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("probe.cache_device must be auto, cpu, or cuda.")
    thresholds = config["metrics"]["rarity_thresholds"]
    if not isinstance(thresholds, list) or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in thresholds):
        raise ValueError("metrics.rarity_thresholds must be a list of positive integers.")
    if thresholds != sorted(set(thresholds)):
        raise ValueError("metrics.rarity_thresholds must be unique and increasing.")
    for path in (
        "train.compile", "train.gradient_checkpointing", "train.loss_on_prompt", "probe.class_balance",
        "generation.syntax_only", "generation.save_full_probe_probabilities", "metrics.compute_topology_distortion",
    ):
        if not isinstance(get(path), bool):
            raise ValueError(f"{path} must be true or false.")
    if not isinstance(config["output"]["directory"], str) or not config["output"]["directory"].strip():
        raise ValueError("output.directory must be a nonempty path string.")


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a partial JSON config and apply repeatable ``key=JSON`` overrides.

    Bare values such as ``train.device=cuda`` are accepted as strings. JSON
    notation is required for booleans, null, arrays, and numeric values.
    Paths inside the configuration are kept relative to the calling directory.
    """
    patch: dict[str, Any] = {}
    if path is not None:
        patch = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(patch, dict):
            raise ValueError("The configuration file must contain a JSON object.")
    config = _merge(DEFAULT_CONFIG, patch)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Expected a dotted key=value override, got {override!r}.")
        key, raw = override.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        set_override(config, key.strip(), value)
    validate_config(config)
    return config
