"""Shared runtime, reproducibility, and artifact utilities."""
from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_and_dtype(cfg):
    requested = cfg.get('device', 'auto')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install a CUDA PyTorch wheel or use train.device=cpu.')
    precision = cfg.get('precision', 'auto')
    if precision == 'auto':
        precision = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device.type == 'cuda' else 'fp32'
    if precision not in {'bf16', 'fp16', 'fp32'}:
        raise ValueError('train.precision must be auto, bf16, fp16, or fp32')
    if precision == 'fp16' and device.type != 'cuda':
        raise ValueError('fp16 requires CUDA; use fp32 or bf16 on CPU')
    return device, {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]


def autocast(device, dtype):
    return (contextlib.nullcontext() if dtype == torch.float32 else
            torch.autocast(device_type=device.type, dtype=dtype))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def environment_info(device, dtype):
    import platform
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'device': str(device), 'precision': str(dtype),
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
