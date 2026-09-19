"""Frozen-activation linear location probes, split by whole route."""
from __future__ import annotations

import bisect
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import autocast, save_json, save_torch, seed_everything
from .train import collate_routes


@torch.inference_mode()
def extract_features(model, routes, tokenizer, nodes, cfg, device, dtype, max_positions=None):
    """The prompt-end activation labels the origin; each action labels its arrival.

    For [origin, destination, action1, ...], token positions [1, 2, ...]
    have location labels [origin, node_after_action1, ...]. EOS is excluded.
    """
    model.eval()
    label_index = {node: index for index, node in enumerate(nodes)}
    total = sum(len(r['nodes']) for r in routes)
    chosen = None
    if max_positions is not None and max_positions < total:
        chosen = sorted(random.Random(cfg['seed']).sample(range(total), max_positions))
    features, labels, offset = [], [], 0
    batch_size = cfg['extraction_batch_size']
    for start in range(0, len(routes), batch_size):
        batch = routes[start:start + batch_size]
        tokens = collate_routes(batch, tokenizer).to(device)
        with autocast(device, dtype):
            hidden = model(tokens[:, :-1], probe_layer=cfg['layer']).hidden
        batch_indices, token_indices, batch_labels = [], [], []
        for local_index, route in enumerate(batch):
            n = len(route['nodes'])
            indices = range(n) if chosen is None else [
                index - offset for index in chosen[bisect.bisect_left(chosen, offset):bisect.bisect_left(chosen, offset + n)]]
            for position in indices:
                batch_indices.append(local_index)
                token_indices.append(position + 1)
                batch_labels.append(label_index[route['nodes'][position]])
            offset += n
        if batch_indices:
            features.append(hidden[batch_indices, token_indices].detach().to(device='cpu', dtype=torch.float32))
            labels.append(torch.tensor(batch_labels, dtype=torch.long))
    if not features:
        return torch.empty((0, model.config['dim'])), torch.empty(0, dtype=torch.long)
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def evaluate_probe(probe, features, labels, nodes, cfg, device):
    probe.eval()
    count, classes = len(labels), len(nodes)
    if not count:
        return {'positions': 0, 'accuracy': None, 'balanced_accuracy': None, 'nll': None,
                'brier': None, 'ece': None, 'calibration': [], 'per_node': []}
    class_total, class_correct = torch.zeros(classes, dtype=torch.long), torch.zeros(classes, dtype=torch.long)
    bins = cfg['calibration_bins']
    bin_count, bin_conf, bin_correct = [0] * bins, [0.0] * bins, [0] * bins
    loss = brier = 0.0
    correct = top_correct = 0
    k = min(classes, cfg['top_k'])
    for start in range(0, count, cfg['batch_size']):
        x = features[start:start + cfg['batch_size']].to(device, dtype=torch.float32)
        y = labels[start:start + cfg['batch_size']].to(device)
        logits = probe(x)
        probabilities = logits.softmax(-1)
        confidence, prediction = probabilities.max(-1)
        match = prediction == y
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        brier += (probabilities.square().sum(-1) - 2 * probabilities.gather(1, y[:, None]).squeeze(1) + 1).sum().item()
        correct += match.sum().item()
        top_correct += (logits.topk(k, dim=-1).indices == y[:, None]).any(-1).sum().item()
        y_cpu, match_cpu, confidence_cpu = y.cpu(), match.cpu(), confidence.cpu()
        class_total += torch.bincount(y_cpu, minlength=classes)
        class_correct += torch.bincount(y_cpu[match_cpu], minlength=classes)
        for conf, ok in zip(confidence_cpu.tolist(), match_cpu.tolist()):
            index = min(int(conf * bins), bins - 1)
            bin_count[index] += 1
            bin_conf[index] += conf
            bin_correct[index] += int(ok)
    reliability = [{'lower': i / bins, 'upper': (i + 1) / bins, 'count': bin_count[i],
                    'confidence': bin_conf[i] / bin_count[i] if bin_count[i] else None,
                    'accuracy': bin_correct[i] / bin_count[i] if bin_count[i] else None} for i in range(bins)]
    present = class_total > 0
    return {'positions': count, 'accuracy': correct / count,
            'balanced_accuracy': (class_correct[present].float() / class_total[present]).mean().item(),
            'nll': loss / count, 'brier': brier / count,
            'ece': sum(abs(r['confidence'] - r['accuracy']) * r['count'] / count for r in reliability if r['count']),
            'top_k': k, 'top_k_accuracy': top_correct / count, 'calibration': reliability,
            'per_node': [{'node': node, 'count': class_total[i].item(),
                         'accuracy': class_correct[i].item() / class_total[i].item() if class_total[i] else None}
                         for i, node in enumerate(nodes)]}


def fit_probe(model, dataset, tokenizer, cfg, device, dtype, output):
    if not dataset['splits']['probe_train']:
        raise ValueError('No disjoint probe-training routes available. Increase max_length/max_attempts or reduce held-out counts.')
    seed_everything(cfg['seed'])
    nodes = list(dataset['graph']['nodes'])
    start = time.perf_counter()
    features, labels = extract_features(model, dataset['splits']['probe_train'], tokenizer, nodes, cfg, device, dtype,
                                        cfg.get('max_train_positions'))
    validation_x, validation_y = extract_features(model, dataset['splits']['probe_validation'], tokenizer, nodes, cfg, device, dtype)
    cache_device = cfg.get('cache_device', 'auto')
    on_gpu = cache_device == 'cuda'
    if cache_device == 'auto' and device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        on_gpu = (features.numel() * features.element_size() + labels.numel() * labels.element_size()) < free * cfg.get('cache_gpu_fraction', 0.25)
    if on_gpu:
        if device.type != 'cuda':
            raise ValueError('probe.cache_device=cuda requires a CUDA training device')
        features, labels = features.to(device), labels.to(device)
    probe = nn.Linear(model.config['dim'], len(nodes)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    class_counts = torch.bincount(labels, minlength=len(nodes)).cpu()
    weights = None
    if cfg.get('class_balance', False):
        weights = (class_counts.sum() / class_counts.clamp_min(1).float()).to(device)
        weights /= weights.mean()
    history, best_nll, best_epoch = [], math.inf, None
    for epoch in range(cfg['epochs']):
        probe.train()
        order = torch.randperm(len(labels), device=features.device)
        loss_sum = 0.0
        for indices in order.split(cfg['batch_size']):
            x, y = features[indices].to(device), labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(x), y, weight=weights)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        validation = evaluate_probe(probe, validation_x, validation_y, nodes, cfg, device)
        nll = validation['nll'] if validation['nll'] is not None else loss_sum / len(labels)
        history.append({'epoch': epoch + 1, 'training_loss': loss_sum / len(labels),
                        'validation_nll': validation['nll'], 'validation_accuracy': validation['accuracy']})
        if nll < best_nll:
            best_nll, best_epoch = nll, epoch + 1
            save_torch(Path(output) / 'probe.pt', {'state_dict': probe.state_dict(), 'nodes': nodes,
                       'layer': cfg['layer'], 'epoch': best_epoch})
        if epoch == 0 or epoch + 1 == cfg['epochs']:
            print(f"Probe epoch {epoch + 1}/{cfg['epochs']}, validation accuracy={validation['accuracy']}", flush=True)
    probe.load_state_dict(torch.load(Path(output) / 'probe.pt', map_location=device, weights_only=True)['state_dict'])
    summary = {'kind': 'linear', 'layer': cfg['layer'], 'classes': nodes, 'best_epoch': best_epoch,
               'train_positions': len(labels), 'train_routes': len(dataset['splits']['probe_train']),
               'train_class_counts': {str(node): class_counts[i].item() for i, node in enumerate(nodes)},
               'unrepresented_train_nodes': [node for i, node in enumerate(nodes) if class_counts[i] == 0],
               'feature_cache_device': str(features.device), 'seconds': time.perf_counter() - start,
               'history': history, 'validation': evaluate_probe(probe, validation_x, validation_y, nodes, cfg, device),
               'confidence_note': 'Softmax confidence is uncalibrated. Legal held-out accuracy does not validate activations after illegal prefixes.'}
    # Evaluate on untouched reference routes; these metrics never select a probe checkpoint.
    for cohort in ('seen', 'unseen'):
        x, y = extract_features(model, dataset['splits'][cohort], tokenizer, nodes, cfg, device, dtype)
        summary[f'test_{cohort}'] = evaluate_probe(probe, x, y, nodes, cfg, device)
    save_json(Path(output) / 'probe.json', summary)
    return probe, nodes, summary
