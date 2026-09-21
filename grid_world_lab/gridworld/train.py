"""Teacher-forced next-token learning with padding excluded from the loss."""
from __future__ import annotations

import math
import time
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import build_neighbor_table
from .runtime import autocast, save_json, save_torch, seed_everything


def collate_routes(routes, tokenizer):
    encoded = [torch.tensor(tokenizer.encode(r), dtype=torch.long) for r in routes]
    return torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=tokenizer.pad_id)


def route_loader(routes, tokenizer, cfg, shuffle=False):
    return DataLoader(routes, batch_size=cfg['batch_size'], shuffle=shuffle,
                      num_workers=cfg.get('num_workers', 0),
                      pin_memory=cfg.get('device', 'auto') != 'cpu' and torch.cuda.is_available(),
                      collate_fn=partial(collate_routes, tokenizer=tokenizer))


def targets_for(tokens, tokenizer, loss_on_prompt=True):
    targets = tokens[:, 1:].clone()
    targets[targets == tokenizer.pad_id] = -100
    if not loss_on_prompt:
        targets[:, 0] = -100
    return targets


@torch.inference_mode()
def evaluate_loss(model, routes, tokenizer, cfg, device, dtype, graph=None):
    if not routes:
        return {'loss': None, 'perplexity': None, 'token_accuracy': None, 'tokens': 0,
                'action_loss': None, 'action_token_accuracy': None, 'action_tokens': 0,
                'legal_action_accuracy': None, 'legal_action_correct': 0, 'legal_action_tokens': 0}
    model.eval()
    total_loss = total_correct = total_tokens = 0
    action_loss = action_correct = action_tokens = 0
    legal_correct = legal_tokens = 0
    neighbors = build_neighbor_table(graph) if graph is not None else None
    for tokens in route_loader(routes, tokenizer, cfg):
        tokens = tokens.to(device, non_blocking=True)
        targets = targets_for(tokens, tokenizer, cfg.get('loss_on_prompt', True))
        with autocast(device, dtype):
            logits = model(tokens[:, :-1]).logits
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100, reduction='sum')
            per_action_loss = F.cross_entropy(logits[:, 1:].float().flatten(0, 1), targets[:, 1:].flatten(),
                                              ignore_index=-100, reduction='sum')
        valid = targets != -100
        valid_actions = targets[:, 1:] != -100
        total_loss += loss.item()
        total_correct += ((logits.argmax(-1) == targets) & valid).sum().item()
        total_tokens += valid.sum().item()
        predictions = logits.argmax(-1)
        action_loss += per_action_loss.item()
        action_correct += ((predictions[:, 1:] == targets[:, 1:]) & valid_actions).sum().item()
        action_tokens += valid_actions.sum().item()
        if neighbors is not None:
            for row in range(tokens.size(0)):
                current = int(tokens[row, 0].item()) - tokenizer.node_offset + 1
                destination = int(tokens[row, 1].item()) - tokenizer.node_offset + 1
                for position in range(1, targets.size(1)):
                    target = int(targets[row, position].item())
                    if target == -100:
                        break
                    predicted = int(predictions[row, position].item())
                    legal = predicted == tokenizer.eos_id and current == destination
                    direction = tokenizer.id_directions.get(predicted)
                    if direction is not None and direction in neighbors[current]:
                        legal = True
                    legal_correct += int(legal)
                    legal_tokens += 1
                    true_direction = tokenizer.id_directions.get(target)
                    if true_direction is not None:
                        current = neighbors[current][true_direction]
    mean = total_loss / total_tokens
    return {'loss': mean, 'perplexity': math.exp(min(mean, 700)),
            'token_accuracy': total_correct / total_tokens, 'tokens': total_tokens,
            'action_loss': action_loss / action_tokens,
            'action_token_accuracy': action_correct / action_tokens,
            'action_tokens': action_tokens,
            'legal_action_accuracy': legal_correct / legal_tokens if neighbors is not None else None,
            'legal_action_correct': legal_correct, 'legal_action_tokens': legal_tokens}


def train_model(model, dataset, tokenizer, cfg, device, dtype, output):
    seed_everything(cfg['seed'])
    output = Path(output)
    model.to(device)
    model.gradient_checkpointing = cfg.get('gradient_checkpointing', False)
    forward_model = torch.compile(model) if cfg.get('compile', False) else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'], fused=device.type == 'cuda')
    loader = route_loader(dataset['splits']['train'], tokenizer, cfg, shuffle=True)
    accumulation = cfg['gradient_accumulation']
    planned_steps = cfg['epochs'] * math.ceil(len(loader) / accumulation)
    if cfg.get('max_steps') is not None:
        planned_steps = min(planned_steps, cfg['max_steps'])
    warmup = int(planned_steps * cfg.get('warmup_fraction', 0.05))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, planned_steps - warmup - 1)
        floor = cfg.get('min_lr_ratio', 0.1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
    optimizer.zero_grad(set_to_none=True)
    history, step, best_loss, best_epoch = [], 0, float('inf'), None
    skipped_overflows = 0
    # A constant token-scale divisor keeps FP16 backprop near a mean loss while
    # preserving exact weighting across unequal sequence lengths and microbatches.
    backward_normalizer = cfg['batch_size'] * accumulation * model.context_length
    start = time.perf_counter()
    total_positions = 0
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss, epoch_tokens, accumulated_tokens = 0.0, 0, 0
        for microstep, tokens in enumerate(loader):
            tokens = tokens.to(device, non_blocking=True)
            targets = targets_for(tokens, tokenizer, cfg.get('loss_on_prompt', True))
            count = (targets != -100).sum().item()
            with autocast(device, dtype):
                logits = forward_model(tokens[:, :-1]).logits
                loss_sum = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                                           ignore_index=-100, reduction='sum')
            scaler.scale(loss_sum / backward_normalizer).backward()
            epoch_loss += loss_sum.detach().item()
            epoch_tokens += count
            accumulated_tokens += count
            total_positions += count
            if (microstep + 1) % accumulation == 0 or microstep + 1 == len(loader):
                scaler.unscale_(optimizer)
                # Normalize across actual (un-padded) tokens, including a final partial batch.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(backward_normalizer / accumulated_tokens)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                if not torch.isfinite(grad_norm) and not scaler.is_enabled():
                    raise FloatingPointError('Non-finite gradient; try fp32/bf16 or a smaller learning rate.')
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_update = scaler.get_scale() >= previous_scale
                if did_update:
                    scheduler.step()
                    step += 1
                else:
                    skipped_overflows += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated_tokens = 0
                if did_update and (step % cfg.get('log_every_steps', 50) == 0 or step == planned_steps):
                    print(f"LM epoch {epoch + 1}, step {step}/{planned_steps}, loss {epoch_loss / epoch_tokens:.4f}", flush=True)
                if step >= planned_steps:
                    break
        row = {'epoch': epoch + 1, 'step': step, 'training_loss': epoch_loss / epoch_tokens}
        should_evaluate = (epoch + 1) % cfg.get('eval_every_epochs', 1) == 0 or step >= planned_steps or epoch + 1 == cfg['epochs']
        if should_evaluate:
            validation = evaluate_loss(model, dataset['splits']['validation'], tokenizer, cfg, device, dtype,
                                       dataset.get('graph'))
            row['validation'] = validation
            selection_loss = validation['loss'] if validation['loss'] is not None else row['training_loss']
            if selection_loss < best_loss:
                best_loss, best_epoch = selection_loss, epoch + 1
                save_torch(output / 'model.pt', {'state_dict': model.state_dict(), 'epoch': best_epoch,
                           'validation_loss': validation['loss'], 'context_length': model.context_length})
        history.append(row)
        save_json(output / 'training_history.json', history)
        if should_evaluate:
            legal = row['validation']['legal_action_accuracy']
            legal_text = 'n/a' if legal is None else f'{legal:.4f}'
            print(f"LM epoch {epoch + 1}: train={row['training_loss']:.4f}, "
                  f"validation={row['validation']['loss']}, legal_action_accuracy={legal_text}", flush=True)
        if step >= planned_steps:
            break
    if step == 0:
        raise FloatingPointError('Every FP16 optimizer update overflowed. Use bf16/fp32 or allow more epochs for dynamic loss scaling.')
    model.load_state_dict(torch.load(output / 'model.pt', map_location=device, weights_only=True)['state_dict'])
    elapsed = time.perf_counter() - start
    teacher_forced = {
        split: evaluate_loss(model, dataset['splits'].get(split, []), tokenizer, cfg, device, dtype,
                             dataset.get('graph'))
        for split in ('validation', 'seen', 'unseen')
    }
    summary = {'history': history, 'best_epoch': best_epoch, 'optimization_steps': step, 'skipped_overflow_updates': skipped_overflows,
               'parameters': sum(p.numel() for p in model.parameters()), 'seconds': elapsed,
               'training_tokens_processed': total_positions, 'tokens_per_second': total_positions / max(elapsed, 1e-9),
               'checkpoint_selection': 'validation_loss' if dataset['splits']['validation'] else 'training_loss',
               'final_validation': teacher_forced['validation'],
               'teacher_forced_evaluation': teacher_forced}
    save_json(output / 'training.json', summary)
    return summary
