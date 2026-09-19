"""Batched autoregression with probe-priority reconstruction and an unchanged history."""
from __future__ import annotations

import torch

from .data import build_neighbor_table, route_key
from .model import select_cache
from .runtime import autocast


def transition(neighbors, source, direction, predicted_node):
    """The frozen graph judges the action; the probe always determines arrival."""
    expected = neighbors.get(source, {}).get(direction)
    kind = 'illegal' if expected is None else ('legal_correct' if predicted_node == expected else 'legal_mismatch')
    return {'source': source, 'direction': direction, 'target': predicted_node,
            'expected_target': expected, 'kind': kind}


def sample_tokens(logits, cfg, tokenizer, generator):
    adjusted = logits.float().clone()
    if cfg.get('syntax_only', False):
        permitted = torch.zeros(adjusted.size(-1), dtype=torch.bool, device=adjusted.device)
        permitted[[tokenizer.eos_id, *tokenizer.direction_ids.values()]] = True
        adjusted[:, ~permitted] = -torch.inf
    if cfg['temperature'] == 0:
        chosen = adjusted.argmax(-1)
        return chosen, torch.ones(len(chosen), device=logits.device)
    adjusted /= cfg['temperature']
    k = cfg.get('top_k', 0)
    if k and k < adjusted.size(-1):
        # Keep exactly k token indices even when values tie.
        top_values, top_indices = adjusted.topk(k, dim=-1)
        adjusted.fill_(-torch.inf).scatter_(1, top_indices, top_values)
    probabilities = adjusted.softmax(-1)
    chosen = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    return chosen, probabilities.gather(1, chosen[:, None]).squeeze(1)


@torch.inference_mode()
def generate_samples(model, probe, probe_nodes, dataset, tokenizer, config, device, dtype):
    model.eval()
    probe.eval()
    cfg, probe_cfg = config['generation'], config['probe']
    neighbors = build_neighbor_table(dataset['graph'])
    node_to_class = {node: i for i, node in enumerate(probe_nodes)}
    training_routes = {route_key(route) for route in dataset['splits']['train']}
    generator = torch.Generator(device=device).manual_seed(cfg['seed'])
    records = []
    for cohort in ('seen', 'unseen'):
        references = dataset['splits'][cohort]
        for start in range(0, len(references), cfg['batch_size']):
            batch = references[start:start + cfg['batch_size']]
            samples = [{'id': route['id'], 'cohort': cohort, 'origin': route['origin'],
                        'destination': route['destination'], 'reference_nodes': route['nodes'],
                        'reference_directions': route['directions'], 'events': [], 'generated_directions': [],
                        'generated_tokens': [], 'termination': 'max_new_tokens', 'physical_valid': True,
                        'physical_final_node': route['origin'], 'inferred_final_node': route['origin']}
                       for route in batch]
            if not batch:
                continue
            tokens = torch.tensor([[tokenizer.node(r['origin']), tokenizer.node(r['destination'])] for r in batch], device=device)
            with autocast(device, dtype):
                output = model(tokens, use_cache=True, probe_layer=probe_cfg['layer'])
            initial_probs = probe(output.hidden[:, -1].float()).softmax(-1)
            initial_conf, initial_pred = initial_probs.max(-1)
            for i, sample in enumerate(samples):
                sample['initial_probe'] = {'node': probe_nodes[initial_pred[i].item()], 'confidence': initial_conf[i].item(),
                                           'used_as_origin': False}
            active = list(range(len(batch)))
            for step in range(1, cfg['max_new_tokens'] + 1):
                logits = output.logits[:, -1]
                chosen, sampling_probs = sample_tokens(logits, cfg, tokenizer, generator)
                raw_probs = logits.float().softmax(-1).gather(1, chosen[:, None]).squeeze(1).tolist()
                selected_tokens, sampling_probs_cpu = chosen.tolist(), sampling_probs.tolist()
                continuing, next_active = [], []
                for local, sample_index in enumerate(active):
                    sample = samples[sample_index]
                    token = selected_tokens[local]
                    sample['generated_tokens'].append(tokenizer.decode_token(token))
                    if token == tokenizer.eos_id:
                        sample['termination'] = 'eos'
                        sample['termination_probability'] = raw_probs[local]
                    elif token not in tokenizer.id_directions:
                        sample['termination'] = 'invalid_token'
                        sample['invalid_token'] = tokenizer.decode_token(token)
                    else:
                        continuing.append(local)
                        next_active.append(sample_index)
                if not continuing:
                    break
                indices = torch.tensor(continuing, device=device, dtype=torch.long)
                # Feed the selected action even when it is illegal. Never inject a probed node.
                with autocast(device, dtype):
                    output = model(chosen.index_select(0, indices)[:, None],
                                   past=select_cache(output.cache, indices), use_cache=True,
                                   probe_layer=probe_cfg['layer'])
                probabilities = probe(output.hidden[:, -1].float()).softmax(-1)
                confidence, predicted = probabilities.max(-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
                top_values, top_indices = probabilities.topk(min(len(probe_nodes), probe_cfg['top_k']), dim=-1)
                # Batched transfers; Python only does graph lookups and output bookkeeping.
                probs_cpu = probabilities.cpu()
                confidence_cpu, predicted_cpu = confidence.tolist(), predicted.tolist()
                values_cpu, indices_cpu, entropy_cpu = top_values.tolist(), top_indices.tolist(), entropy.tolist()
                for local, (old_local, sample_index) in enumerate(zip(continuing, next_active)):
                    sample = samples[sample_index]
                    direction = tokenizer.id_directions[selected_tokens[old_local]]
                    target = probe_nodes[predicted_cpu[local]]
                    event = transition(neighbors, sample['inferred_final_node'], direction, target)
                    physical_source = sample['physical_final_node']
                    physical_target = neighbors.get(physical_source, {}).get(direction) if physical_source is not None else None
                    if physical_target is None:
                        sample['physical_valid'] = False
                    sample['physical_final_node'] = physical_target
                    expected = event['expected_target']
                    event.update({'step': step, 'probe_confidence': confidence_cpu[local],
                                  'probe_entropy': entropy_cpu[local],
                                  'source_probe_confidence': sample['events'][-1]['probe_confidence'] if sample['events'] else 1.0,
                                  'expected_target_probability': probs_cpu[local, node_to_class[expected]].item() if expected is not None else None,
                                  'direction_probability': raw_probs[old_local],
                                  'sampling_probability': sampling_probs_cpu[old_local],
                                  'physical_source': physical_source, 'physical_target': physical_target,
                                  'probe_topk': [{'node': probe_nodes[index], 'probability': value}
                                                 for index, value in zip(indices_cpu[local], values_cpu[local])]})
                    if cfg.get('save_full_probe_probabilities', False):
                        event['probe_probabilities'] = {str(node): value for node, value in zip(probe_nodes, probs_cpu[local].tolist())}
                    sample['events'].append(event)
                    sample['generated_directions'].append(direction)
                    sample['inferred_final_node'] = target
                active = next_active
            for sample in samples:
                sample['destination_reached'] = sample['physical_valid'] and sample['physical_final_node'] == sample['destination']
                sample['probe_destination_reached'] = sample['inferred_final_node'] == sample['destination']
                sample['route_success'] = sample['termination'] == 'eos' and sample['destination_reached']
                sample['exact_training_route_match'] = (sample['termination'] == 'eos' and
                    (sample['origin'], sample['destination'], tuple(sample['generated_directions'])) in training_routes)
                sample['reference_sequence_match'] = sample['generated_directions'] == sample['reference_directions']
                sample['generated_length'] = len(sample['events'])
            records.extend(samples)
        print(f"Generated {len(references)} {cohort}-pair samples", flush=True)
    return records
