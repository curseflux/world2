"""Counterfactual action and inverse-action probes of saved checkpoints.

An extracted arrow is a probe readout under an intervention, not a claim that
the LM would choose that action or that the probe is valid off distribution.
"""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random

from .data import DIRECTIONS, build_neighbor_table
from .pipeline import export_csv, json_read, json_write

OPPOSITE = dict(zip(DIRECTIONS, ('S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE')))


def collect_contexts(dataset, samples, per_node, seed, context_length, splits):
    """Reservoir-sample unique prefixes separately per source and context kind."""
    if per_node < 1:
        raise ValueError('contexts_per_node must be positive')
    rng = random.Random(seed)
    buckets, counts, used = defaultdict(list), Counter(), set()
    skipped = 0

    def add(context):
        nonlocal skipped
        key = (context['kind'], context['origin'], context['destination'], tuple(context['prefix']))
        if key in used:
            return
        used.add(key)
        if len(context['prefix']) + 4 > context_length:
            skipped += 1
            return
        bucket = context['kind'], context['source']
        counts[bucket] += 1
        if len(buckets[bucket]) < per_node:
            buckets[bucket].append(context)
        else:
            index = rng.randrange(counts[bucket])
            if index < per_node:
                buckets[bucket][index] = context

    for split in splits:
        if split not in dataset['splits'] or split in {'train', 'probe_train'}:
            raise ValueError('Use held-out context splits: seen, unseen, validation, probe_validation')
        for route in dataset['splits'][split]:
            # Only positions before a reference action: do not extend past EOS.
            for step, source in enumerate(route['nodes'][:-1]):
                add({'kind': 'reference', 'split': split, 'route_id': route['id'],
                     'origin': route['origin'], 'destination': route['destination'],
                     'source': source, 'known_source': source,
                     'prefix': route['directions'][:step], 'step': step + 1})
    neighbors = build_neighbor_table(dataset['graph'])
    for sample in samples:
        physical = sample['origin']
        prefix = []
        for event in sample['events']:
            add({'kind': 'generated_valid' if physical is not None else 'generated_invalid',
                 'split': sample['cohort'], 'route_id': sample['id'],
                 'origin': sample['origin'], 'destination': sample['destination'],
                 'source': physical if physical is not None else event['source'],
                 'known_source': physical, 'original_inferred_source': event['source'],
                 'original_action': event['direction'], 'prefix': list(prefix),
                 'step': len(prefix) + 1})
            physical = neighbors.get(physical, {}).get(event['direction'])
            prefix.append(event['direction'])
    contexts = [context for key in sorted(buckets) for context in buckets[key]]
    for index, context in enumerate(contexts):
        context['id'] = index
    return contexts, {'eligible_by_source': [
        {'kind': kind, 'source': source, 'available': count, 'selected': len(buckets[kind, source])}
        for (kind, source), count in sorted(counts.items())], 'skipped_context_limit': skipped}


def summarize_transitions(contexts, events, nodes, require_correct_source=False):
    by_id = {context['id']: context for context in contexts}
    groups = defaultdict(list)
    for event in events:
        context = by_id[event['context_id']]
        if require_correct_source and (context['known_source'] is None or
                                       context['pre_node'] != context['known_source']):
            continue
        groups[context['kind'], context['source'], event['direction']].append(event)
    rows = []
    for (kind, source, direction), group in sorted(groups.items()):
        votes = Counter(event['target'] for event in group)
        target = min(votes, key=lambda node: (-votes[node], node))
        count = len(group)
        mean = [sum(e['probabilities'][i] for e in group) / count for i in range(len(nodes))]
        expected = group[0]['true_neighbor']
        rows.append({'kind': kind, 'source': source, 'direction': direction,
                     'contexts': count, 'target': target, 'agreement': votes[target] / count,
                     'true_neighbor': expected,
                     'correct_target_rate': sum(e['target'] == expected for e in group) / count if expected is not None else None,
                     'mean_direction_probability': sum(e['direction_probability'] for e in group) / count,
                     'mean_probe_confidence': sum(e['confidence'] for e in group) / count,
                     'cycle_return_to_source_rate': sum(e['return_node'] == source for e in group) / count,
                     'cycle_return_to_preprobe_rate': sum(e['return_node'] == by_id[e['context_id']]['pre_node'] for e in group) / count,
                     'distinct_destinations': len({by_id[e['context_id']]['destination'] for e in group}),
                     'votes': dict(sorted(votes.items())), 'mean_probabilities': mean})
    lookup = {(row['kind'], row['source'], row['direction']): row for row in rows}
    for row in rows:
        reverse = lookup.get((row['kind'], row['target'], OPPOSITE[row['direction']]))
        # This reverse starts from independently collected contexts at the target.
        row['independent_reverse_target'] = reverse['target'] if reverse else None
        row['independent_reverse_agreement'] = reverse['agreement'] if reverse else None
        row['independent_reciprocal'] = reverse['target'] == row['source'] if reverse else None
    return rows


def probe_actions(model, probe, tokenizer, nodes, contexts, graph, layer, batch_size, device, dtype):
    import torch
    from .model import select_cache
    from .runtime import autocast

    neighbors = build_neighbor_table(graph)
    by_length = defaultdict(list)
    for context in contexts:
        by_length[len(context['prefix'])].append(context)
    directions = list(tokenizer.direction_ids)
    events = []
    model.eval()
    probe.eval()
    with torch.inference_mode():
        for length, group in sorted(by_length.items()):
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
                tokens = torch.tensor([[tokenizer.node(c['origin']), tokenizer.node(c['destination'])] +
                                       [tokenizer.direction_ids[a] for a in c['prefix']] for c in batch], device=device)
                with autocast(device, dtype):
                    initial = model(tokens, use_cache=True, probe_layer=layer)
                pre = probe(initial.hidden[:, -1].float()).softmax(-1).cpu().tolist()
                action_p = initial.logits[:, -1].float().softmax(-1).cpu().tolist()
                for context, probabilities in zip(batch, pre):
                    index = max(range(len(nodes)), key=probabilities.__getitem__)
                    context.update(pre_node=nodes[index], pre_confidence=probabilities[index])
                # Each independent branch receives the original prefix cache.
                branch_indices = torch.arange(len(batch), device=device).repeat_interleave(len(directions))
                actions = torch.tensor([tokenizer.direction_ids[a] for a in directions] * len(batch), device=device)
                inverse = torch.tensor([tokenizer.direction_ids[OPPOSITE[a]] for a in directions] * len(batch), device=device)
                with autocast(device, dtype):
                    after = model(actions[:, None], past=select_cache(initial.cache, branch_indices),
                                  use_cache=True, probe_layer=layer)
                probabilities = probe(after.hidden[:, -1].float()).softmax(-1).cpu().tolist()
                inverse_p = after.logits[:, -1].float().softmax(-1).gather(1, inverse[:, None]).squeeze(1).cpu().tolist()
                with autocast(device, dtype):
                    returned = model(inverse[:, None], past=after.cache, probe_layer=layer)
                return_p = probe(returned.hidden[:, -1].float()).softmax(-1).cpu().tolist()
                for i, context in enumerate(batch):
                    for j, direction in enumerate(directions):
                        branch = i * len(directions) + j
                        p, q = probabilities[branch], return_p[branch]
                        target, return_target = max(range(len(nodes)), key=p.__getitem__), max(range(len(nodes)), key=q.__getitem__)
                        events.append({'context_id': context['id'], 'direction': direction,
                                       'target': nodes[target], 'confidence': p[target], 'probabilities': p,
                                       'direction_probability': action_p[i][tokenizer.direction_ids[direction]],
                                       'true_neighbor': neighbors.get(context['source'], {}).get(direction),
                                       'inverse_direction': OPPOSITE[direction],
                                       'inverse_direction_probability': inverse_p[branch],
                                       'return_node': nodes[return_target], 'return_confidence': q[return_target]})
            print(f'Extracted {len(events)} action branches; prefix length {length}', flush=True)
    return events


def extract_run(run_path, output, contexts_per_node=30, batch_size=64, seed=42,
                splits=('seen', 'unseen'), include_generated=False, device='auto', precision='auto', sample_ids=None):
    import torch
    from .config import load_config
    from .model import GridTransformer, Tokenizer
    from .runtime import device_and_dtype, environment_info
    from .extract_report import render_extraction

    run_path, output = Path(run_path).resolve(), Path(output).resolve()
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Extraction output is nonempty; choose a new --output directory.')
    config = load_config(run_path / 'config.json')
    dataset = json_read(run_path / 'dataset.json')
    resolved_device, dtype = device_and_dtype({'device': device, 'precision': precision})
    torch.set_num_threads(config['train'].get('cpu_threads', 4))
    tokenizer = Tokenizer(config['data']['rows'] * config['data']['cols'], dataset['graph'].get('directions'))
    saved_model = torch.load(run_path / 'model.pt', map_location='cpu', weights_only=True)
    model = GridTransformer(tokenizer.vocab_size, config['model'], saved_model['context_length'])
    model.load_state_dict(saved_model['state_dict'])
    model.to(resolved_device)
    saved_probe = torch.load(run_path / 'probe.pt', map_location='cpu', weights_only=True)
    nodes = saved_probe['nodes']
    probe = torch.nn.Linear(config['model']['dim'], len(nodes))
    probe.load_state_dict(saved_probe['state_dict'])
    probe.to(resolved_device)
    include_generated = include_generated or bool(sample_ids)
    samples = json_read(run_path / 'samples.json') if include_generated else []
    if sample_ids:
        requested = set(sample_ids)
        missing = requested - {sample['id'] for sample in samples}
        if missing:
            raise ValueError(f'Unknown saved sample IDs: {sorted(missing)}')
        samples = [sample for sample in samples if sample['id'] in requested]
    contexts, coverage = collect_contexts(dataset, samples, contexts_per_node, seed, model.context_length, splits)
    if not contexts:
        raise ValueError('No eligible contexts; check split sizes and checkpoint context length.')
    events = probe_actions(model, probe, tokenizer, nodes, contexts, dataset['graph'],
                          saved_probe['layer'], batch_size, resolved_device, dtype)
    covered = {c['source'] for c in contexts if c['kind'] == 'reference'}
    coverage['missing_reference_sources'] = sorted(set(dataset['graph']['nodes']) - covered)
    known = [c for c in contexts if c['known_source'] is not None]
    coverage['pre_probe_accuracy_known_contexts'] = (sum(c['pre_node'] == c['known_source'] for c in known) / len(known)
                                                     if known else None)
    rows = summarize_transitions(contexts, events, nodes)
    matched = summarize_transitions(contexts, events, nodes, require_correct_source=True)
    def digest(path):
        hasher = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    payload = {'graph': dataset['graph'], 'probe_nodes': nodes, 'contexts': contexts,
               'events': events, 'transitions': rows, 'source_correct_transitions': matched,
               'coverage': coverage, 'settings': {'run': str(run_path), 'contexts_per_node': contexts_per_node,
               'seed': seed, 'splits': list(splits), 'include_generated': include_generated,
               'sample_ids': list(sample_ids or []), 'batch_size': batch_size, 'probe_layer': saved_probe['layer']},
               'provenance': {name: digest(run_path / name) for name in ('model.pt', 'probe.pt', 'dataset.json', 'config.json')},
               'implementation': {name: digest(Path(__file__).parent / name)
                                  for name in ('extract.py', 'model.py', 'runtime.py')},
               'environment': environment_info(resolved_device, dtype),
               'interpretation': 'Forced-action probe readouts. Illegal branches have no true arrival label. '
               'Inverse return can reflect history cancellation, not a stable map. Independent reverse tests '
               'use separate target-node contexts. Generated-invalid sources are inferred, not known.'}
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / 'extraction.json', payload)
    export_csv(output / 'transitions.csv', rows)
    export_csv(output / 'source_correct_transitions.csv', matched)
    export_csv(output / 'contexts.csv', contexts)
    export_csv(output / 'branches.csv', events)
    render_extraction(payload, output / 'extraction.html')
    print(f'Extraction: {output / "extraction.html"}', flush=True)
    return payload
