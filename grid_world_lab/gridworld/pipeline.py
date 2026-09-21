"""Stages and auditable, reusable experiment artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .config import validate_config
from .data import build_dataset


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()


def implementation_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ('config.py', 'data.py', 'model.py', 'train.py', 'probe.py', 'reconstruct.py', 'runtime.py'):
        digest.update(name.encode('utf-8'))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def prepare(config, output, resume=False):
    """Generate data without importing the GPU runtime."""
    validate_config(config)
    output = Path(output).resolve()
    config_path = output / 'config.json'
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError(f'{output} is nonempty. Choose a new --output directory or use --resume with identical configuration.')
        if not config_path.exists() or json_read(config_path) != config:
            raise ValueError('Resume configuration differs from config.json. Use a fresh output directory for a new experiment.')
    output.mkdir(parents=True, exist_ok=True)
    json_write(config_path, config)
    if resume and (output / 'dataset.json').exists():
        dataset = json_read(output / 'dataset.json')
    else:
        dataset = build_dataset(config['data'])
        json_write(output / 'dataset.json', dataset)
    for warning in dataset.get('split_warnings', []):
        print(f'DATA WARNING: {warning}', flush=True)
    sampling = dataset.get('sampling', {})
    mode_text = f"mode={sampling.get('mode', 'union')}, map_samples={sampling.get('map_samples', len(dataset['splits']['train']))}; "
    print(f"Data: {mode_text}{len(dataset['graph']['nodes'])} nodes, {len(dataset['graph']['edges'])} undirected edges; "
          + ', '.join(f'{key}={len(routes)}' for key, routes in dataset['splits'].items()), flush=True)
    coverage = dataset['graph'].get('training_coverage')
    if coverage:
        print(f"Coverage: origins={coverage['unique_origins']}/{len(dataset['graph']['nodes'])}, "
              f"directed_edges={coverage['observed_directed_edges']}/{coverage['legal_directed_edges']}, "
              f"direction_steps={coverage['direction_steps']}", flush=True)
    return dataset


def export_csv(path, rows):
    rows = list(rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_evaluation(output, dataset, samples, config):
    from .metrics import summarize
    metrics = {}
    for cohort in ('seen', 'unseen', 'all'):
        subset = samples if cohort == 'all' else [sample for sample in samples if sample['cohort'] == cohort]
        metrics[cohort] = summarize(dataset['graph'], subset, config['metrics'])
        metrics[cohort]['summary']['exact_training_route_match_rate'] = (
            sum(s['exact_training_route_match'] for s in subset) / len(subset) if subset else None)
    json_write(output / 'metrics.json', metrics)
    export_csv(output / 'events.csv', ({'sample_id': sample['id'], 'cohort': sample['cohort'],
                                      'origin': sample['origin'], 'destination': sample['destination'], **event}
                                     for sample in samples for event in sample['events']))
    export_csv(output / 'sample_metrics.csv', ({key: value for key, value in sample.items()
                                               if key not in {'events', 'reference_nodes', 'reference_directions', 'generated_tokens', 'generated_directions'}}
                                              for sample in samples))
    export_csv(output / 'edge_metrics.csv', ({'cohort': cohort, **row} for cohort, result in metrics.items() for row in result['edge_rows']))
    export_csv(output / 'node_metrics.csv', ({'cohort': cohort, **row} for cohort, result in metrics.items() for row in result['node_rows']))
    return metrics


def create_report(output):
    from .report import render_report
    output = Path(output).resolve()
    dataset = json_read(output / 'dataset.json')
    payload = {'config': json_read(output / 'config.json'),
               'dataset': {key: value for key, value in dataset.items() if key != 'splits'},
               'training': json_read(output / 'training.json'), 'probe': json_read(output / 'probe.json'),
               'samples': json_read(output / 'samples.json'), 'metrics': json_read(output / 'metrics.json'),
               'environment': json_read(output / 'environment.json') if (output / 'environment.json').exists() else {}}
    json_write(output / 'report.json', payload)
    render_report(payload, output / 'report.html')
    return output / 'report.html'


def run(config, output, resume=False):
    import torch
    from .model import GridTransformer, Tokenizer
    from .probe import fit_probe
    from .reconstruct import generate_samples
    from .runtime import device_and_dtype, environment_info, seed_everything
    from .train import train_model

    output = Path(output).resolve()
    dataset = prepare(config, output, resume)
    if not dataset['splits']['probe_train']:
        raise ValueError('The frozen graph has no unused probe-training routes. Review dataset.json warnings; increase max_length or reduce held-out counts.')
    device, dtype = device_and_dtype(config['train'])
    torch.set_num_threads(config['train'].get('cpu_threads', 4))
    seed_everything(config['train']['seed'])
    tokenizer = Tokenizer(config['data']['rows'] * config['data']['cols'])
    needed_length = max(config['data']['max_length'] + 3,
                        (config['data'].get('heldout_max_length') or config['data']['max_length']) + 3,
                        config['generation']['max_new_tokens'] + 2)
    context_length = config['model'].get('context_length') or needed_length
    if context_length < needed_length:
        raise ValueError(f'model.context_length={context_length} is too small; need at least {needed_length}.')
    environment = environment_info(device, dtype)
    environment.update({'config_sha256': fingerprint(config), 'dataset_sha256': fingerprint(dataset),
                        'implementation_sha256': implementation_fingerprint(),
                        'resolved_context_length': context_length})
    reuse_model = resume and (output / 'training.json').exists() and (output / 'model.pt').exists()
    if reuse_model and (output / 'environment.json').exists():
        original_environment = json_read(output / 'environment.json')
        for field in ('dataset_sha256', 'implementation_sha256'):
            if field in original_environment and original_environment[field] != environment[field]:
                raise ValueError(f'Resume {field} differs from the saved run. Use a new output directory, or report --run to only rebuild its viewer.')
        # Preserve the hardware/library provenance of the checkpoint being reused.
    else:
        json_write(output / 'environment.json', environment)
    print(f"Runtime: {environment['device']}, {environment['precision']}, {environment['gpu'] or 'CPU'}", flush=True)
    model = GridTransformer(tokenizer.vocab_size, config['model'], context_length).to(device)
    if reuse_model:
        model.load_state_dict(torch.load(output / 'model.pt', map_location=device, weights_only=True)['state_dict'])
    else:
        train_model(model, dataset, tokenizer, config['train'], device, dtype, output)
    reuse_probe = reuse_model and (output / 'probe.json').exists() and (output / 'probe.pt').exists()
    if reuse_probe:
        saved = torch.load(output / 'probe.pt', map_location=device, weights_only=True)
        probe_nodes = saved['nodes']
        probe = torch.nn.Linear(config['model']['dim'], len(probe_nodes)).to(device)
        probe.load_state_dict(saved['state_dict'])
    else:
        probe, probe_nodes, _ = fit_probe(model, dataset, tokenizer, config['probe'], device, dtype, output)
    if reuse_probe and (output / 'samples.json').exists():
        samples = json_read(output / 'samples.json')
    else:
        samples = generate_samples(model, probe, probe_nodes, dataset, tokenizer, config, device, dtype)
        json_write(output / 'samples.json', samples)
    metrics = write_evaluation(output, dataset, samples, config)
    report_path = create_report(output)
    json_write(output / 'complete.json', {'config_sha256': fingerprint(config), 'report': str(report_path)})
    print(f'Report: {report_path}', flush=True)
    return metrics
