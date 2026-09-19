"""Command-line entry point: python -m gridworld --help."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import statistics
import sys
from pathlib import Path

from .config import load_config, set_override, validate_config
from .pipeline import create_report, export_csv, fingerprint, json_read, json_write, prepare, run


def sweep(spec_path, output, resume=False, limit=None):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    spec = json_read(spec_path)
    base = load_config(spec_path.parent / spec['base_config'])
    parameters = spec.get('parameters', {})
    if not isinstance(parameters, dict) or any(not isinstance(values, list) or not values for values in parameters.values()):
        raise ValueError('Each sweep parameter must have a nonempty JSON list of candidate values.')
    seeds = spec.get('seeds', [None])
    if not isinstance(seeds, list) or not seeds or any((seed is None and 'seeds' in spec) or
            (seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0)) for seed in seeds):
        raise ValueError('sweep.seeds must be a nonempty list of nonnegative integers.')
    if 'seeds' in spec and any(key.endswith('.seed') for key in parameters):
        raise ValueError('Use either sweep.seeds or explicit seed parameters, not both.')
    if len(seeds) != len(set(seeds)):
        raise ValueError('Repeated seeds would duplicate an experiment. Use unique seed values.')
    manifest = {'base_config': base, 'parameters': parameters, 'seeds': seeds}
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError('Sweep output is nonempty. Choose a new directory or use --resume.')
        if not (output / 'sweep.json').exists() or json_read(output / 'sweep.json') != manifest:
            raise ValueError('Sweep definition changed. Use a new output directory; existing summaries are preserved.')
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / 'sweep.json', manifest)
    total = len(seeds)
    for values in parameters.values():
        total *= len(values)
    print(f'Sweep: {min(total, limit) if limit is not None else total} configurations; runs are sequential on one GPU.', flush=True)
    rows, seen_hashes = [], set()
    combinations = itertools.product(*parameters.values())
    for values in combinations:
        changes = dict(zip(parameters, values))
        group = fingerprint(changes)[:12]
        for seed in seeds:
            if limit is not None and len(rows) >= limit:
                break
            cfg = copy.deepcopy(base)
            for key, value in changes.items():
                set_override(cfg, key, value)
            if seed is not None:
                for section, offset in (('data', 0), ('train', 0), ('probe', 1), ('generation', 2)):
                    cfg[section]['seed'] = seed + offset
            validate_config(cfg)
            tag = fingerprint(cfg)[:12]
            if tag in seen_hashes:
                continue
            seen_hashes.add(tag)
            run_path = output / f'run-{tag}'
            cfg['output']['directory'] = str(run_path)
            row = {'run_id': tag, 'group': group, 'seed': seed, 'output': str(run_path), **changes}
            try:
                result = run(cfg, run_path, resume=resume)
                row['status'] = 'complete'
                for cohort, metrics in result.items():
                    row.update({f'{cohort}.{key}': value for key, value in metrics['summary'].items()
                                if isinstance(value, (int, float)) or value is None})
            except Exception as error:
                row.update({'status': 'failed', 'error': str(error)})
                print(f'Sweep run failed: {error}', file=sys.stderr, flush=True)
            rows.append(row)
            json_write(output / 'summary.json', rows)
            export_csv(output / 'summary.csv', rows)
        if limit is not None and len(rows) >= limit:
            break
    aggregates = []
    for group in sorted({row['group'] for row in rows}):
        complete = [row for row in rows if row['group'] == group and row['status'] == 'complete']
        numeric_keys = sorted({key for row in complete for key in row if key.startswith(('all.', 'seen.', 'unseen.'))})
        for key in numeric_keys:
            values = [row[key] for row in complete if isinstance(row.get(key), (int, float))]
            if values:
                aggregates.append({'group': group, 'metric': key, 'n': len(values), 'mean': statistics.mean(values),
                                   'std': statistics.stdev(values) if len(values) > 1 else None})
    json_write(output / 'aggregate.json', aggregates)
    export_csv(output / 'aggregate.csv', aggregates)
    if any(row['status'] == 'failed' for row in rows):
        raise RuntimeError('Some sweep runs failed; inspect summary.json for errors. Completed runs were preserved.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Standalone grid navigation, location probing, and map reconstruction.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('run', 'Train, probe, reconstruct, and render an experiment.'),
                            ('prepare', 'Generate and audit the data only; no PyTorch required.')]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument('--config', type=Path, help='Partial JSON configuration; defaults use the 12-layer paper architecture.')
        command.add_argument('--output', type=Path, help='Artifact directory (overrides output.directory).')
        command.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='Repeatable dotted config override; value is JSON or a bare string.')
        command.add_argument('--resume', action='store_true', help='Reuse completed stages only if the resolved config matches exactly.')
    report = commands.add_parser('report', help='Regenerate the offline viewer from an existing run.')
    report.add_argument('--run', required=True, type=Path)
    sweep_parser = commands.add_parser('sweep', help='Run a Cartesian parameter/seed sweep sequentially.')
    sweep_parser.add_argument('--spec', required=True, type=Path)
    sweep_parser.add_argument('--output', required=True, type=Path)
    sweep_parser.add_argument('--resume', action='store_true')
    sweep_parser.add_argument('--limit', type=int, help='Run only the first N configurations for a pilot.')
    args = parser.parse_args(argv)
    try:
        if args.command in {'run', 'prepare'}:
            cfg = load_config(args.config, args.set)
            output = (args.output or Path(cfg['output']['directory'])).resolve()
            cfg['output']['directory'] = str(output)
            (run if args.command == 'run' else prepare)(cfg, output, args.resume)
        elif args.command == 'report':
            print(create_report(args.run))
        else:
            if args.limit is not None and args.limit < 1:
                raise ValueError('--limit must be positive.')
            sweep(args.spec, args.output, args.resume, args.limit)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
