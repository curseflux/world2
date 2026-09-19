import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from gridworld.__main__ import sweep
from gridworld.config import load_config
from gridworld.pipeline import prepare


class PipelineTests(unittest.TestCase):
    def test_resume_rejects_changed_config_before_touching_data(self):
        cfg = load_config()
        old = copy.deepcopy(cfg)
        old['data']['rows'] += 1
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'iterdir', return_value=iter([Path('config.json')])), \
             patch('gridworld.pipeline.json_read', return_value=old), \
             patch('gridworld.pipeline.json_write') as write, \
             patch('gridworld.pipeline.build_dataset') as build:
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                prepare(cfg, 'runs/test', resume=True)
            write.assert_not_called()
            build.assert_not_called()

    def test_new_run_preserves_existing_output(self):
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'iterdir', return_value=iter([Path('model.pt')])):
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                prepare(load_config(), 'runs/test')

    def test_sweep_expands_seeds_and_aggregates_only_comparable_runs(self):
        spec = {'base_config': 'tiny.json', 'parameters': {'model.layers': [1, 2]}, 'seeds': [11, 12]}
        base = load_config(Path(__file__).parents[1] / 'configs' / 'tiny.json')
        writes, configs = {}, []

        def run_stub(cfg, path, resume):
            configs.append(copy.deepcopy(cfg))
            return {'seen': {'summary': {'edge_recall': cfg['data']['seed'] / 100}}}

        with patch('gridworld.__main__.json_read', return_value=spec), \
             patch('gridworld.__main__.load_config', return_value=base), \
             patch.object(Path, 'exists', return_value=False), patch.object(Path, 'mkdir'), \
             patch('gridworld.__main__.json_write', side_effect=lambda path, value: writes.update({path.name: copy.deepcopy(value)})), \
             patch('gridworld.__main__.export_csv'), patch('gridworld.__main__.run', side_effect=run_stub):
            sweep('spec.json', 'runs/test-sweep')
        self.assertEqual(len(configs), 4)
        self.assertEqual([cfg['model']['layers'] for cfg in configs], [1, 1, 2, 2])
        for cfg in configs:
            self.assertEqual(cfg['probe']['seed'], cfg['data']['seed'] + 1)
            self.assertEqual(cfg['generation']['seed'], cfg['data']['seed'] + 2)
        self.assertEqual(len(writes['aggregate.json']), 2)
        self.assertTrue(all(row['n'] == 2 for row in writes['aggregate.json']))
        self.assertTrue(all(abs(row['mean'] - 0.115) < 1e-8 for row in writes['aggregate.json']))

    def test_sweep_preserves_manifest_and_summary_for_changed_definition(self):
        spec = {'base_config': 'tiny.json', 'seeds': [42]}
        with patch('gridworld.__main__.json_read', side_effect=[spec, {'wrong': 'manifest'}]), \
             patch('gridworld.__main__.load_config', return_value=load_config()), \
             patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'iterdir', return_value=iter([Path('summary.json')])), \
             patch('gridworld.__main__.json_write') as write:
            with self.assertRaisesRegex(ValueError, 'definition changed'):
                sweep('spec.json', 'runs/test', resume=True)
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
