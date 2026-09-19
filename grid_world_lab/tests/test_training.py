import copy
import unittest
from unittest.mock import patch

import torch

from gridworld.config import load_config
from gridworld.model import GridTransformer, Tokenizer
from gridworld.train import train_model


class TrainingTests(unittest.TestCase):
    def test_accumulation_weights_unequal_walk_lengths_by_valid_tokens(self):
        torch.set_num_threads(2)
        cfg = load_config(overrides=['model.layers=1', 'model.dim=16', 'model.heads=2',
                                     'model.dropout=0', 'train.epochs=1', 'train.max_steps=1',
                                     'train.device=cpu', 'train.weight_decay=0'])
        tokenizer = Tokenizer(4)
        torch.manual_seed(12)
        original = GridTransformer(tokenizer.vocab_size, cfg['model'], 10)
        dataset = {'splits': {'train': [
            {'origin': 1, 'destination': 2, 'directions': ['E'], 'nodes': [1, 2]},
            {'origin': 1, 'destination': 4, 'directions': ['E', 'S', 'N', 'S'], 'nodes': [1, 2, 4, 2, 4]}],
            'validation': []}}

        def fit(batch, accumulation):
            model = copy.deepcopy(original)
            train_cfg = {**cfg['train'], 'batch_size': batch, 'gradient_accumulation': accumulation}
            saved = {}

            def capture(path, value):
                saved.update(copy.deepcopy(value))

            with patch('gridworld.train.save_torch', side_effect=capture), \
                 patch('gridworld.train.save_json'), patch('gridworld.train.torch.load', side_effect=lambda *a, **k: saved):
                result = train_model(model, dataset, tokenizer, train_cfg, torch.device('cpu'), torch.float32, '.')
            self.assertEqual(result['optimization_steps'], 1)
            self.assertEqual(result['training_tokens_processed'], 9)
            return model.state_dict()

        together, accumulated = fit(2, 1), fit(1, 2)
        for name in together:
            torch.testing.assert_close(together[name], accumulated[name], atol=2e-7, rtol=2e-6, msg=name)


if __name__ == '__main__':
    unittest.main()
