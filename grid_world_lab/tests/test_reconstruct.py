"""End-to-end semantic tests with deterministic model states and token outputs."""
import copy
import unittest

import torch

from gridworld.config import load_config
from gridworld.model import ModelOutput, Tokenizer
from gridworld.reconstruct import generate_samples, sample_tokens


class ScriptedModel:
    def __init__(self, tokenizer, scripts, states):
        self.tokenizer, self.scripts, self.states = tokenizer, scripts, states
        self.histories = {}

    def eval(self):
        return self

    def __call__(self, tokens, past=None, use_cache=False, probe_layer=-1):
        if past is None:
            origins = (tokens[:, 0] - self.tokenizer.node_offset + 1).tolist()
            step = 0
            for origin, row in zip(origins, tokens.tolist()):
                self.histories[origin] = row
        else:
            origins = past[0][0][:, 0, 0, 0].long().tolist()
            step = past[0][0].size(-2) - 1
            for origin, row in zip(origins, tokens.tolist()):
                self.histories[origin] += row
        logits = torch.full((len(origins), tokens.size(1), self.tokenizer.vocab_size), -30.0)
        hidden = torch.full((len(origins), tokens.size(1), 4), -20.0)
        for i, origin in enumerate(origins):
            next_token = self.scripts[origin][step]
            logits[i, :, next_token] = 30
            hidden[i, :, self.states[origin][step] - 1] = 20
        cache = torch.tensor(origins, dtype=torch.float32)[:, None, None, None].expand(-1, 1, step + 2, 1)
        return ModelOutput(logits, hidden, [(cache, cache)])


class ReconstructionTests(unittest.TestCase):
    def fixture(self):
        cfg = load_config(overrides=['generation.temperature=0', 'generation.max_new_tokens=8',
                                     'generation.save_full_probe_probabilities=true'])
        tokenizer = Tokenizer(4)
        encode = lambda seq: [tokenizer.eos_id if action == 'EOS' else tokenizer.direction_ids[action] for action in seq]
        model = ScriptedModel(tokenizer, {1: encode(['N', 'W', 'SW', 'E', 'EOS']), 2: encode(['S', 'EOS'])},
                              {1: [1, 4, 2, 3, 4], 2: [2, 4]})
        routes = [{'id': 'seen-1', 'origin': 1, 'destination': 4, 'directions': ['E', 'S'], 'nodes': [1, 2, 4]},
                  {'id': 'seen-2', 'origin': 2, 'destination': 4, 'directions': ['S'], 'nodes': [2, 4]}]
        dataset = {'graph': {'rows': 2, 'cols': 2, 'nodes': [1, 2, 3, 4], 'edges': [[1, 2], [2, 4], [3, 4]]},
                   'splits': {'train': [routes[1]], 'seen': routes, 'unseen': []}}
        return cfg, tokenizer, model, dataset

    def test_probe_priority_consumes_bad_moves_and_keeps_uncorrected_history(self):
        cfg, tokenizer, model, dataset = self.fixture()
        original = copy.deepcopy(dataset)
        samples = generate_samples(model, torch.nn.Identity(), [1, 2, 3, 4], dataset, tokenizer,
                                   cfg, torch.device('cpu'), torch.float32)
        first, second = samples
        self.assertEqual([e['kind'] for e in first['events']], ['illegal', 'legal_mismatch', 'illegal', 'legal_correct'])
        self.assertEqual([e['source'] for e in first['events']], [1, 4, 2, 3])
        self.assertEqual([e['target'] for e in first['events']], [4, 2, 3, 4])
        self.assertEqual(first['events'][1]['expected_target'], 3)
        self.assertEqual(first['termination'], 'eos')
        self.assertEqual(first['generated_length'], 4)
        self.assertFalse(first['physical_valid'])
        self.assertFalse(first['destination_reached'])
        self.assertTrue(first['probe_destination_reached'])
        self.assertTrue(all(event['physical_target'] is None for event in first['events']))
        self.assertEqual(model.histories[1], [tokenizer.node(1), tokenizer.node(4)] +
                         [tokenizer.direction_ids[d] for d in ['N', 'W', 'SW', 'E']])
        self.assertTrue(second['physical_valid'])
        self.assertTrue(second['route_success'])
        self.assertTrue(second['exact_training_route_match'])
        self.assertEqual(len(second['events']), 1)  # Finished row is removed from KV cache.
        self.assertAlmostEqual(sum(first['events'][0]['probe_probabilities'].values()), 1)
        self.assertEqual(dataset, original)

    def test_token_cap_does_not_prevent_consuming_final_action(self):
        cfg, tokenizer, model, dataset = self.fixture()
        cfg['generation']['max_new_tokens'] = 1
        samples = generate_samples(model, torch.nn.Identity(), [1, 2, 3, 4], dataset, tokenizer,
                                   cfg, torch.device('cpu'), torch.float32)
        self.assertEqual(samples[0]['termination'], 'max_new_tokens')
        self.assertEqual(samples[0]['inferred_final_node'], 4)
        self.assertEqual(len(samples[0]['events']), 1)
        self.assertFalse(samples[1]['exact_training_route_match'])  # No EOS was generated.

    def test_invalid_node_token_stops_without_drawing_a_spatial_edge(self):
        cfg, tokenizer, model, dataset = self.fixture()
        model.scripts[1][0] = tokenizer.node(3)
        samples = generate_samples(model, torch.nn.Identity(), [1, 2, 3, 4], dataset, tokenizer,
                                   cfg, torch.device('cpu'), torch.float32)
        self.assertEqual(samples[0]['termination'], 'invalid_token')
        self.assertEqual(samples[0]['events'], [])
        self.assertTrue(samples[1]['route_success'])

    def test_syntax_mask_does_not_mask_illegal_directions(self):
        tokenizer = Tokenizer(4)
        logits = torch.zeros(1, tokenizer.vocab_size)
        logits[0, tokenizer.node(1)] = 100
        logits[0, tokenizer.direction_ids['NW']] = 50
        cfg = {'syntax_only': True, 'temperature': 0, 'top_k': 0}
        choice, _ = sample_tokens(logits, cfg, tokenizer, torch.Generator())
        self.assertEqual(choice.item(), tokenizer.direction_ids['NW'])


if __name__ == '__main__':
    unittest.main()
