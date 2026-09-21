import copy
import unittest

import torch

from gridworld.extract import collect_contexts, probe_actions
from gridworld.model import GridTransformer, Tokenizer
from tests.test_extract import fixture


class BranchTests(unittest.TestCase):
    def test_cached_batched_branches_match_independent_full_prefix_forwards(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        dataset = fixture()
        tokenizer = Tokenizer(4, dataset['graph']['directions'])
        model = GridTransformer(tokenizer.vocab_size, {'layers': 2, 'dim': 16, 'heads': 2, 'dropout': 0}, 10).eval()
        probe = torch.nn.Linear(16, 4).eval()
        contexts, _ = collect_contexts(dataset, [], 20, 42, 10, ['seen'])
        events = probe_actions(model, probe, tokenizer, [1, 2, 3, 4], contexts,
                               dataset['graph'], -1, 2, torch.device('cpu'), torch.float32)
        self.assertEqual(len(events), len(contexts) * 4)
        lookup = {c['id']: c for c in contexts}
        with torch.inference_mode():
            for event in events:
                context = lookup[event['context_id']]
                base = [tokenizer.node(context['origin']), tokenizer.node(context['destination'])] + [tokenizer.direction_ids[a] for a in context['prefix']]
                forced = base + [tokenizer.direction_ids[event['direction']]]
                returned = forced + [tokenizer.direction_ids[event['inverse_direction']]]
                p = probe(model(torch.tensor([forced])).hidden[:, -1]).softmax(-1)[0]
                q = probe(model(torch.tensor([returned])).hidden[:, -1]).softmax(-1)[0]
                torch.testing.assert_close(p, torch.tensor(event['probabilities']), atol=1e-6, rtol=1e-5)
                self.assertEqual(int(q.argmax()) + 1, event['return_node'])
                initial = model(torch.tensor([base])).logits[0, -1].softmax(-1)
                self.assertAlmostEqual(initial[tokenizer.direction_ids[event['direction']]].item(), event['direction_probability'], places=6)
        second = probe_actions(model, probe, tokenizer, [1, 2, 3, 4], copy.deepcopy(contexts),
                               dataset['graph'], -1, 1, torch.device('cpu'), torch.float32)
        for first, other in zip(events, second):
            self.assertEqual(first['target'], other['target'])
            self.assertEqual(first['return_node'], other['return_node'])


if __name__ == '__main__':
    unittest.main()
