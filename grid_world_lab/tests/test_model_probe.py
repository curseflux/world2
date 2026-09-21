"""Behavioral checks for causal decoding, padding, and probe state alignment."""

from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from gridworld.model import GridTransformer, Tokenizer, select_cache
from gridworld.probe import evaluate_probe, extract_features
from gridworld.train import collate_routes, evaluate_loss, targets_for


def make_model():
    torch.manual_seed(7)
    tokenizer = Tokenizer(9)
    config = {"layers": 2, "dim": 32, "heads": 4, "dropout": 0.0, "mlp_ratio": 2}
    return GridTransformer(tokenizer.vocab_size, config, context_length=20).eval(), tokenizer


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_future_tokens_cannot_change_prefix_states_or_logits(self):
        model, tok = make_model()
        prefix = [tok.node(1), tok.node(9), tok.direction_ids["E"]]
        a = torch.tensor([prefix + [tok.direction_ids["S"], tok.direction_ids["SE"], tok.eos_id]])
        b = torch.tensor([prefix + [tok.direction_ids["NW"], tok.direction_ids["W"], tok.pad_id]])
        with torch.no_grad():
            first, second = model(a), model(b)
        torch.testing.assert_close(first.hidden[:, :3], second.hidden[:, :3], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first.logits[:, :3], second.logits[:, :3], atol=1e-6, rtol=1e-6)

    def test_prefix_state_gradient_does_not_reach_future_token_embeddings(self):
        model, tok = make_model()
        # All input token IDs are distinct, so a future embedding gradient cannot
        # be attributed to an earlier occurrence of the same vocabulary item.
        tokens = torch.tensor([[tok.node(1), tok.node(9), tok.direction_ids["E"], tok.direction_ids["S"]]])
        model(tokens).hidden[0, 1, 0].backward()
        gradients = model.token_embedding.weight.grad
        self.assertGreater(gradients[tok.node(1)].abs().sum().item(), 0)
        self.assertEqual(0, gradients[tok.direction_ids["E"]].abs().sum().item())
        self.assertEqual(0, gradients[tok.direction_ids["S"]].abs().sum().item())

    def test_cached_single_token_and_chunked_decoding_match_full_sequence(self):
        model, tok = make_model()
        tokens = torch.tensor([[tok.node(1), tok.node(9), tok.direction_ids["E"],
                                tok.direction_ids["S"], tok.direction_ids["NE"],
                                tok.direction_ids["W"], tok.eos_id]])
        for probe_layer in (-1, 0, 1):
            with self.subTest(probe_layer=probe_layer), torch.no_grad():
                full = model(tokens, probe_layer=probe_layer)
                for lengths in ([2, 1, 1, 1, 1, 1], [2, 3, 2]):
                    cache, offset, logits, hidden = None, 0, [], []
                    for length in lengths:
                        result = model(tokens[:, offset:offset + length], past=cache,
                                       use_cache=True, probe_layer=probe_layer)
                        cache = result.cache
                        logits.append(result.logits)
                        hidden.append(result.hidden)
                        offset += length
                    torch.testing.assert_close(torch.cat(logits, dim=1), full.logits, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(torch.cat(hidden, dim=1), full.hidden, atol=2e-6, rtol=1e-5)
                    self.assertEqual(tokens.shape[1], cache[0][0].shape[-2])

    def test_cache_row_selection_keeps_each_surviving_route_history(self):
        model, tok = make_model()
        tokens = torch.tensor([
            [tok.node(1), tok.node(9), tok.direction_ids["E"], tok.direction_ids["S"]],
            [tok.node(2), tok.node(8), tok.direction_ids["SW"], tok.direction_ids["N"]],
            [tok.node(3), tok.node(7), tok.direction_ids["W"], tok.direction_ids["SE"]],
        ])
        keep = torch.tensor([2, 0])
        with torch.no_grad():
            first = model(tokens[:, :2], use_cache=True)
            continuation = model(tokens[keep, 2:], past=select_cache(first.cache, keep), use_cache=True)
            full = model(tokens[keep])
        torch.testing.assert_close(continuation.hidden, full.hidden[:, 2:], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(continuation.logits, full.logits[:, 2:], atol=1e-6, rtol=1e-5)

    def test_context_limit_applies_to_cached_total_length(self):
        model, tok = make_model()
        prefix = torch.full((1, model.context_length), tok.node(1), dtype=torch.long)
        with torch.no_grad():
            first = model(prefix, use_cache=True)
            with self.assertRaisesRegex(ValueError, "context_length"):
                model(prefix[:, :1], past=first.cache, use_cache=True)

    def test_cardinal_tokenizer_has_only_four_action_tokens(self):
        tokenizer = Tokenizer(9, ["N", "E", "S", "W"])
        self.assertEqual({"N", "E", "S", "W"}, set(tokenizer.direction_ids))
        self.assertEqual(6, tokenizer.node_offset)
        self.assertEqual(15, tokenizer.vocab_size)
        self.assertNotIn("NE", tokenizer.direction_ids)


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.model, self.tokenizer = make_model()
        self.routes = [
            {"origin": 1, "destination": 2, "directions": ["E"], "nodes": [1, 2]},
            {"origin": 1, "destination": 8, "directions": ["SE", "E", "SW"], "nodes": [1, 5, 6, 8]},
        ]

    def test_padding_targets_have_no_loss_or_gradient(self):
        tokens = collate_routes(self.routes, self.tokenizer)
        targets = targets_for(tokens, self.tokenizer)
        self.assertEqual(8, int((targets != -100).sum()))
        self.assertEqual([-100, -100], targets[0, -2:].tolist())
        logits = torch.randn((*targets.shape, self.tokenizer.vocab_size), requires_grad=True)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum")
        valid = targets != -100
        expected = F.cross_entropy(logits[valid], targets[valid], reduction="sum")
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(0, logits.grad[~valid].abs().sum().item())

    def test_prompt_loss_switch_excludes_only_destination_target(self):
        tokens = collate_routes(self.routes, self.tokenizer)
        targets = targets_for(tokens, self.tokenizer, loss_on_prompt=False)
        self.assertTrue((targets[:, 0] == -100).all())
        self.assertEqual(6, int((targets != -100).sum()))
        # The first action and the terminating EOS remain supervised.
        self.assertEqual(self.tokenizer.direction_ids["E"], targets[0, 1].item())
        self.assertEqual(self.tokenizer.eos_id, targets[0, 2].item())

    def test_loss_is_invariant_to_batch_padding(self):
        config = {"batch_size": 2, "num_workers": 0, "device": "cpu", "loss_on_prompt": True}
        device = torch.device("cpu")
        combined = evaluate_loss(self.model, self.routes, self.tokenizer, config, device, torch.float32)
        singles = [evaluate_loss(self.model, [route], self.tokenizer, config, device, torch.float32) for route in self.routes]
        expected = sum(row["loss"] * row["tokens"] for row in singles) / sum(row["tokens"] for row in singles)
        self.assertAlmostEqual(expected, combined["loss"], places=6)
        self.assertEqual(8, combined["tokens"])
        self.assertEqual(6, combined["action_tokens"])

    def test_legal_action_metric_excludes_destination_prediction(self):
        config = {"batch_size": 2, "num_workers": 0, "device": "cpu", "loss_on_prompt": True}
        graph = {"rows": 3, "cols": 3, "nodes": [1, 2, 5, 6, 8],
                 "edges": [[1, 2], [1, 5], [5, 6], [6, 8]]}
        result = evaluate_loss(self.model, self.routes, self.tokenizer, config,
                               torch.device("cpu"), torch.float32, graph)
        self.assertEqual(6, result["legal_action_tokens"])
        self.assertEqual(6, result["action_tokens"])
        self.assertIsNotNone(result["legal_action_accuracy"])
        self.assertGreaterEqual(result["legal_action_accuracy"], 0)
        self.assertLessEqual(result["legal_action_accuracy"], 1)


class SentinelModel:
    """Expose consumed token ID and position so off-by-one errors are visible."""
    config = {"dim": 2}

    def eval(self):
        return self

    def __call__(self, tokens, probe_layer=-1):
        positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        return SimpleNamespace(hidden=torch.stack((tokens.float(), positions.float()), dim=-1))


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tokenizer = Tokenizer(9)
        self.routes = [
            {"origin": 1, "destination": 5, "directions": ["E", "S"], "nodes": [1, 2, 5]},
            {"origin": 9, "destination": 6, "directions": ["N"], "nodes": [9, 6]},
        ]
        self.nodes = [1, 2, 5, 6, 9]
        self.config = {"seed": 11, "extraction_batch_size": 2, "layer": -1}

    def test_labels_align_to_prompt_end_and_consumed_action(self):
        x, y = extract_features(SentinelModel(), self.routes, self.tokenizer, self.nodes,
                                self.config, torch.device("cpu"), torch.float32)
        expected_tokens = [self.tokenizer.node(5), self.tokenizer.direction_ids["E"], self.tokenizer.direction_ids["S"],
                           self.tokenizer.node(6), self.tokenizer.direction_ids["N"]]
        self.assertEqual(expected_tokens, x[:, 0].long().tolist())
        self.assertEqual([1, 2, 3, 1, 2], x[:, 1].long().tolist())
        self.assertEqual([1, 2, 5, 9, 6], [self.nodes[index] for index in y.tolist()])
        self.assertNotIn(self.tokenizer.eos_id, x[:, 0].long().tolist())
        self.assertNotIn(self.tokenizer.pad_id, x[:, 0].long().tolist())

    def test_feature_subsampling_keeps_labels_and_is_independent_of_batch_size(self):
        outputs = []
        for batch_size in (1, 2):
            cfg = {**self.config, "extraction_batch_size": batch_size}
            outputs.append(extract_features(SentinelModel(), self.routes, self.tokenizer, self.nodes,
                                            cfg, torch.device("cpu"), torch.float32, max_positions=3))
        for first, second in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(first, second)
        full_x, full_y = extract_features(SentinelModel(), self.routes, self.tokenizer, self.nodes,
                                          self.config, torch.device("cpu"), torch.float32)
        self.assertEqual(3, len(outputs[0][1]))
        for x, label in zip(*outputs[0]):
            matches = (full_x == x).all(dim=-1)
            self.assertTrue(matches.any())
            self.assertTrue((full_y[matches] == label).all())

    def test_real_model_features_match_the_state_used_after_cached_actions(self):
        model, tok = make_model()
        route = self.routes[0]
        x, y = extract_features(model, [route], tok, self.nodes, self.config,
                                torch.device("cpu"), torch.float32)
        tokens = torch.tensor([tok.encode(route, eos=False)])
        hidden = []
        with torch.no_grad():
            output = model(tokens[:, :2], use_cache=True)
            hidden.append(output.hidden[0, -1])
            for position in range(2, tokens.shape[1]):
                output = model(tokens[:, position:position + 1], past=output.cache, use_cache=True)
                hidden.append(output.hidden[0, 0])
        torch.testing.assert_close(x, torch.stack(hidden), atol=2e-6, rtol=1e-5)
        self.assertEqual(route["nodes"], [self.nodes[index] for index in y.tolist()])

    def test_uniform_probe_has_known_accuracy_nll_brier_and_calibration(self):
        probe = torch.nn.Linear(2, 2)
        with torch.no_grad():
            probe.weight.zero_()
            probe.bias.zero_()
        features, labels = torch.zeros(4, 2), torch.tensor([0, 0, 1, 1])
        cfg = {"batch_size": 3, "calibration_bins": 5, "top_k": 2}
        metrics = evaluate_probe(probe, features, labels, [10, 20], cfg, torch.device("cpu"))
        self.assertEqual(0.5, metrics["accuracy"])
        self.assertEqual(0.5, metrics["balanced_accuracy"])
        self.assertAlmostEqual(torch.log(torch.tensor(2.0)).item(), metrics["nll"], places=6)
        self.assertEqual(0.5, metrics["brier"])
        self.assertEqual(0.0, metrics["ece"])
        self.assertEqual(1.0, metrics["top_k_accuracy"])


if __name__ == "__main__":
    unittest.main()
