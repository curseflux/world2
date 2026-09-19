import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from gridworld.config import load_config, set_override, validate_config
from gridworld.data import DIRECTIONS, bounded_neighbor, build_dataset, build_neighbor_table, edge_key, graph_neighbor, route_key


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            "rows": 4, "cols": 4, "train_samples": 30, "min_length": 1,
            "max_length": 7, "validation_samples": 10, "probe_train_samples": 12,
            "probe_validation_samples": 10, "test_samples_per_cohort": 12,
            "max_attempts": 20000, "seed": 19,
        }
        cls.dataset = build_dataset(cls.config)

    def test_union_retains_intermediate_nodes_and_counts(self):
        graph = self.dataset["graph"]
        train = self.dataset["splits"]["train"]
        nodes = {node for route in train for node in route["nodes"]}
        edges = {tuple(sorted((u, v))) for route in train for u, v in zip(route["nodes"], route["nodes"][1:])}
        self.assertEqual(nodes, set(graph["nodes"]))
        self.assertEqual(edges, {tuple(edge) for edge in graph["edges"]})
        self.assertEqual(sum(len(route["directions"]) for route in train), sum(graph["edge_counts"].values()))
        self.assertEqual(sum(len(route["nodes"]) for route in train), sum(graph["node_counts"].values()))
        # A single long walk makes internal-only nodes meaningful even when not endpoints.
        single = build_dataset({**self.config, "train_samples": 1, "min_length": 7,
                                "test_samples_per_cohort": 0, "validation_samples": 0,
                                "probe_train_samples": 0, "probe_validation_samples": 0})
        route = single["splits"]["train"][0]
        self.assertEqual(set(route["nodes"]), set(single["graph"]["nodes"]))

    def test_all_heldout_routes_are_disjoint_and_legal(self):
        graph, splits = self.dataset["graph"], self.dataset["splits"]
        used = {route_key(route) for route in splits["train"]}
        seen_pairs = {(r["origin"], r["destination"]) for r in splits["train"]}
        for split in ("seen", "unseen", "validation", "probe_train", "probe_validation"):
            self.assertEqual(self.dataset["split_stats"][split]["requested"], len(splits[split]))
            for route in splits[split]:
                key = route_key(route)
                self.assertNotIn(key, used)
                used.add(key)
                pair = route["origin"], route["destination"]
                if split == "seen":
                    self.assertIn(pair, seen_pairs)
                if split == "unseen":
                    self.assertNotIn(pair, seen_pairs)
                self.assertEqual(route["origin"], route["nodes"][0])
                self.assertEqual(route["destination"], route["nodes"][-1])
                for node, direction, next_node in zip(route["nodes"], route["directions"], route["nodes"][1:]):
                    self.assertEqual(next_node, graph_neighbor(graph, node, direction))

    def test_edges_are_undirected_and_grid_does_not_wrap(self):
        graph = {"rows": 2, "cols": 3, "nodes": [1, 5], "edges": [[1, 5]]}
        table = build_neighbor_table(graph)
        self.assertEqual(table, {1: {"SE": 5}, 5: {"NW": 1}})
        self.assertIsNone(graph_neighbor(graph, 1, "E"))
        self.assertIsNone(bounded_neighbor(3, "E", 2, 3))
        self.assertIsNone(bounded_neighbor(1, "NW", 2, 3))
        self.assertEqual(4, bounded_neighbor(1, "S", 2, 3))
        self.assertEqual("1-5", edge_key(5, 1))
        self.assertEqual(8, len(DIRECTIONS))

    def test_named_random_streams_are_repeatable(self):
        self.assertEqual(self.dataset, build_dataset(copy.deepcopy(self.config)))
        changed = build_dataset({**self.config, "test_samples_per_cohort": 1})
        self.assertEqual(self.dataset["graph"], changed["graph"])
        self.assertEqual(self.dataset["splits"]["train"], changed["splits"]["train"])

    def test_exhausted_splits_warn_without_relaxing_definitions(self):
        # Only two possible directed length-one routes exist; training observes both.
        dataset = build_dataset({"rows": 1, "cols": 2, "train_samples": 100,
                                 "max_length": 1, "max_attempts": 20,
                                 "validation_samples": 2, "probe_train_samples": 2,
                                 "probe_validation_samples": 2, "test_samples_per_cohort": 2})
        self.assertEqual(100, len(dataset["splits"]["train"]))
        self.assertEqual([], dataset["splits"]["unseen"])
        self.assertEqual(0, dataset["split_stats"]["unseen"]["attempts"])
        self.assertEqual([], dataset["splits"]["seen"])
        self.assertEqual(20, dataset["split_stats"]["seen"]["attempts"])
        self.assertTrue(dataset["split_warnings"])

    def test_frozen_union_does_not_expand_for_evaluation(self):
        train_only = build_dataset({**self.config, "test_samples_per_cohort": 0,
                                   "validation_samples": 0, "probe_train_samples": 0,
                                   "probe_validation_samples": 0})
        self.assertEqual(train_only["graph"], self.dataset["graph"])

    def test_input_validation(self):
        for patch in ({"rows": 1, "cols": 1}, {"train_samples": 0},
                      {"min_length": 8, "max_length": 7}, {"max_attempts": 0}):
            with self.assertRaises(ValueError):
                build_dataset({**self.config, **patch})


class ConfigTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        config = load_config(overrides=["model.dim=64", "model.heads=4", "train.device=cpu",
                                        "generation.syntax_only=true", "train.max_steps=null"])
        self.assertEqual(12, config["model"]["layers"])
        self.assertEqual(64, config["model"]["dim"])
        self.assertEqual("cpu", config["train"]["device"])
        self.assertTrue(config["generation"]["syntax_only"])
        self.assertIsNone(config["train"]["max_steps"])

    def test_partial_file_and_typo_rejection(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"
        self.assertEqual(4, load_config(path)["data"]["rows"])
        with patch("gridworld.config.Path.read_text", return_value=json.dumps({"data": {"rowz": 6}})):
            with self.assertRaisesRegex(ValueError, "rowz"):
                load_config(path)
        with self.assertRaises(ValueError):
            load_config(overrides=["model.dim=65", "model.heads=4"])
        with self.assertRaises(ValueError):
            load_config(overrides=["data.train_samples=0"])

    def test_sweep_override_uses_existing_leaf(self):
        config = load_config()
        set_override(config, "data.rows", 6)
        validate_config(config)
        with self.assertRaises(ValueError):
            set_override(config, "data.unknown", 6)

    def test_runtime_choices_and_nonfinite_numbers_are_rejected(self):
        for override in ("train.precision=typo", "train.device=cuda:abc", "probe.cache_device=gpu",
                         "train.learning_rate=NaN", "generation.temperature=Infinity",
                         "probe.cache_gpu_fraction=1.1"):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    load_config(overrides=[override])
        config = load_config(overrides=["train.device=cuda:1", "probe.cache_gpu_fraction=0"])
        self.assertEqual("cuda:1", config["train"]["device"])
        self.assertEqual(0, config["probe"]["cache_gpu_fraction"])


if __name__ == "__main__":
    unittest.main()
