"""Small counterexamples protect metric meanings, not just arithmetic paths."""

import copy
import unittest

from gridworld.metrics import summarize


def graph():
    return {
        "rows": 2, "cols": 2, "nodes": [1, 2, 3, 4],
        "edges": [[1, 2], [2, 4], [3, 4]],
        "edge_counts": {"1-2": 1, "2-4": 100, "3-4": 9},
        "node_counts": {"1": 1, "2": 100, "3": 10, "4": 99},
    }


def event(source, target, kind="legal_correct", direction="E", confidence=0.75):
    return {"step": 1, "source": source, "target": target, "kind": kind,
            "direction": direction, "probe_confidence": confidence,
            "direction_probability": 0.2}


def sample(ident, events, origin=1, reference_nodes=None, **kwargs):
    return {"id": ident, "origin": origin, "destination": 4,
            "events": events, "reference_nodes": reference_nodes or [],
            "termination": "max_new_tokens", **kwargs}


class MetricsTests(unittest.TestCase):
    def test_action_error_does_not_imply_a_fake_topological_edge(self):
        # NW is invalid at node 1 but the probe jumps along the real 1--2 edge.
        result = summarize(graph(), [sample(0, [event(1, 2, "illegal", "NW")])])
        stats = result["summary"]
        self.assertEqual(stats["illegal_events"], 1)
        self.assertEqual(stats["error_event_rate"], 1)
        self.assertEqual(stats["fake_edges"], 0)
        self.assertEqual(stats["fake_topology_events"], 0)
        self.assertEqual(stats["edge_precision"], 1)
        self.assertAlmostEqual(stats["edge_recall"], 1 / 3)
        self.assertEqual(stats["solid_edge_recall"], 0)

    def test_repeated_fake_edges_distinguish_occurrences_and_additions(self):
        samples = [
            sample(10, [event(1, 4, "illegal"), event(4, 1, "illegal")]),
            sample(11, [event(1, 4, "legal_mismatch")]),
        ]
        result = summarize(graph(), samples)
        stats = result["summary"]
        self.assertEqual(stats["fake_edges"], 1)
        self.assertEqual(stats["fake_topology_events"], 3)
        self.assertEqual(stats["mean_fake_topology_events_per_sample"], 1.5)
        self.assertEqual(stats["mean_sample_unique_fake_topology_edges"], 1)
        self.assertEqual(stats["mean_new_fake_topology_edges_per_sample"], 0.5)
        self.assertEqual([row["new_fake_edges"] for row in result["prefixes"]], [1, 0])
        fake = next(row for row in result["edge_rows"] if row["edge"] == "1-4")
        self.assertEqual(fake["illegal_count"], 2)
        self.assertEqual(fake["legal_mismatch_count"], 1)
        self.assertEqual(fake["sample_count"], 2)

    def test_rare_edge_recovery_and_frequent_omission_are_both_visible(self):
        result = summarize(graph(), [sample(0, [event(1, 2)], reference_nodes=[2, 4, 3])],
                           {"rarity_thresholds": [1, 10]})
        stats = result["summary"]
        self.assertAlmostEqual(stats["edge_recall"], 1 / 3)
        self.assertAlmostEqual(stats["training_frequency_weighted_edge_recall"], 1 / 110)
        rare = next(row for row in result["edge_rows"] if row["edge"] == "1-2")
        frequent = next(row for row in result["edge_rows"] if row["edge"] == "2-4")
        self.assertEqual(rare["training_count"], 1)
        self.assertEqual(rare["reconstructed_count"], 1)
        self.assertEqual(frequent["training_count"], 100)
        self.assertEqual(frequent["reconstructed_count"], 0)
        self.assertEqual(frequent["reference_count"], 1)
        buckets = {row["label"]: row for row in result["rarity"]["edges"]}
        self.assertEqual(buckets["1"]["coverage"], 1)
        self.assertEqual(buckets[">=11"]["coverage"], 0)

    def test_density_and_baseline_have_distinct_denominators(self):
        result = summarize(graph(), [sample(0, [event(1, 2), event(2, 4, direction="S")])])
        stats = result["summary"]
        self.assertEqual(stats["grid_possible_edge_count"], 6)
        self.assertEqual(stats["graph_density_full_grid"], 0.5)
        self.assertEqual(stats["graph_density_visited_nodes"], 0.5)
        # The two actual sources have degrees 1 and 2, out of 8 compass actions.
        self.assertAlmostEqual(stats["uniform8_illegal_baseline"], (7 / 8 + 6 / 8) / 2)
        # Each 2x2 corner has three geometrically in-bounds directions.
        self.assertAlmostEqual(stats["uniform_inbounds_illegal_baseline"], 0.5)

    def test_four_direction_density_and_baseline(self):
        cardinal = graph()
        cardinal["directions"] = ["N", "E", "S", "W"]
        result = summarize(cardinal, [sample(0, [event(1, 2), event(2, 4, direction="S")])])
        stats = result["summary"]
        self.assertEqual(stats["direction_count"], 4)
        self.assertEqual(stats["grid_possible_edge_count"], 4)
        self.assertEqual(stats["graph_density_full_grid"], 0.75)
        self.assertAlmostEqual(stats["uniform_direction_illegal_baseline"], (3 / 4 + 2 / 4) / 2)
        self.assertIsNone(stats["uniform8_illegal_baseline"])

    def test_empty_selection_has_undefined_precision_and_zero_coverage(self):
        result = summarize(graph(), [])
        stats = result["summary"]
        self.assertIsNone(stats["edge_precision"])
        self.assertIsNone(stats["edge_f1"])
        self.assertEqual(stats["edge_recall"], 0)
        self.assertEqual(stats["node_coverage"], 0)
        self.assertIsNone(stats["mean_error_events_per_sample"])
        self.assertIsNone(result["edge_rows"][0]["usage_share_ratio_to_training"])

    def test_self_loop_is_fake_and_confidences_retained(self):
        result = summarize(graph(), [sample(0, [event(1, 1, "legal_mismatch", confidence=0.99)])])
        stats = result["summary"]
        self.assertEqual(stats["fake_edges"], 1)
        self.assertEqual(stats["legal_mismatch_events"], 1)
        self.assertEqual(stats["mean_probe_confidence_legal_mismatch"], 0.99)

    def test_node_coverage_separates_prompted_origins_from_model_targets(self):
        result = summarize(graph(), [sample(0, [event(1, 2)]), sample(1, [], origin=3)])
        self.assertEqual(result["summary"]["node_coverage"], 0.75)
        self.assertEqual(result["summary"]["predicted_target_node_coverage"], 0.25)
        rows = {row["node"]: row for row in result["node_rows"]}
        self.assertEqual(rows[3]["prompt_origin_count"], 1)
        self.assertEqual(rows[3]["predicted_target_count"], 0)

    def test_shortcut_and_cross_component_bridge_are_distinguished(self):
        # True path 1--2--4--3; inferred extra 3--1 makes shorter true paths.
        samples = [sample(0, [event(1, 2), event(2, 4), event(4, 3),
                              event(3, 1, "illegal")])]
        result = summarize(graph(), samples)
        self.assertGreater(result["topology_distortion"]["shortcut_pairs"], 0)
        disconnected = graph()
        disconnected["edges"] = [[1, 2], [3, 4]]
        disconnected["edge_counts"] = {"1-2": 1, "3-4": 1}
        samples[0]["events"][1]["kind"] = "illegal"
        bridged = summarize(disconnected, samples)["topology_distortion"]
        self.assertEqual(bridged["falsely_connected_pairs"], 4)

    def test_validation_and_optional_topology_limit(self):
        with self.assertRaises(ValueError):
            summarize(graph(), [], {"rarity_thresholds": [5, 1]})
        with self.assertRaises(ValueError):
            summarize(graph(), [sample(0, [event(1, 4)])])
        with self.assertRaises(ValueError):
            summarize(graph(), [sample(0, []), sample(0, [])])
        result = summarize(graph(), [], {"topology_max_nodes": 2})
        self.assertEqual(result["topology_distortion"]["reason"], "node_limit")

    def test_independent_route_validity_and_inputs_are_preserved(self):
        true_graph = graph()
        samples = [sample(0, [event(1, 4, "illegal")], physical_valid=False,
                          destination_reached=False, probe_destination_reached=True,
                          route_success=False)]
        original_graph, original_samples = copy.deepcopy(true_graph), copy.deepcopy(samples)
        stats = summarize(true_graph, samples)["summary"]
        self.assertEqual(stats["physical_valid_rate"], 0)
        self.assertEqual(stats["probe_destination_reached_rate"], 1)
        self.assertEqual(stats["route_success_rate"], 0)
        self.assertEqual(stats["max_new_tokens_rate"], 1)
        self.assertEqual(true_graph, original_graph)
        self.assertEqual(samples, original_samples)


if __name__ == "__main__":
    unittest.main()
