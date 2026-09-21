import copy
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from gridworld.extract import collect_contexts, infer_geometry, summarize_transitions
from gridworld.extract_report import render_extraction


def fixture():
    graph = {'rows': 2, 'cols': 2, 'nodes': [1, 2, 3, 4],
             'directions': ['N', 'E', 'S', 'W'], 'edges': [[1, 2], [1, 3], [2, 4], [3, 4]]}
    routes = [
        {'id': 'seen-1', 'origin': 1, 'destination': 4, 'directions': ['E', 'S'], 'nodes': [1, 2, 4]},
        {'id': 'seen-2', 'origin': 1, 'destination': 4, 'directions': ['S', 'E'], 'nodes': [1, 3, 4]},
        {'id': 'seen-3', 'origin': 4, 'destination': 1, 'directions': ['N', 'W'], 'nodes': [4, 2, 1]}]
    return {'graph': graph, 'splits': {'seen': routes, 'unseen': []}}


class ExtractionTests(unittest.TestCase):
    def test_sampling_uses_known_nodes_deduplicates_prefixes_and_respects_context(self):
        dataset = fixture()
        original = copy.deepcopy(dataset)
        contexts, coverage = collect_contexts(dataset, [], 20, 42, 8, ['seen'])
        self.assertEqual(len(contexts), 5)  # Shared [1, 4] prompt counted only once.
        self.assertEqual({c['source'] for c in contexts}, {1, 2, 3, 4})
        self.assertEqual(original, dataset)
        self.assertEqual((contexts, coverage), collect_contexts(dataset, [], 20, 42, 8, ['seen']))
        short, diagnostics = collect_contexts(dataset, [], 20, 42, 4, ['seen'])
        self.assertTrue(all(len(c['prefix']) == 0 for c in short))
        self.assertGreater(diagnostics['skipped_context_limit'], 0)
        limited, _ = collect_contexts(dataset, [], 1, 42, 8, ['seen'])
        self.assertEqual(len(limited), 4)

    def test_generated_validity_tracks_physics_without_probe_relocations(self):
        samples = [{'id': 'seen-1', 'cohort': 'seen', 'origin': 1, 'destination': 4,
                    'events': [{'source': 1, 'direction': 'E'},
                               {'source': 3, 'direction': 'E'},
                               {'source': 4, 'direction': 'N'}]}]
        contexts, _ = collect_contexts(fixture(), samples, 20, 0, 10, [])
        contexts.sort(key=lambda c: c['step'])
        self.assertEqual([c['kind'] for c in contexts], ['generated_valid', 'generated_valid', 'generated_invalid'])
        self.assertEqual([c['known_source'] for c in contexts], [1, 2, None])
        self.assertEqual(contexts[1]['source'], 2)  # Ignore incorrect decoded source 3.
        self.assertEqual(contexts[2]['source'], 4)  # Only an inferred label now.

    def test_cycle_return_does_not_imply_independent_reciprocity(self):
        contexts = [{'id': 0, 'kind': 'reference', 'source': 1, 'known_source': 1, 'pre_node': 1, 'destination': 3},
                    {'id': 1, 'kind': 'reference', 'source': 2, 'known_source': 2, 'pre_node': 2, 'destination': 1},
                    {'id': 2, 'kind': 'reference', 'source': 1, 'known_source': 1, 'pre_node': 3, 'destination': 2}]
        def event(ident, direction, target, returned):
            return {'context_id': ident, 'direction': direction, 'target': target,
                    'return_node': returned, 'true_neighbor': None, 'confidence': .9,
                    'direction_probability': .1, 'probabilities': [.05, .9, .05]}
        events = [event(0, 'E', 2, 1), event(1, 'W', 3, 2), event(2, 'E', 3, 1)]
        rows = summarize_transitions(contexts, events, [1, 2, 3])
        forward = next(r for r in rows if r['source'] == 1)
        self.assertEqual(forward['agreement'], .5)
        self.assertEqual(forward['cycle_return_to_source_rate'], 1)
        self.assertFalse(forward['independent_reciprocal'])
        self.assertIsNone(forward['correct_target_rate'])  # No truth label for illegal moves.
        matched = summarize_transitions(contexts, events, [1, 2, 3], True)
        self.assertEqual(next(r for r in matched if r['source'] == 1)['contexts'], 1)

    def test_geometry_uses_direction_constraints_instead_of_node_numbering(self):
        def row(source, direction, target, reciprocal=True):
            return {'kind': 'reference', 'source': source, 'direction': direction,
                    'target': target, 'contexts': 10, 'agreement': .9,
                    'mean_probe_confidence': .95, 'independent_reciprocal': reciprocal,
                    'true_neighbor': None}
        geometry = infer_geometry([
            row(2, 'S', 13), row(13, 'N', 2),
            row(13, 'E', 14), row(14, 'W', 13),
        ], [2, 13, 14, 99])
        raw = {p['node']: (p['raw_x'], p['raw_y']) for p in geometry['positions']}
        self.assertAlmostEqual(raw[13][0], raw[2][0], places=6)
        self.assertAlmostEqual(raw[13][1] - raw[2][1], 1, places=6)
        self.assertAlmostEqual(raw[14][0] - raw[13][0], 1, places=6)
        self.assertAlmostEqual(geometry['weighted_rmse'], 0, places=6)
        self.assertEqual(geometry['component_count'], 2)  # Unconstrained 99 is still displayed.

    def test_geometry_exposes_inconsistent_direction_constraints(self):
        rows = [
            {'kind': 'reference', 'source': 2, 'direction': direction, 'target': 13,
             'contexts': 10, 'agreement': .9, 'mean_probe_confidence': .9,
             'independent_reciprocal': False, 'true_neighbor': None}
            for direction in ('S', 'E')]
        geometry = infer_geometry(rows, [2, 13])
        self.assertGreater(geometry['weighted_rmse'], .5)
        self.assertTrue(all(edge['residual'] > .5 for edge in geometry['constraints']))

    def test_viewer_data_cannot_escape_script(self):
        payload = {'note': '</script><script>bad()</script>'}
        with patch.object(Path, 'write_text') as write:
            render_extraction(payload, Path('unused.html'))
        html = write.call_args.args[0]
        embedded = re.search(r'type="application/json">(.*?)</script>', html, re.S).group(1)
        self.assertNotIn('<script>', embedded)
        self.assertEqual(json.loads(embedded), payload)
        self.assertIn('id="directionMarks"', html)
        self.assertIn('id="conflicts"', html)
        self.assertIn('Reciprocal pairs are collapsed into one line', html)


if __name__ == '__main__':
    unittest.main()
