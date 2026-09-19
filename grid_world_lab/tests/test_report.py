import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from gridworld.report import render_report


class ReportTests(unittest.TestCase):
    def test_report_is_standalone_and_embedded_data_cannot_end_script(self):
        payload = {
            "config": {"note": "</script><script>alert('no')</script> & \u2028"},
            "dataset": {"graph": {"rows": 2, "cols": 2, "nodes": [1, 2], "edges": [[1, 2]]}},
            "samples": [],
            "metrics": {},
            "probe": {"undefined_metric": float("nan")},
        }
        with patch.object(Path, 'mkdir'), patch.object(Path, 'write_text') as write:
            render_report(payload, Path('nested/report.html'))
            html = write.call_args.args[0]
        match = re.search(r'<script id="experiment-data" type="application/json">(.*?)</script>', html, re.S)
        self.assertIsNotNone(match)
        loaded = json.loads(match.group(1))
        self.assertEqual(loaded["config"], payload["config"])
        self.assertIsNone(loaded["probe"]["undefined_metric"])
        self.assertNotIn("<script>", match.group(1))
        self.assertNotRegex(html, r'<(?:script|link)[^>]+(?:src|href)="https?://')
        self.assertIn('id="sample-count"', html)
        self.assertIn('id="sample-subset"', html)


if __name__ == "__main__":
    unittest.main()
