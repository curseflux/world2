# Implementation verification

Verified locally on 2026-09-19 with Python 3.12, PyTorch 2.11.0+cu128,
NumPy 2.5.3, and an NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB).

- **43 unit tests pass:** frozen graph construction, split separation,
  causal attention, cache equivalence, probe timing, padding, exact token
  weighting with gradient accumulation, continuation after illegal moves,
  probe priority after legal disagreements, edge metrics, resume guards,
  sweep expansion, and safe HTML embedding.
- **GPU integration runs:** FP32, BF16, and FP16. The 12-layer, 768-width,
  12-head architecture completed a one-update check; this does not establish
  convergence or performance for the paper-style configuration.
- **CPU integration:** complete tiny run with gradient accumulation and
  activation checkpointing enabled.
- **Sweeps:** two seeds completed; resume reused their completed stages.
- **Browser checks:** local Chromium loaded the offline HTML with no script
  errors. Zero/all/subset selections, numeric ranges and IDs, cohort switching,
  display filters, frequency tables, CSV downloads, and mobile width passed.

The current sparse pilot is in `runs/sparse-demo-verified/report.html`:
10 by 10 grid, 80 training walks, lengths 1 through 8, 96 retained nodes,
199 undirected edges, and 50 generated samples in each test cohort.
It uses a 4-layer, width-128 model with 810,752 parameters. Its best probe has
16.55% validation location accuracy, 24.55% seen-pair test accuracy, and 15.11%
unseen-pair test accuracy on legal reference walks. These low scores make it a
pipeline demonstration rather than evidence of a reliable reconstructed world
model. Dotted edges must be interpreted jointly with probe quality.

To verify again, run from this folder:

```powershell
& '../.venv/Scripts/python.exe' -m unittest discover -s tests -v
& '../.venv/Scripts/python.exe' -m gridworld run --config configs/sparse_demo.json --output runs/new-pilot
& '../.venv/Scripts/python.exe' -m gridworld sweep --spec configs/verify_sweep.json --output runs/new-sweep
```

The optional browser integration test is `tests/browser_check.cjs`. It requires
Node.js and Playwright with Chromium; the experiment and report themselves do
not require either. Core implementation hashes are recorded in new runs so
later implementation changes cannot silently reuse their trained artifacts.
