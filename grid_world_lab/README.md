# Grid World Lab

A standalone experiment for studying the map implicit in a causal transformer. Training walks create the true map; a frozen-activation location probe determines where the reconstructed trajectory goes after **every** generated direction, including illegal directions.

This project implements the agreed grid experiment independently. It imports no code, datasets, or checkpoints from `world-model-evaluation-main`. It follows the paper's general decoder-training setup, but its grid data, probing protocol, reconstruction procedure, and evaluation are new. The supplied smoke configuration checks the pipeline; it is not evidence that a trained model or probe has learned a reliable map.

## Run it

Use Python 3.10 or newer, PyTorch, and NumPy. Install a CUDA-enabled PyTorch build using the command offered by the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your platform and GPU, then install this project's remaining dependencies. Run commands from this folder:

```powershell
python -m pip install -r requirements.txt
python -m gridworld run --config configs/smoke.json --output runs/smoke
```

Open `runs/smoke/report.html` in a browser. The report is self-contained and works offline. Each run also saves raw JSON and CSV files for independent analysis.

`train.device=auto` chooses CUDA when available and otherwise uses CPU. On CUDA, automatic precision prefers BF16 when supported and otherwise FP16; CPU uses FP32. An installed CPU-only PyTorch wheel cannot use the GPU. Explicitly require CUDA when you want a failed GPU setup to stop instead of falling back:

```powershell
python -m gridworld run --config configs/smoke.json --output runs/gpu-smoke --set train.device=cuda
```

The available configurations have different purposes:

| Configuration | Grid and training walks | Transformer | Purpose |
| --- | --- | --- | --- |
| `configs/tiny.json` | 3 × 3, 30 walks, maximum 6 steps | 1 layer, width 32, 4 heads | Very small functional check |
| `configs/smoke.json` | 4 × 4, 200 walks, maximum 12 steps | 2 layers, width 64, 4 heads | End-to-end smoke experiment |
| `configs/sparse_demo.json` | 10 × 10, 80 walks, maximum 8 steps | 4 layers, width 128, 4 heads | Sparse-map pilot and viewer demonstration |
| `configs/frozen_sparse.json` | 10 × 10, map from 30 walks, LM trained on 20,000 walks | 2 layers, width 64, 4 heads | Recommended sparse-map learning experiment |
| `configs/paper.json` | 10 × 10, 20,000 walks, maximum 32 steps | 12 layers, width 768, 12 heads | Paper-style starting architecture |

`paper.json` does not reproduce the Manhattan dataset or every training hyperparameter from the paper. Its vocabulary and positional embeddings are much smaller, so its parameter count differs from a conventional GPT-2 model. A large training-walk count can make the grid almost complete and consume most reachable origin–destination pairs; use smaller counts and several seeds to study sparse maps.

## Change parameters and run sweeps

Configuration files are partial JSON objects merged with the defaults in `gridworld/config.py`. Unknown keys are rejected. Dimensions, walk counts and lengths, model size, optimization, probe settings, generation, metric thresholds, and seeds are configurable. Repeat `--set` to override individual leaves:

```powershell
python -m gridworld run --config configs/smoke.json --output runs/sparse-10x10 --set data.rows=10 --set data.cols=10 --set data.train_samples=80 --set data.max_length=8 --set model.layers=4 --set model.dim=128 --set model.heads=4 --set train.epochs=20
```

Use JSON values for numbers, booleans, `null`, and arrays; bare strings such as `cuda` are accepted. For example, `--set generation.temperature=0` uses greedy generation, and `--set generation.save_full_probe_probabilities=true` retains all node probabilities rather than just the top candidates. File paths in a run configuration are relative to the directory from which the command is run.

To inspect graph density, available test pairs, and split sizes before training:

```powershell
python -m gridworld prepare --config configs/paper.json --output runs/inspect-map
```

To run the Cartesian product of a sweep specification:

```powershell
python -m gridworld sweep --spec configs/sweep.json --output runs/sweep
```

The example specification varies rows, columns, training-walk count, and maximum length over two seeds. Rows and columns are independent axes, so rectangular grids are included. `base_config` is resolved relative to the sweep specification. Edit the parameter arrays to sweep other supported configuration leaves. Each run gets its own artifacts; `summary.json` and `summary.csv` collect run results.

`aggregate.json` and `aggregate.csv` report the mean and sample standard deviation across seeds for each parameter combination. A single successful seed has no estimated standard deviation. Use `--limit N` for a pilot; `configs/verify_sweep.json` is a two-run functional check. Sweeps run sequentially on one GPU to avoid competing for its memory.

Resume a run or sweep with `--resume`:

```powershell
python -m gridworld run --config configs/smoke.json --output runs/smoke --resume
python -m gridworld sweep --spec configs/sweep.json --output runs/sweep --resume
```

Resume requires the same resolved configuration and reuses completed stages. An interrupted stage is rerun; this is not a mid-optimizer-step restart. Use a new output directory when changing settings.

New runs record hashes of the configuration, dataset, and core implementation. Resume rejects a changed dataset or core implementation when those hashes are present; report regeneration can still be used to update only the viewer. A saved sweep manifest prevents unrelated sweeps from overwriting each other's summaries.

Rebuild a report from an existing run without retraining:

```powershell
python -m gridworld report --run runs/smoke
```

## Exactly what the data means

Nodes are numbered `1..rows*cols` in row-major order. Node 1 is the upper-left cell; increasing rows goes south and increasing columns goes east. For a 10 × 10 grid, the top row contains nodes 1–10 and the next row contains 11–20. `data.num_directions=8` uses `N, NE, E, SE, S, SW, W, NW`; setting it to `4` uses `N, E, S, W`. These are **absolute compass directions**, not turns relative to the previous heading. In eight-direction mode, diagonal edges may cross in the drawing without creating a new intersection node.

For example, run the frozen-map experiment using only cardinal directions with:

```powershell
python -m gridworld run --config configs/frozen_sparse.json --output runs/frozen-sparse-4dir --set data.num_directions=4
```

In the original `data.mode="union"`, for each of exactly `data.train_samples` walks:

1. Choose a start uniformly from the complete bounded grid.
2. Choose a step count uniformly from `data.min_length..data.max_length`, inclusive. The default minimum is 1.
3. At each step, choose uniformly among directions that remain inside the complete grid. There is no wrapping.

Training duplicates are allowed. Every visited vertex, including intermediate vertices, and every traversed undirected edge enters the true map. Traversing A → B establishes B → A as a legal reverse edge even when no training walk traversed it backward. Training frequency records actual occurrences, so it does not invent a reverse traversal count. The union is then frozen: held-out walks and reconstruction never add edges to the true graph.

In `data.mode="frozen_map"`, exactly `data.map_samples` walks are generated by that same bounded-grid procedure and their union defines the true map. These walks are included in LM training. Unseen test endpoint pairs are then reserved, and the remaining walks needed to reach `data.train_samples` are generated using only frozen-map edges while excluding those reserved pairs. Extra training walks therefore improve state and transition coverage without adding nodes or edges. `graph.map_*_counts` records map-construction occurrences; the ordinary node and edge counts cover all LM training walks.

For a first substantive 10 × 10 run:

```powershell
python -m gridworld run --config configs/frozen_sparse.json --output runs/frozen-sparse
```

Length counts **directions**, not tokens. A walk with `n` steps visits `n+1` node positions, possibly revisiting nodes, and is represented by:

```text
ORIGIN DESTINATION DIRECTION_1 ... DIRECTION_n EOS
```

This is `n+3` atomic tokens. Node numbers are lookup-table tokens; the model is not given their coordinates or decimal-digit structure. The token vocabulary contains all grid node IDs, the configured direction tokens, `EOS`, and `PAD`. Right-padding is excluded from the language-model loss.

## Held-out routes and the two experiments

Reference walks are sampled on the frozen true map with uniformly chosen starts, lengths, and available legal directions. `data.heldout_min_length` and `data.heldout_max_length` default to the training length range. Rejection sampling then imposes the split constraints:

| Cohort | Reference route must be new | Ordered origin–destination pair |
| --- | --- | --- |
| `seen` | Yes | Appeared in language-model training |
| `unseen` | Yes | Never appeared in language-model training |

Pairs are ordered: `(A, B)` and `(B, A)` are different prompts despite the graph being undirected. Walks may return to their origin. Multiple test samples can share the same prompt if their reference direction sequences differ; the test is not uniform over unique endpoint pairs.

All held-out complete routes are distinct from every training route and every other held-out route. Test cohorts are reserved before the separate language-model validation, probe-training, and probe-validation routes. “Unseen” refers to language-model training, not to the probe's separate training split. No test route is used for language-model fitting, probe fitting, or checkpoint selection.

The model receives only the two endpoint tokens at generation time. It does **not** receive the held-out reference directions or desired route length. A new reference route therefore does not guarantee a new generated route: the model can reproduce a training route. Reference novelty and generated memorization must be interpreted separately.

Some parameter choices leave no eligible unseen pairs or too few unique routes. The sampler has a finite `data.max_attempts` budget per split and reports requested versus actual counts plus explicit warnings. It does not relax cohort definitions, add new true edges, or duplicate held-out examples to fill the quota. Inspect these warnings before comparing runs. Increasing the attempt budget only helps if valid routes exist.

## Model and probe

The transformer is trained from scratch with an independently implemented GPT-2-style architecture: pre-layer normalization, causal multi-head self-attention, GELU feed-forward blocks, learned absolute positional embeddings, and a tied token-embedding/output matrix. Model width must be divisible by the head count. `model.context_length=null` lets the pipeline choose a context budget for the configured training and generation lengths.

Training uses AdamW, learning-rate warmup and cosine decay, gradient clipping, and optional gradient accumulation. By default, `train.loss_on_prompt=true` includes prediction of the destination token from the origin as well as directions and `EOS`; setting it to false removes that destination-prediction term. The best validation-NLL checkpoint is selected. If a validation split is empty, the recorded selection criterion falls back to training loss. Validation records exact next-token accuracy and the paper-style `legal_action_accuracy`, which accepts any direction legal from the teacher-forced current node and accepts `EOS` only at the destination. Destination-token prediction is excluded from both action metrics.

Efficiency features include PyTorch scaled-dot-product attention, CUDA mixed precision, batched generation with a key/value cache, automatic probe-feature caching when GPU memory permits, and configurable `torch.compile` and gradient checkpointing. Compilation and checkpointing are disabled by default. Reduce batch size or enable checkpointing if the larger architecture exceeds GPU memory.

After the language model is frozen, a linear classifier predicts the current node from its activations. The classifier's classes are the retained true-map nodes. Probe training and validation are separated by **whole route**, so positions from the same route do not leak between those splits. Some rare nodes may have no probe-training examples; `probe.json` records those nodes explicitly.

With token positions numbered from zero:

| Activation position | Consumed tokens | Location label |
| --- | --- | --- |
| 1 | Origin, destination | Origin |
| 2 | Origin, destination, first direction | Node after first direction |
| `j+1` | Prompt and `j` directions | Node after direction `j` |

`probe.layer=-1` uses the final normalized hidden state. Nonnegative values select a zero-based transformer-block output. `EOS` positions are excluded from location-probe training. The best probe checkpoint is selected by probe-validation NLL, with training loss as a fallback only when probe validation is empty. Accuracy, balanced accuracy, per-node accuracy, top-k accuracy, NLL, Brier score, and calibration bins are reported on legal held-out routes.

## Reconstruction follows the probe

The initial reconstruction location is the known origin. The prompt-end probe is diagnostic. For each generated direction:

1. Check whether that direction has a successor in the **frozen true graph**, starting from the current inferred location.
2. Consume the direction token through the transformer.
3. Probe the resulting activation and take the highest-probability node as the next inferred location.
4. Record the transition and update the current inferred location to that node, regardless of the probe confidence.

| True-graph direction check | Probe result | Event kind and drawing |
| --- | --- | --- |
| Legal | Agrees with expected successor | `legal_correct`, solid edge |
| Legal | Disagrees with expected successor | `legal_mismatch`, differently colored dotted edge |
| Illegal | Any predicted node | `illegal`, dotted edge |

If an illegal NW from node A produces probe node B, the next direction is checked from B. The model retains its original token history: the inferred node is not inserted into the prompt and does not repair or reset the transformer. An error does not stop generation. Repeated and conflicting transitions remain in the event log with occurrence counts and confidence statistics. An inferred self-loop is retained as a fake topological edge.

Generation ends at `EOS`, the `generation.max_new_tokens` cap, or an invalid token type. By default, no vocabulary mask is applied; emitting a node or `PAD` after the prompt is recorded as `invalid_token`, with no invented spatial edge. Optional `generation.syntax_only=true` restricts outputs to directions and `EOS`, but still never masks directions according to true-map legality. The token cap includes a generated `EOS` when one occurs.

Each event records the direction probability, inferred source and target, expected true successor when one exists, probe confidence and top candidates, and an independent physical-tracking diagnostic. `generation.save_full_probe_probabilities=true` adds the full probe distribution.

Two location tracks answer different questions:

- **Probe track:** always follows the probe and defines the reconstructed map.
- **Physical track:** follows the generated directions from the original origin using the true graph; after its first invalid direction it becomes undefined and stays undefined.

`physical_valid` means the generated action prefix stayed legal on the physical track. `destination_reached` additionally requires that track to end at the requested destination. `route_success` requires `EOS` and `destination_reached`; merely reaching the goal before running into the token cap does not count. `probe_destination_reached` only asks whether the inferred final node equals the goal. An early `EOS` can be physically valid but unsuccessful.

A probe validated on legal routes can fail after an illegal prefix, and its softmax confidence can remain high. The dotted map therefore records **probe-decoded model behavior**, not established ground truth about the model's internal state. Before interpreting a map scientifically, inspect probe accuracy, rare-node support, calibration, and sensitivity to probe layer and random seed. This implementation logs confidence; it does not automatically calibrate it or reject low-confidence relocations.

## Reading the report

The offline report places the true map beside the reconstruction. Use the cohort selector, cumulative sample slider, or explicit sample selection to inspect a subset. Line-type filters and edge aggregation help with dense overlays. Select individual samples to inspect the step/confidence table; use the CSV files for larger analyses. Configurations, training/probe diagnostics, map statistics, and metric definitions are included with the experiment details.

The true-map edge frequencies and reconstructed frequencies answer different questions. A frequently drawn edge can be rare in training and still be correct. A missing frequent edge affects coverage, not fake-edge precision. Start with the complementary metric families in [METRICS.md](METRICS.md), rather than choosing a single penalty that depends arbitrarily on density.

## Saved artifacts

| File | Contents |
| --- | --- |
| `config.json` | Fully resolved configuration |
| `dataset.json` | Training and held-out routes, frozen graph, counts, split diagnostics |
| `model.pt`, `training.json` | Selected model checkpoint and training/validation results |
| `training_history.json` | Training progress recorded during fitting |
| `probe.pt`, `probe.json` | Selected probe checkpoint, class mapping, probe diagnostics |
| `samples.json` | Generated samples, per-step events, termination and tracking results |
| `metrics.json` | Cohort-level reconstruction, coverage, frequency, and topology metrics |
| `report.json`, `report.html` | Viewer payload and offline interactive report |
| `events.csv` | Per-step records for external analysis |
| `edge_metrics.csv`, `node_metrics.csv` | Training/reference/reconstruction usage and coverage |

The saved configuration and seeds support repeatable comparisons, but bitwise equivalence across different hardware, PyTorch versions, or attention kernels is not guaranteed. Keep the recorded runtime information when comparing results.

## Verify the implementation

```powershell
python -m unittest discover -s tests -v
```

The tests check graph construction, undirected reverse legality, held-out separation, probe-priority transitions, transformer behavior, and metric counterexamples. The tiny and smoke runs exercise the complete pipeline. Larger training and parameter sweeps should be treated as new experiments with their own validation, rather than as conclusions established by the smoke run.

See [VALIDATION.md](VALIDATION.md) for the completed checks and measured limitations of the supplied sparse pilot.
