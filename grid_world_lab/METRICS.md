# Measuring the reconstructed map

Use separate views of **action errors**, **topological accuracy**, **coverage**, and **usage**. A single score hides the distinction between adding fictitious connections and failing to recover real ones.

Let `T` be the set of undirected edges in the frozen true map and `R` the set of distinct inferred source–target edges in the selected generated samples. Every line type contributes to `R`. Edges are canonicalized as `(min(u,v), max(u,v))`, so reverse traversals are the same topology edge. A predicted self-loop is an edge in `R` and is always absent from `T`. Let `S` be the number of samples and `A` the number of generated direction events; `EOS` and malformed non-direction tokens do not contribute to `A`.

## An erroneous action is not necessarily a fake edge

Every event belongs to exactly one kind:

- `legal_correct`: the direction exists at the current inferred node and the probe agrees with its true successor.
- `legal_mismatch`: the direction exists, but the probe chooses a different successor.
- `illegal`: the direction does not exist at the current inferred node; the probe still supplies a successor.

An **error event** is `legal_mismatch` or `illegal`. A **fake topology event** is an event whose inferred undirected edge is absent from `T`. These can differ: an illegal NW action from A might be decoded to B along a real A–B edge. It remains an action error, even though it adds no fake topological edge. Conversely, a legal E action with an incorrect probe relocation can add an absent edge.

Report both families:

| Metric key | Definition |
| --- | --- |
| `mean_error_events_per_sample` | `(illegal events + legal mismatches) / S` |
| `error_event_rate` | `(illegal events + legal mismatches) / A` |
| `illegal_event_rate` | `illegal events / A` |
| `legal_mismatch_event_rate` | `legal mismatches / A` |
| `legal_mismatch_rate_given_legal_direction` | `legal mismatches / (legal correct + legal mismatches)` |
| `mean_fake_topology_events_per_sample` | All traversals of absent edges, including repeats, divided by `S` |
| `fake_topology_event_rate` | All traversals of absent edges divided by `A` |

The per-step rates matter when samples generate different lengths. A model that emits `EOS` immediately can have no direction errors while recovering almost no map and failing the destination task; examine termination, route success, and coverage alongside error rates.

## Three meanings of “fake edges added per sample”

Suppose sample 1 traverses fake edge A–B twice and sample 2 traverses A–B once:

| Metric | Value in this example |
| --- | --- |
| `mean_fake_topology_events_per_sample` | `3 / 2 = 1.5` |
| `mean_sample_unique_fake_topology_edges` | `(1 + 1) / 2 = 1` |
| `mean_new_fake_topology_edges_per_sample` | `1 / 2 = 0.5` |

The first measures repeated erroneous behavior. The second counts an edge once within each route. The third counts an edge only when it first enters the union across the selected samples: `|R \ T| / S`. Per-prefix `new_fake_edges` assigns each first appearance to its sample, so its temporal pattern depends on sample order. Final distinct-edge counts do not depend on order.

## Map precision, recall, and missing edges

| Metric key | Formula or meaning |
| --- | --- |
| `edge_precision` | `|R ∩ T| / |R|` |
| `edge_recall` | `|R ∩ T| / |T|` |
| `edge_f1` | `2|R ∩ T| / (|R| + |T|)` |
| `edge_jaccard` | `|R ∩ T| / |R ∪ T|` |
| `missing_true_edges` | `|T \ R|` |
| `fake_edges` | `|R \ T|` |
| `solid_edge_recall` | Fraction of true edges recovered by at least one `legal_correct` event |
| `node_coverage` | Fraction of retained nodes visited by reconstruction, including supplied origins |
| `predicted_target_node_coverage` | Fraction of retained nodes appearing as post-action probe targets, excluding credit for merely supplying them as origins |

Topology recall deliberately allows a dotted event to recover a real edge; `solid_edge_recall` answers the stricter question. Compare them when a map appears geometrically plausible but its action labels are wrong.

`node_rows` separates `prompt_origin_count` from `predicted_target_count`. A node can appear frequently on the reconstruction because the experiment supplied it as an origin, even if the probe never predicts it. The default probe class set contains only retained true nodes, so its node precision is constrained by construction and is less informative than its accuracy and coverage.

Empty denominators produce JSON `null`, not zero. In particular, an empty reconstruction has undefined precision and F1; recall is zero when the true map is nonempty. Never replace these values with “perfect accuracy.”

## Comparing sparse and dense maps

There are two valid concerns: a sparse map is more easily distorted by one extra connection, while a high-degree node offers more legitimate actions. The metrics expose both without imposing one arbitrary density-dependent penalty.

**Relative topological corruption:** `fake_edges_per_true_edge = |R \ T| / |T|`. One fake edge is a larger fraction of a sparse map. Report it together with ordinary precision and recall, and the actual edge counts. It does not distinguish an innocuous redundant connection from a major shortcut; topology distortion addresses that below.

**Action opportunity baseline:** at a current inferred node of true degree `d`, choosing uniformly among the configured `k` compass actions would be illegal with probability `1 − d/k`. `uniform_direction_illegal_baseline` averages that probability over the actual generated source-node exposures. `illegal_rate_over_uniform_direction_baseline` divides the observed illegal-event rate by this baseline. Values below 1 beat uniform compass choice for those source exposures; values above 1 are worse. It is undefined when the baseline is zero. The legacy `uniform8_*` fields are populated only for eight-direction experiments.

This is a descriptive, exposure-matched baseline, not a separate random-walk policy evaluation: the model and probe determine which source nodes receive exposure. `degree_exposure` reports observed illegality by degree, so a model that mostly visits corners or low-degree nodes can be distinguished from one visiting well-connected nodes.

`uniform_inbounds_illegal_baseline` uses only geometrically in-bounds configured directions in its denominator. It isolates missing true edges from ordinary grid boundaries. Do not divide the observed illegal rate by this alternative baseline and call it a controlled comparison: out-of-bounds actions contribute to the former but cannot occur under the latter.

Two density statistics are saved:

- `graph_density_full_grid`: true edge count divided by the number of potential undirected edges allowed by the configured directions in the entire rectangle.
- `graph_density_visited_nodes`: true edge count divided by potential configured-direction edges whose endpoints are both retained.

For `r` rows and `c` columns, the full-grid edge count is `r(c−1) + (r−1)c + 2(r−1)(c−1)`. The second density distinguishes missing links among retained vertices from simply having unvisited regions.

A useful comparison table across runs therefore includes actual graph density, edge precision/recall, fake edges per true edge, error events per direction, illegal rate by degree, and coverage. Keep generation budget and cohort sampling comparable and summarize multiple seeds before interpreting a density trend.

## Rare edges, forgotten frequent edges, and node usage

Training occurrence counts and reconstructed occurrence counts remain separate. Training counts include repeated visits and repeated traversals; an undirected edge's training count sums its actual traversals in both directions.

| Metric or table field | Interpretation |
| --- | --- |
| `edge_recall` | Every true edge gets equal weight, so recovery of rare edges matters |
| `training_frequency_weighted_edge_recall` | Fraction of all training edge traversals belonging to recovered true edges |
| `training_frequency_weighted_solid_edge_recall` | Same weighting, but recovery must include a `legal_correct` traversal |
| `training_frequency_weighted_node_recall` | Fraction of training node visits belonging to recovered nodes, including reconstruction origins |
| `training_count` and `reconstructed_count` | Which edges/nodes were common or rare, recovered repeatedly, or absent |
| `training_usage_share` and `reconstructed_usage_share` | Counts normalized by their respective total occurrences |
| `usage_share_ratio_to_training` | Reconstructed share divided by training share, without smoothing |
| `reference_count` and `reference_usage_share` | Usage in the held-out legal reference walks for the selected samples |

For example, recovering a once-seen edge while missing a hundred-times-seen edge can improve ordinary recall but leave weighted recall low. Neither creates a fake-edge error. Both differences are visible in the per-edge tables and rarity bins. Node tables additionally report training start and end counts.

The usage-share ratio is a descriptive enrichment measure, **not an error score**. A rare edge used often may be a legitimate route choice. A frequent edge never used may have received little opportunity under the test prompts. Reference-route usage gives an additional exposure baseline, but it is still only one sampled route per prompt, not a proof that another route should use the same edge. Avoid concluding that a model “forgot” an edge solely from a zero reconstruction count.

`metrics.rarity_thresholds` defines nonoverlapping training-count bins. With `[1,5,20]`, the bins are `0`, `1`, `2–5`, `6–20`, and `>=21`. Each bin reports its item count, recovered item count, coverage, training and reconstruction occurrences, reference occurrences, and weighted coverage; edge bins also report solid recovery. Empty bins have undefined coverage. Always inspect bin sizes when comparing runs.

## Does a fake edge substantially distort connectivity?

For graphs no larger than `metrics.topology_max_nodes`, optional shortest-path diagnostics compare unordered pairs of retained true nodes using all inferred line types. Every edge has unit length. Reconstruction paths may use any inferred node, and an absent reconstruction path is treated as disconnected rather than as distance zero.

| Metric key within `topology_distortion` | Meaning |
| --- | --- |
| `lost_reachable_pair_rate` | Fraction of true-connected pairs disconnected in reconstruction |
| `shortcut_pair_rate` | Fraction of true-connected pairs with a shorter reconstructed path |
| `falsely_connected_pair_rate` | Fraction of true-disconnected pairs made connected in reconstruction |
| `mean_distance_change_on_retained_pairs` | Mean reconstructed minus true distance, only for pairs connected in both |
| `mean_absolute_distance_change_on_retained_pairs` | Same pairs, using absolute distance change |

These distinguish a fake bridge between components from an edge that only adds a small local shortcut. Missing true edges can lengthen paths or disconnect pairs. The signed mean can cancel shortcuts against lengthened paths, so read it with the absolute mean and pair counts. When the size cap is exceeded or the feature is disabled, the report records a skip reason rather than claiming zero distortion.

## Route validity and stopping behavior

The probe trajectory and the physically valid trajectory from the original origin are different diagnostics:

- `physical_valid_rate`: fraction of generated direction prefixes that remain legal from the original origin, independently of all probe relocations.
- `destination_reached_rate`: fraction physically valid and ending at the requested destination.
- `route_success_rate`: fraction ending with `EOS` at the physically correct destination.
- `probe_destination_reached_rate`: fraction whose inferred final node equals the destination, even if physical execution failed.
- `eos_rate`, `max_new_tokens_rate`, and `invalid_token_rate`: why generation stopped.

An empty direction sequence can be physically valid; that does not mean it solved the requested route. A probe may also relocate to the goal after an illegal action. These cases are why the independent physical track and explicit stopping metrics are retained.

## Confidence is diagnostic, not a guarantee

`probe.json` reports legal-route held-out accuracy, class-balanced accuracy, NLL, multiclass Brier score, top-k accuracy, and expected calibration error (ECE). Brier score is the sum of squared probability errors over classes, averaged over positions. ECE averages the absolute gap between bin accuracy and bin mean confidence, weighted by the number of positions in each confidence bin. Per-node supports expose classes whose accuracy estimate rests on very few observations.

These calibration statistics do not recalibrate the probe. More importantly, they are measured on legal reference histories. After an illegal token history, there is no true current node under the original map's transition rules, and a confidently decoded node need not be a correct interpretation of the model's state. The experiment deliberately follows that prediction to expose its consequences; it does not turn the prediction into a validated semantic fact.

Use the per-step confidence table, compare legal-mismatch and illegal-edge confidence distributions, and repeat analyses across probe layers and seeds. If probe performance on untouched legal references is poor, the reconstruction mixes model behavior with probe error and should be interpreted accordingly.

## Suggested first comparison

For each cohort and run, inspect:

1. Actual training-map node/edge counts and density; actual held-out route and ordered-pair counts.
2. Probe validation/test accuracy, balanced accuracy, calibration, and per-node support.
3. Physical route success and stopping behavior.
4. Error events and fake topology events per sample and per direction.
5. Unique edge precision/recall, solid recall, target-node coverage, and their evolution with sample count.
6. Rarity-bin coverage, training-weighted recall, usage tables, and reference exposure.
7. Shortcut and connectivity distortion when computed.

No single item substitutes for the others. Report the same selection rules and generation budgets across comparable runs, and retain counts alongside rates so sparse cohorts and small rarity bins remain visible.
