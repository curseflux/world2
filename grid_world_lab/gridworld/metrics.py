"""Reconstruction metrics with separate action errors and topological errors.

An illegal compass move can be decoded to a *real* undirected edge. Conversely,
a legal compass move with a disagreeing probe can create an absent edge. These
are different questions and must not be counted interchangeably.
"""

from collections import Counter, defaultdict, deque
import math

from .data import DELTAS, DIRECTIONS


KINDS = ("legal_correct", "legal_mismatch", "illegal")


def edge_key(u, v):
    """Canonical undirected edge, including predicted self loops."""
    return (min(int(u), int(v)), max(int(u), int(v)))


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def _mean(values):
    return sum(values) / len(values) if values else None


def _grid_neighbors(node, rows, cols, directions=DIRECTIONS):
    if not 1 <= node <= rows * cols:
        return []
    row, col = divmod(node - 1, cols)
    result = []
    for direction in directions:
        dr, dc = DELTAS[direction]
        rr, cc = row + dr, col + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            result.append(rr * cols + cc + 1)
    return result


def _count(mapping, key):
    value = mapping.get(key, mapping.get(str(key), 0))
    if (not isinstance(value, (int, float)) or not math.isfinite(value) or
            value < 0 or int(value) != value):
        raise ValueError(f"Invalid training count for {key!r}: {value!r}")
    return value


def _edge_count(mapping, edge):
    u, v = edge
    return _count(mapping, f"{u}-{v}")


def _rarity(rows, thresholds, is_edge):
    bins = [(0, 0)]
    lower = 1
    for upper in thresholds:
        bins.append((lower, upper))
        lower = upper + 1
    bins.append((lower, None))
    result = []
    for lower, upper in bins:
        members = [row for row in rows if row["is_true"] and
                   row["training_count"] >= lower and
                   (upper is None or row["training_count"] <= upper)]
        recovered = [row for row in members if row["reconstructed_count"] > 0]
        label = str(lower) if upper == lower else (
            f">={lower}" if upper is None else f"{lower}-{upper}")
        record = {
            "label": label, "min_count": lower, "max_count": upper,
            "items": len(members), "recovered_items": len(recovered),
            "coverage": _ratio(len(recovered), len(members)),
            "training_occurrences": sum(row["training_count"] for row in members),
            "recovered_training_occurrences": sum(row["training_count"] for row in recovered),
            "reconstructed_occurrences": sum(row["reconstructed_count"] for row in members),
            "reference_occurrences": sum(row["reference_count"] for row in members),
        }
        record["training_frequency_weighted_coverage"] = _ratio(
            record["recovered_training_occurrences"], record["training_occurrences"])
        if is_edge:
            record["solid_recovered_items"] = sum(row["legal_correct_count"] > 0 for row in members)
            record["solid_coverage"] = _ratio(record["solid_recovered_items"], len(members))
        result.append(record)
    return result


def _distances(adjacency, source):
    distances = {source: 0}
    pending = deque([source])
    while pending:
        node = pending.popleft()
        for target in adjacency[node]:
            if target not in distances:
                distances[target] = distances[node] + 1
                pending.append(target)
    return distances


def _topology_distortion(true_nodes, true_edges, reconstructed_edges, max_nodes):
    all_nodes = true_nodes | {node for edge in reconstructed_edges for node in edge}
    if len(all_nodes) > max_nodes:
        return {"computed": False, "reason": "node_limit", "node_count": len(all_nodes),
                "max_nodes": max_nodes}
    true_adj, inferred_adj = defaultdict(set), defaultdict(set)
    for u, v in true_edges:
        true_adj[u].add(v)
        true_adj[v].add(u)
    for u, v in reconstructed_edges:
        inferred_adj[u].add(v)
        inferred_adj[v].add(u)
    true_reachable = true_disconnected = reconstructed_reachable = 0
    lost_pairs = shortcuts = longer = unchanged = merged_pairs = 0
    total_delta = total_abs_delta = 0
    nodes = sorted(true_nodes)
    for index, source in enumerate(nodes):
        true_dist = _distances(true_adj, source)
        inferred_dist = _distances(inferred_adj, source)
        for target in nodes[index + 1:]:
            original = true_dist.get(target)
            predicted = inferred_dist.get(target)
            if original is None:
                true_disconnected += 1
                merged_pairs += predicted is not None
                continue
            true_reachable += 1
            if predicted is None:
                lost_pairs += 1
                continue
            reconstructed_reachable += 1
            delta = predicted - original
            shortcuts += delta < 0
            longer += delta > 0
            unchanged += delta == 0
            total_delta += delta
            total_abs_delta += abs(delta)
    return {
        "computed": True,
        "true_reachable_pairs": true_reachable,
        "true_disconnected_pairs": true_disconnected,
        "lost_reachable_pairs": lost_pairs,
        "lost_reachable_pair_rate": _ratio(lost_pairs, true_reachable),
        "shortcut_pairs": shortcuts,
        "shortcut_pair_rate": _ratio(shortcuts, true_reachable),
        "longer_path_pairs": longer,
        "unchanged_distance_pairs": unchanged,
        "falsely_connected_pairs": merged_pairs,
        "falsely_connected_pair_rate": _ratio(merged_pairs, true_disconnected),
        "mean_distance_change_on_retained_pairs": _ratio(total_delta, reconstructed_reachable),
        "mean_absolute_distance_change_on_retained_pairs": _ratio(total_abs_delta, reconstructed_reachable),
    }


def summarize(graph, samples, config=None):
    """Summarize a chosen sample subset; never mutate the graph or samples.

    ``samples`` order defines cumulative prefixes. All direction events require
    source, target, kind, direction, and probe_confidence. Nodes use 1-based row
    major integer IDs. True and inferred topologies are undirected, while kinds
    retain the action-conditioned correctness classification from generation.

    Config accepts rarity_thresholds (positive increasing integers),
    compute_topology_distortion (bool), and topology_max_nodes (nonnegative int).
    Undefined rates use None, which serializes as JSON null, rather than a
    misleading zero. Empty-reconstruction precision and F1 are undefined.
    """
    config = {} if config is None else config
    thresholds = config.get("rarity_thresholds", [1, 2, 5, 10, 25, 100])
    if (not isinstance(thresholds, (list, tuple)) or
            any(type(value) is not int or value <= 0 for value in thresholds) or
            list(thresholds) != sorted(set(thresholds))):
        raise ValueError("rarity_thresholds must be strictly increasing positive integers")
    topology_limit = config.get("topology_max_nodes", 200)
    if type(topology_limit) is not int or topology_limit < 0:
        raise ValueError("topology_max_nodes must be a nonnegative integer")
    compute_topology = config.get("compute_topology_distortion", True)
    if type(compute_topology) is not bool:
        raise ValueError("compute_topology_distortion must be boolean")

    rows = int(graph.get("rows", graph.get("height", 0)))
    cols = int(graph.get("cols", graph.get("width", 0)))
    if rows <= 0 or cols <= 0:
        raise ValueError("graph rows and cols must be positive")
    directions = tuple(graph.get("directions", DIRECTIONS))
    if not directions or len(set(directions)) != len(directions) or any(direction not in DELTAS for direction in directions):
        raise ValueError("graph directions must be a nonempty unique set of compass actions")
    direction_count = len(directions)
    true_nodes = {int(node) for node in graph["nodes"]}
    if any(node < 1 or node > rows * cols for node in true_nodes):
        raise ValueError("True node IDs must be inside the configured grid")
    true_edges = {edge_key(*edge) for edge in graph["edges"]}
    if any(u == v or u not in true_nodes or v not in true_nodes for u, v in true_edges):
        raise ValueError("True edges must join two distinct retained nodes")
    if any(v not in _grid_neighbors(u, rows, cols, directions) for u, v in true_edges):
        raise ValueError("True edges must join nodes adjacent under graph.directions")
    true_adjacency = defaultdict(set)
    for u, v in true_edges:
        true_adjacency[u].add(v)
        true_adjacency[v].add(u)
    train_edge_counts = graph.get("edge_counts", {})
    train_node_counts = graph.get("node_counts", {})

    edge_counts, edge_kind_counts = Counter(), defaultdict(Counter)
    edge_confidences, edge_direction_counts = defaultdict(list), defaultdict(Counter)
    node_counts, reference_node_counts, reference_edge_counts = Counter(), Counter(), Counter()
    origin_counts, predicted_node_counts = Counter(), Counter()
    source_counts, source_illegal_counts = Counter(), Counter()
    sample_edge_presence = Counter()
    sample_node_presence = Counter()
    kind_counts, confidence_by_kind = Counter(), defaultdict(list)
    fake_topology_events = sample_unique_fake_sum = 0
    illegal_baseline_sum = inbounds_baseline_sum = 0.0
    inbounds_baseline_count = 0
    prefixes = []
    running_edges, running_nodes, running_solid_edges = set(), set(), set()
    sample_ids = set()
    for sample_index, sample in enumerate(samples, 1):
        sample_id = sample.get("id", sample_index - 1)
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample ID: {sample_id!r}")
        sample_ids.add(sample_id)
        origin = int(sample["origin"])
        node_counts[origin] += 1
        origin_counts[origin] += 1
        sample_nodes = {origin}
        sample_edges, sample_fakes = set(), set()
        new_edges = set()
        before_true = len(running_edges & true_edges)
        before_nodes = len(running_nodes & true_nodes)
        running_nodes.add(origin)
        for event in sample.get("events", []):
            kind = event["kind"]
            if kind not in KINDS:
                raise ValueError(f"Unknown reconstruction event kind: {kind!r}")
            source, target = int(event["source"]), int(event["target"])
            edge = edge_key(source, target)
            confidence = float(event["probe_confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Probe confidence must be finite and between zero and one")
            if kind == "legal_correct" and edge not in true_edges:
                raise ValueError("A legal_correct event must traverse a true edge")
            edge_counts[edge] += 1
            edge_kind_counts[edge][kind] += 1
            edge_confidences[edge].append(confidence)
            edge_direction_counts[edge][str(event["direction"])] += 1
            kind_counts[kind] += 1
            confidence_by_kind[kind].append(confidence)
            source_counts[source] += 1
            source_illegal_counts[source] += kind == "illegal"
            illegal_baseline_sum += 1 - len(true_adjacency[source]) / direction_count
            grid_degree = len(_grid_neighbors(source, rows, cols, directions))
            if grid_degree:
                inbounds_baseline_sum += 1 - len(true_adjacency[source]) / grid_degree
                inbounds_baseline_count += 1
            node_counts[target] += 1
            predicted_node_counts[target] += 1
            sample_nodes.add(target)
            sample_edges.add(edge)
            if edge not in running_edges:
                new_edges.add(edge)
                running_edges.add(edge)
            running_nodes.update((source, target))
            if kind == "legal_correct":
                running_solid_edges.add(edge)
            if edge not in true_edges:
                fake_topology_events += 1
                sample_fakes.add(edge)
        sample_unique_fake_sum += len(sample_fakes)
        sample_edge_presence.update(sample_edges)
        sample_node_presence.update(sample_nodes)
        reference_nodes = [int(node) for node in sample.get("reference_nodes", [])]
        reference_node_counts.update(reference_nodes)
        reference_edge_counts.update(edge_key(u, v) for u, v in zip(reference_nodes, reference_nodes[1:]))
        prefix_steps = sum(kind_counts.values())
        prefix_correct = len(running_edges & true_edges)
        prefix_precision = _ratio(prefix_correct, len(running_edges))
        prefix_recall = _ratio(prefix_correct, len(true_edges))
        prefixes.append({
            "sample_count": sample_index, "sample_id": sample_id,
            "direction_steps": prefix_steps,
            "error_events": kind_counts["illegal"] + kind_counts["legal_mismatch"],
            "fake_topology_events": fake_topology_events,
            "unique_reconstructed_edges": len(running_edges),
            "true_edges_recovered": prefix_correct,
            "fake_edges": len(running_edges - true_edges),
            "new_fake_edges": len(new_edges - true_edges),
            "new_true_edges": prefix_correct - before_true,
            "new_true_nodes": len(running_nodes & true_nodes) - before_nodes,
            "edge_precision": prefix_precision, "edge_recall": prefix_recall,
            "edge_f1": _ratio(2 * prefix_correct, len(running_edges) + len(true_edges))
                if running_edges else None,
            "node_coverage": _ratio(len(running_nodes & true_nodes), len(true_nodes)),
        })

    total_steps = sum(kind_counts.values())
    sample_count = len(samples)
    reconstructed_edges = set(edge_counts)
    recovered_edges = reconstructed_edges & true_edges
    fake_edges = reconstructed_edges - true_edges
    recovered_nodes = running_nodes & true_nodes
    errors = kind_counts["illegal"] + kind_counts["legal_mismatch"]
    edge_training_total = sum(_edge_count(train_edge_counts, edge) for edge in true_edges)
    node_training_total = sum(_count(train_node_counts, node) for node in true_nodes)
    grid_max_edges = sum(len(_grid_neighbors(node, rows, cols, directions))
                         for node in range(1, rows * cols + 1)) // 2
    visited_max_edges = sum(target in true_nodes for node in true_nodes
                            for target in _grid_neighbors(node, rows, cols, directions)) // 2
    summary = {
        "sample_count": sample_count, "direction_steps": total_steps,
        "true_node_count": len(true_nodes), "true_edge_count": len(true_edges),
        "grid_node_count": rows * cols, "direction_count": direction_count,
        "grid_possible_edge_count": grid_max_edges,
        "visited_possible_grid_edge_count": visited_max_edges,
        "graph_density_full_grid": _ratio(len(true_edges), grid_max_edges),
        "graph_density_visited_nodes": _ratio(len(true_edges), visited_max_edges),
        "legal_correct_events": kind_counts["legal_correct"],
        "illegal_events": kind_counts["illegal"],
        "legal_mismatch_events": kind_counts["legal_mismatch"],
        "error_events": errors, "fake_topology_events": fake_topology_events,
        "error_event_rate": _ratio(errors, total_steps),
        "illegal_event_rate": _ratio(kind_counts["illegal"], total_steps),
        "legal_mismatch_event_rate": _ratio(kind_counts["legal_mismatch"], total_steps),
        "legal_mismatch_rate_given_legal_direction": _ratio(kind_counts["legal_mismatch"],
            kind_counts["legal_correct"] + kind_counts["legal_mismatch"]),
        "fake_topology_event_rate": _ratio(fake_topology_events, total_steps),
        "mean_error_events_per_sample": _ratio(errors, sample_count),
        "mean_fake_topology_events_per_sample": _ratio(fake_topology_events, sample_count),
        "mean_sample_unique_fake_topology_edges": _ratio(sample_unique_fake_sum, sample_count),
        "mean_new_fake_topology_edges_per_sample": _ratio(len(fake_edges), sample_count),
        "unique_reconstructed_edges": len(reconstructed_edges),
        "true_edges_recovered": len(recovered_edges), "fake_edges": len(fake_edges),
        "missing_true_edges": len(true_edges - recovered_edges),
        "fake_edges_per_true_edge": _ratio(len(fake_edges), len(true_edges)),
        "edge_precision": _ratio(len(recovered_edges), len(reconstructed_edges)),
        "edge_recall": _ratio(len(recovered_edges), len(true_edges)),
        "edge_f1": _ratio(2 * len(recovered_edges), len(reconstructed_edges) + len(true_edges))
            if reconstructed_edges else None,
        "edge_jaccard": _ratio(len(recovered_edges), len(reconstructed_edges | true_edges)),
        "solid_edges_recovered": len(running_solid_edges),
        "solid_edge_recall": _ratio(len(running_solid_edges), len(true_edges)),
        "reconstructed_node_count": len(running_nodes),
        "true_nodes_recovered": len(recovered_nodes),
        "unretained_nodes_predicted": len(running_nodes - true_nodes),
        "node_coverage": _ratio(len(recovered_nodes), len(true_nodes)),
        "predicted_target_node_coverage": _ratio(len(set(predicted_node_counts) & true_nodes), len(true_nodes)),
        "node_precision": _ratio(len(recovered_nodes), len(running_nodes)),
        "training_frequency_weighted_edge_recall": _ratio(
            sum(_edge_count(train_edge_counts, edge) for edge in recovered_edges), edge_training_total),
        "training_frequency_weighted_solid_edge_recall": _ratio(
            sum(_edge_count(train_edge_counts, edge) for edge in running_solid_edges), edge_training_total),
        "training_frequency_weighted_node_recall": _ratio(
            sum(_count(train_node_counts, node) for node in recovered_nodes), node_training_total),
        "uniform_direction_illegal_baseline": _ratio(illegal_baseline_sum, total_steps),
        "illegal_rate_over_uniform_direction_baseline": _ratio(kind_counts["illegal"], illegal_baseline_sum),
        "uniform8_illegal_baseline": _ratio(illegal_baseline_sum, total_steps) if direction_count == 8 else None,
        "illegal_rate_over_uniform8_baseline": (_ratio(kind_counts["illegal"], illegal_baseline_sum)
                                                 if direction_count == 8 else None),
        "uniform_inbounds_illegal_baseline": _ratio(inbounds_baseline_sum, inbounds_baseline_count),
        "uniform_inbounds_baseline_steps": inbounds_baseline_count,
        "mean_probe_confidence": _mean([value for values in confidence_by_kind.values() for value in values]),
        "mean_probe_confidence_legal_correct": _mean(confidence_by_kind["legal_correct"]),
        "mean_probe_confidence_legal_mismatch": _mean(confidence_by_kind["legal_mismatch"]),
        "mean_probe_confidence_illegal": _mean(confidence_by_kind["illegal"]),
    }
    for flag in ("physical_valid", "destination_reached", "probe_destination_reached", "route_success"):
        available = [sample[flag] for sample in samples if sample.get(flag) is not None]
        if any(type(value) is not bool for value in available):
            raise ValueError(f"{flag} must be boolean when provided")
        summary[f"{flag}_sample_count"] = len(available)
        summary[f"{flag}_rate"] = _ratio(sum(available), len(available))
    termination_counts = dict(sorted(Counter(str(sample.get("termination", "unknown")) for sample in samples).items()))
    for termination in ("eos", "max_new_tokens", "invalid_token"):
        summary[f"{termination}_sample_count"] = termination_counts.get(termination, 0)
        summary[f"{termination}_rate"] = _ratio(termination_counts.get(termination, 0), sample_count)

    edge_rows = []
    reference_steps = sum(reference_edge_counts.values())
    for edge in sorted(true_edges | reconstructed_edges):
        u, v = edge
        reconstructed = edge_counts[edge]
        train_count = _edge_count(train_edge_counts, edge)
        row = {
            "source": u, "target": v, "edge": f"{u}-{v}",
            "is_true": edge in true_edges, "training_count": train_count,
            "reconstructed_count": reconstructed,
            "reference_count": reference_edge_counts[edge],
            "sample_count": sample_edge_presence[edge],
            "legal_correct_count": edge_kind_counts[edge]["legal_correct"],
            "legal_mismatch_count": edge_kind_counts[edge]["legal_mismatch"],
            "illegal_count": edge_kind_counts[edge]["illegal"],
            "mean_probe_confidence": _mean(edge_confidences[edge]),
            "min_probe_confidence": min(edge_confidences[edge]) if edge_confidences[edge] else None,
            "max_probe_confidence": max(edge_confidences[edge]) if edge_confidences[edge] else None,
            "direction_counts": dict(sorted(edge_direction_counts[edge].items())),
            "training_usage_share": _ratio(train_count, edge_training_total),
            "reconstructed_usage_share": _ratio(reconstructed, total_steps),
            "reference_usage_share": _ratio(reference_edge_counts[edge], reference_steps),
        }
        row["usage_share_ratio_to_training"] = _ratio(
            row["reconstructed_usage_share"], row["training_usage_share"])
        edge_rows.append(row)
    node_rows = []
    total_node_visits = sum(node_counts.values())
    total_reference_visits = sum(reference_node_counts.values())
    for node in sorted(true_nodes | running_nodes):
        train_count = _count(train_node_counts, node)
        row = {
            "node": node, "is_true": node in true_nodes,
            "training_count": train_count, "reconstructed_count": node_counts[node],
            "prompt_origin_count": origin_counts[node],
            "predicted_target_count": predicted_node_counts[node],
            "training_start_count": _count(graph.get("start_counts", {}), node),
            "training_end_count": _count(graph.get("end_counts", {}), node),
            "reference_count": reference_node_counts[node],
            "sample_count": sample_node_presence[node],
            "true_degree": len(true_adjacency[node]),
            "source_opportunities": source_counts[node],
            "illegal_events_from_node": source_illegal_counts[node],
            "illegal_event_rate_from_node": _ratio(source_illegal_counts[node], source_counts[node]),
            "training_usage_share": _ratio(train_count, node_training_total),
            "reconstructed_usage_share": _ratio(node_counts[node], total_node_visits),
            "reference_usage_share": _ratio(reference_node_counts[node], total_reference_visits),
        }
        row["usage_share_ratio_to_training"] = _ratio(
            row["reconstructed_usage_share"], row["training_usage_share"])
        node_rows.append(row)
    degree_rows = []
    for degree in sorted({len(true_adjacency[node]) for node in source_counts}):
        sources = [node for node in source_counts if len(true_adjacency[node]) == degree]
        exposures = sum(source_counts[node] for node in sources)
        illegal = sum(source_illegal_counts[node] for node in sources)
        degree_rows.append({"true_degree": degree, "direction_steps": exposures,
                            "illegal_events": illegal, "illegal_event_rate": _ratio(illegal, exposures),
                            "uniform_direction_illegal_baseline": 1 - degree / direction_count,
                            "uniform8_illegal_baseline": 1 - degree / 8 if direction_count == 8 else None})
    result = {
        "summary": summary, "termination_counts": termination_counts,
        "edge_rows": edge_rows, "node_rows": node_rows,
        "rarity": {"edges": _rarity(edge_rows, thresholds, True),
                   "nodes": _rarity(node_rows, thresholds, False)},
        "degree_exposure": degree_rows, "prefixes": prefixes,
        "definitions": {
            "error_events": "Illegal directions plus legal directions whose post-token probe disagrees with the graph successor.",
            "fake_topology_events": "Every traversal whose inferred undirected source-target edge is absent from the frozen true graph; includes repeated traversals and self loops.",
            "fake_edges": "Distinct absent undirected source-target edges, counting a repeatedly drawn edge only once.",
            "mean_new_fake_topology_edges_per_sample": "Distinct absent edges in the union divided by sample count; prefix rows attribute first appearances in sample order.",
            "mean_sample_unique_fake_topology_edges": "Average number of distinct absent edges within each sample; an edge shared by two samples counts in each.",
            "edge_recall": "Recovered true undirected edges / all true edges. All line types contribute; solid_edge_recall restricts to legal_correct events.",
            "training_frequency_weighted_edge_recall": "Training traversals on recovered true edges / all training traversals. This complements, rather than replaces, unweighted coverage.",
            "usage_share_ratio_to_training": "Reconstruction relative frequency / training relative frequency; descriptive usage, not a correctness score. Undefined for unseen training items or empty reconstruction.",
            "uniform_direction_illegal_baseline": "Mean 1-degree(current inferred node)/configured direction count over generated direction opportunities; includes boundary-invalid moves.",
            "uniform8_illegal_baseline": "Legacy eight-direction alias; null for four-direction experiments.",
            "uniform_inbounds_illegal_baseline": "Mean missing-edge fraction among geometrically in-bounds compass actions at generated source nodes. Excludes nodes with no in-bounds action.",
            "graph_density_full_grid": "True edges / all undirected edges allowed by the configured directions in the entire rectangular grid.",
            "graph_density_visited_nodes": "True edges / potential configured-direction edges whose endpoints were both retained.",
            "fake_edges_per_true_edge": "Distinct absent edges / true edge count, indicating corruption relative to map size without assigning arbitrary density penalties.",
            "reference_count": "Usage in held-out reference walks, an exposure baseline; zero exposure does not establish that an item was forgotten.",
            "node_coverage": "Fraction of retained nodes visited by reconstruction, including supplied origins. predicted_target_node_coverage excludes supplied origins and counts only post-action probe predictions.",
            "probe_confidence": "Probe softmax confidence is diagnostic; it does not establish location correctness after an invalid history.",
            "undefined_rates": "A zero denominator produces null, not zero. Empty-reconstruction precision and F1 are also null.",
            "topology_distortion": "Unordered pairs of retained nodes, shortest unweighted paths in the true versus all-line reconstruction. Paths may traverse any inferred node; no unreachable distance is treated as zero.",
        },
    }
    result["topology_distortion"] = (_topology_distortion(true_nodes, true_edges, reconstructed_edges, topology_limit)
        if compute_topology else {"computed": False, "reason": "disabled"})
    return result
