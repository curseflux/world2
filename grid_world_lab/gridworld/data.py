"""Bounded random walks and disjoint evaluation routes on their frozen union graph.

This module uses only Python's standard library. Node IDs are one-based row-major
indices; increasing row means south. A direction always denotes an absolute
compass direction, never a relative turn.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import random
from typing import Any


DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DELTAS = {
    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
}


def edge_key(u: int, v: int) -> str:
    """Canonical key for an undirected edge."""
    return f"{min(u, v)}-{max(u, v)}"


def node_position(node: int, cols: int) -> tuple[int, int]:
    return divmod(node - 1, cols)


def bounded_neighbor(node: int, direction: str, rows: int, cols: int) -> int | None:
    """Neighbor in the original complete bounded grid, before edge removal."""
    if direction not in DELTAS or not 1 <= node <= rows * cols:
        return None
    r, c = node_position(node, cols)
    dr, dc = DELTAS[direction]
    nr, nc = r + dr, c + dc
    return nr * cols + nc + 1 if 0 <= nr < rows and 0 <= nc < cols else None


def build_neighbor_table(graph: dict[str, Any]) -> dict[int, dict[str, int]]:
    """Materialize legal absolute-direction successors, including reverse edges."""
    rows, cols = int(graph["rows"]), int(graph["cols"])
    edges = {tuple(sorted((int(u), int(v)))) for u, v in graph["edges"]}
    result: dict[int, dict[str, int]] = {}
    for raw_node in graph["nodes"]:
        node = int(raw_node)
        result[node] = {}
        for direction in DIRECTIONS:
            other = bounded_neighbor(node, direction, rows, cols)
            if other is not None and tuple(sorted((node, other))) in edges:
                result[node][direction] = other
    return result


def graph_neighbor(graph: dict[str, Any], node: int, direction: str) -> int | None:
    """Convenience one-off lookup; use build_neighbor_table in rollout loops."""
    candidate = bounded_neighbor(int(node), direction, int(graph["rows"]), int(graph["cols"]))
    if candidate is None:
        return None
    key = (min(int(node), candidate), max(int(node), candidate))
    return candidate if any(tuple(sorted((int(u), int(v)))) == key for u, v in graph["edges"]) else None


def route_key(route: dict[str, Any]) -> tuple[int, int, tuple[str, ...]]:
    return int(route["origin"]), int(route["destination"]), tuple(route["directions"])


def _rng(seed: int, name: str) -> random.Random:
    # Stable named streams make changing a split count leave the training set intact.
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _random_route(
    rng: random.Random, nodes: list[int], neighbors: dict[int, dict[str, int]],
    min_length: int, max_length: int,
) -> dict[str, Any]:
    origin = rng.choice(nodes)
    length = rng.randint(min_length, max_length)
    path, directions = [origin], []
    for _ in range(length):
        options = neighbors[path[-1]]
        if not options:
            raise ValueError(f"Cannot generate a positive-length walk from isolated node {path[-1]}.")
        direction = rng.choice(tuple(options))
        directions.append(direction)
        path.append(options[direction])
    return {"origin": origin, "destination": path[-1], "directions": directions, "nodes": path}


def _components(neighbors: dict[int, dict[str, int]]) -> list[list[int]]:
    remaining, components = set(neighbors), []
    while remaining:
        frontier, component = [min(remaining)], []
        remaining.remove(frontier[0])
        while frontier:
            node = frontier.pop()
            component.append(node)
            for other in neighbors[node].values():
                if other in remaining:
                    remaining.remove(other)
                    frontier.append(other)
        components.append(sorted(component))
    return components


def _reachable_pairs(
    neighbors: dict[int, dict[str, int]], min_length: int, max_length: int,
) -> set[tuple[int, int]]:
    """Exactly identify ordered pairs reachable at one of the allowed lengths."""
    result: set[tuple[int, int]] = set()
    for origin in neighbors:
        frontier = {origin}
        for length in range(1, max_length + 1):
            frontier = {other for node in frontier for other in neighbors[node].values()}
            if length >= min_length:
                result.update((origin, end) for end in frontier)
    return result


def _positive_int(cfg: dict[str, Any], key: str, default: int, *, allow_zero: bool = False) -> int:
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"data.{key} must be an integer >= {0 if allow_zero else 1}; got {value!r}.")
    return value


def build_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Build training walks, their exact union graph, and disjoint held-out splits.

    The training sample count is exact and duplicate training walks are allowed.
    All held-out complete token sequences are unique and excluded from training
    and every other held-out split. Test cohorts are reserved before auxiliary
    validation/probe splits. A finite attempt budget returns smaller splits with
    explicit warnings, never silently resampling training routes or changing
    cohort definitions.
    """
    rows = _positive_int(config, "rows", 10)
    cols = _positive_int(config, "cols", 10)
    if rows * cols < 2:
        raise ValueError("A positive-length walk requires at least two grid nodes.")
    train_count = _positive_int(config, "train_samples", 20000)
    min_length = _positive_int(config, "min_length", 1)
    max_length = _positive_int(config, "max_length", 32)
    if min_length > max_length:
        raise ValueError("data.min_length must not exceed data.max_length.")
    heldout_min = config.get("heldout_min_length")
    heldout_max = config.get("heldout_max_length")
    heldout_min = min_length if heldout_min is None else heldout_min
    heldout_max = max_length if heldout_max is None else heldout_max
    for key, value in (("heldout_min_length", heldout_min), ("heldout_max_length", heldout_max)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"data.{key} must be a positive integer or null.")
    if heldout_min > heldout_max:
        raise ValueError("Held-out minimum length exceeds maximum length.")
    seed = int(config.get("seed", 42))
    max_attempts = _positive_int(config, "max_attempts", 200000)
    counts = {
        "train": train_count,
        "seen": _positive_int(config, "test_samples_per_cohort", 500, allow_zero=True),
        "unseen": _positive_int(config, "test_samples_per_cohort", 500, allow_zero=True),
        "validation": _positive_int(config, "validation_samples", 1000, allow_zero=True),
        "probe_train": _positive_int(config, "probe_train_samples", 2000, allow_zero=True),
        "probe_validation": _positive_int(config, "probe_validation_samples", 500, allow_zero=True),
    }
    full_nodes = list(range(1, rows * cols + 1))
    full_neighbors = {
        node: {direction: other for direction in DIRECTIONS
               if (other := bounded_neighbor(node, direction, rows, cols)) is not None}
        for node in full_nodes
    }
    train_rng = _rng(seed, "train")
    train = []
    edge_counts: Counter[str] = Counter()
    directed_counts: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()
    starts: Counter[str] = Counter()
    ends: Counter[str] = Counter()
    edges: set[tuple[int, int]] = set()
    for index in range(train_count):
        route = _random_route(train_rng, full_nodes, full_neighbors, min_length, max_length)
        route["id"] = f"train-{index + 1}"
        train.append(route)
        node_counts.update(str(node) for node in route["nodes"])
        starts[str(route["origin"])] += 1
        ends[str(route["destination"])] += 1
        for u, v in zip(route["nodes"], route["nodes"][1:]):
            edges.add((min(u, v), max(u, v)))
            edge_counts[edge_key(u, v)] += 1
            directed_counts[f"{u}-{v}"] += 1
    graph: dict[str, Any] = {
        "rows": rows, "cols": cols,
        "nodes": sorted(int(node) for node in node_counts),
        "edges": [list(edge) for edge in sorted(edges)],
        "node_counts": dict(sorted(node_counts.items(), key=lambda item: int(item[0]))),
        "edge_counts": dict(sorted(edge_counts.items())),
        "directed_edge_counts": dict(sorted(directed_counts.items())),
        "start_counts": dict(starts), "end_counts": dict(ends),
        "undirected": True,
    }
    neighbors = build_neighbor_table(graph)
    graph["components"] = _components(neighbors)
    graph["degrees"] = {str(node): len(neighbors[node]) for node in graph["nodes"]}
    graph["full_grid_edge_count"] = sum(len(value) for value in full_neighbors.values()) // 2
    graph["edge_density"] = len(edges) / graph["full_grid_edge_count"]
    graph["node_coverage"] = len(graph["nodes"]) / (rows * cols)
    train_pairs = {(route["origin"], route["destination"]) for route in train}
    reachable_pairs = _reachable_pairs(neighbors, heldout_min, heldout_max)
    admissible = {"seen": reachable_pairs & train_pairs, "unseen": reachable_pairs - train_pairs}
    used = {route_key(route) for route in train}
    splits: dict[str, list[dict[str, Any]]] = {"train": train}
    warnings: list[str] = []
    split_stats: dict[str, Any] = {
        "train": {"requested": train_count, "actual": train_count, "attempts": train_count,
                  "unique_routes": len(used), "unique_ordered_pairs": len(train_pairs)},
    }
    for split in ("seen", "unseen", "validation", "probe_train", "probe_validation"):
        requested = counts[split]
        routes: list[dict[str, Any]] = []
        attempts = 0
        available_pairs = admissible.get(split)
        if requested and available_pairs is not None and not available_pairs:
            warnings.append(
                f"{split}: requested {requested}, produced 0; no reachable ordered origin-destination "
                f"pairs satisfy the cohort definition at lengths {heldout_min}..{heldout_max}. "
                "Try fewer training samples, a larger grid, or a different length range."
            )
        else:
            rng = _rng(seed, split)
            while len(routes) < requested and attempts < max_attempts:
                attempts += 1
                route = _random_route(rng, graph["nodes"], neighbors, heldout_min, heldout_max)
                pair = route["origin"], route["destination"]
                if available_pairs is not None and pair not in available_pairs:
                    continue
                signature = route_key(route)
                if signature in used:
                    continue
                used.add(signature)
                route["id"] = f"{split}-{len(routes) + 1}"
                routes.append(route)
            if len(routes) < requested:
                warnings.append(
                    f"{split}: requested {requested}, produced {len(routes)} after {attempts} attempts. "
                    "Unique disjoint routes may be exhausted or rare under random-walk sampling. "
                    "Increase max_attempts/max_length, reduce split sizes, or change the graph seed."
                )
        splits[split] = routes
        split_stats[split] = {
            "requested": requested, "actual": len(routes), "attempts": attempts,
            "unique_routes": len(routes),
            "unique_ordered_pairs": len({(route["origin"], route["destination"]) for route in routes}),
            "eligible_ordered_pairs": len(available_pairs) if available_pairs is not None else len(reachable_pairs),
        }
    return {
        "graph": graph, "splits": splits, "split_warnings": warnings,
        "split_stats": split_stats,
        "sampling": {"seed": seed, "train_length_range": [min_length, max_length],
                     "heldout_length_range": [heldout_min, heldout_max],
                     "pair_semantics": "ordered", "heldout_distribution": "uniform start and length; uniform legal moves; rejection by split constraints",
                     "split_reservation_order": ["seen", "unseen", "validation", "probe_train", "probe_validation"]},
    }
