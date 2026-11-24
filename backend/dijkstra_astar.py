"""Travel-time baselines for a fixed road-network snapshot."""
from dataclasses import dataclass
from math import asin, cos, inf, isfinite, radians, sin, sqrt
from time import perf_counter

import networkx as nx


@dataclass(frozen=True)
class Route:
    nodes: tuple
    edges: tuple
    travel_time_s: float
    length_m: float
    runtime_s: float

    @property
    def success(self):
        return bool(self.nodes)


def road_snapshot(graph, weight="travel_time"):
    """Keep the fastest open parallel edge and preserve its original key."""
    roads = nx.DiGraph()
    roads.add_nodes_from(graph.nodes(data=True))
    edges = graph.edges(keys=True, data=True) if graph.is_multigraph() else (
        (u, v, None, data) for u, v, data in graph.edges(data=True)
    )
    for u, v, key, data in edges:
        if data.get("blocked", False):
            continue
        seconds = float(data[weight])
        if seconds == inf:
            continue
        length = float(data["length"])
        if not isfinite(seconds) or not isfinite(length) or min(seconds, length) < 0:
            raise ValueError(f"Invalid edge cost or length on {(u, v, key)}")
        directions = [(u, v)] if graph.is_directed() else [(u, v), (v, u)]
        for start, end in directions:
            previous = roads.get_edge_data(start, end)
            if previous is None or (seconds, length) < (previous["seconds"], previous["length"]):
                roads.add_edge(start, end, seconds=seconds, length=length, key=key)
    return roads


def route_result(roads, nodes, started):
    pairs = list(zip(nodes, nodes[1:]))
    data = [roads[u][v] for u, v in pairs]
    return Route(
        tuple(nodes),
        tuple((u, v, edge["key"]) for (u, v), edge in zip(pairs, data)),
        sum(edge["seconds"] for edge in data) if nodes else inf,
        sum(edge["length"] for edge in data) if nodes else inf,
        perf_counter() - started,
    )


def haversine(a, b):
    lon1, lat1, lon2, lat2 = map(radians, (a["x"], a["y"], b["x"], b["y"]))
    h = sin((lat2 - lat1) / 2) ** 2
    h += cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 12_742_000 * asin(sqrt(max(0, min(1, h))))


def _heuristic(roads):
    positions = dict(roads.nodes(data=True))
    if any(
        not (-180 <= node.get("x", inf) <= 180 and -90 <= node.get("y", inf) <= 90)
        for node in positions.values()
    ):
        return lambda _u, _v: 0
    scale = inf
    for u, v, data in roads.edges(data=True):
        metres = haversine(positions[u], positions[v])
        if metres > 0:
            scale = min(scale, data["seconds"] / metres)
    if scale == inf:
        scale = 0
    return lambda u, v: scale * haversine(positions[u], positions[v])


def _search(roads, source, target, method):
    if source not in roads or target not in roads:
        return []
    try:
        if method == "astar":
            return nx.astar_path(roads, source, target, _heuristic(roads), weight="seconds")
        return nx.dijkstra_path(roads, source, target, weight="seconds")
    except nx.NetworkXNoPath:
        return []


def dijkstra(graph, source, target, *, weight="travel_time"):
    started = perf_counter()
    roads = road_snapshot(graph, weight)
    return route_result(roads, _search(roads, source, target, "dijkstra"), started)


def astar(graph, source, target, *, weight="travel_time"):
    started = perf_counter()
    roads = road_snapshot(graph, weight)
    return route_result(roads, _search(roads, source, target, "astar"), started)
