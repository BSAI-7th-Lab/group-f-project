"""Integer particle swarm and bounded candidate-route optimization."""
from itertools import islice
from math import isfinite
from random import Random
from time import perf_counter

import networkx as nx

from dijkstra_astar import road_snapshot, route_result


def integer_swarm(score, sizes, *, inertia=.7, c1=1.4, c2=1.4,
                  swarm_size=20, iterations=30, seed=None):
    """Minimize a scalar or tuple score over a bounded integer vector."""
    if (not sizes or min(sizes) < 1 or swarm_size < 1 or iterations < 0
            or any(not isfinite(value) or value < 0 for value in (inertia, c1, c2))):
        raise ValueError("Invalid swarm bounds or parameters")
    rng = Random(seed)
    limits = [size - 1 for size in sizes]
    positions = [[rng.randrange(size) for size in sizes] for _ in range(swarm_size)]
    velocities = [[rng.uniform(-limit, limit) for limit in limits] for _ in positions]
    personal = [position.copy() for position in positions]
    scores = [score(position) for position in personal]
    leader_index = min(range(swarm_size), key=scores.__getitem__)
    leader, best = personal[leader_index], scores[leader_index]
    history = [best]
    for _ in range(iterations):
        iteration_leader = leader
        for i, position in enumerate(positions):
            for j, limit in enumerate(limits):
                current = position[j]
                velocity = (
                    inertia * velocities[i][j]
                    + c1 * rng.random() * (personal[i][j] - current)
                    + c2 * rng.random() * (iteration_leader[j] - current)
                )
                velocity = max(-limit, min(limit, velocity))
                velocities[i][j] = velocity
                position[j] = max(0, min(limit, round(current + velocity)))
            value = score(position)
            if value < scores[i]:
                personal[i], scores[i] = position.copy(), value
            if value < best:
                leader, best = position.copy(), value
        history.append(best)
    return tuple(leader), tuple(history)


def pso(graph, source, target, *, weight="travel_time", candidate_limit=24,
        inertia=.4, c1=2., c2=1., swarm_size=6, iterations=20, seed=None):
    """Search the k shortest-by-distance open paths for low travel time."""
    started = perf_counter()
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    roads = road_snapshot(graph, weight)
    candidates = []
    if source in roads and target in roads:
        try:
            paths = nx.shortest_simple_paths(roads, source, target, weight="length")
            candidates = list(islice(paths, candidate_limit))
        except nx.NetworkXNoPath:
            pass
    if not candidates:
        return route_result(roads, [], started), ()
    costs = [
        sum(roads[u][v]["seconds"] for u, v in zip(path, path[1:]))
        for path in candidates
    ]
    best, history = integer_swarm(
        lambda position: costs[position[0]], [len(candidates)], inertia=inertia,
        c1=c1, c2=c2, swarm_size=swarm_size, iterations=iterations, seed=seed,
    )
    return route_result(roads, candidates[best[0]], started), history
