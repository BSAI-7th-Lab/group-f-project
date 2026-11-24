"""Reproduce PSO tuning and evaluate the selected settings on held-out cases."""
import csv
from itertools import product
from pathlib import Path
from statistics import mean

import networkx as nx

from dijkstra_astar import astar, dijkstra, haversine
from pso import pso


def synthetic_graph():
    """Return a shared 50-node grid with geographic coordinates and traffic."""
    graph = nx.MultiDiGraph()
    for node in range(50):
        row, column = divmod(node, 10)
        graph.add_node(node, x=73.05 + column * .001, y=33.68 + row * .001)
    for node in graph:
        row, column = divmod(node, 10)
        neighbors = []
        if column < 9:
            neighbors.append(node + 1)
        if row < 4:
            neighbors.append(node + 10)
        for neighbor in neighbors:
            length = haversine(graph.nodes[node], graph.nodes[neighbor])
            for start, end in ((node, neighbor), (neighbor, node)):
                traffic = 1 + ((start * 17 + end * 13) % 9) / 4
                graph.add_edge(
                    start, end, length=length,
                    travel_time=length / 12 * traffic,
                    blocked=(start * 7 + end * 19) % 37 == 0,
                )
    return graph


def evaluate(graph, cases, settings):
    gaps, runtimes = [], []
    for source, target, optimum in cases:
        for seed in (0, 1, 2):
            route, _ = pso(graph, source, target, **settings, iterations=20, seed=seed)
            gaps.append(100 * (route.travel_time_s / optimum - 1))
            runtimes.append(route.runtime_s * 1000)
    return mean(gaps), mean(runtimes), len(gaps)


def tune(output):
    graph = synthetic_graph()

    def with_optima(pairs):
        cases = []
        for source, target in pairs:
            exact = dijkstra(graph, source, target)
            assert abs(exact.travel_time_s - astar(graph, source, target).travel_time_s) < 1e-8
            cases.append((source, target, exact.travel_time_s))
        return cases

    training = with_optima([(0, 49), (9, 40), (14, 37), (43, 6)])
    held_out = with_optima([(2, 38), (47, 11)])
    keys = ("inertia", "c1", "c2", "swarm_size")
    rows = []
    for values in product((.4, .8), (1., 2.), (1., 2.), (6, 16)):
        settings = dict(zip(keys, values))
        gap, runtime, runs = evaluate(graph, training, settings)
        rows.append(("tuning", *values, runs, gap, runtime))
    winner = min(rows, key=lambda row: (round(row[-2], 10), row[4], row[1:4]))
    values = winner[1:5]
    gap, runtime, runs = evaluate(graph, held_out, dict(zip(keys, values)))
    rows.append(("holdout", *values, runs, gap, runtime))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", *keys, "runs", "mean_gap_pct", "mean_runtime_ms", "selected"))
        for row in rows:
            writer.writerow((*row[:-2], f"{row[-2]:.6f}", f"{row[-1]:.3f}", row == winner or row[0] == "holdout"))


if __name__ == "__main__":
    tune(Path(__file__).with_name("pso_tuning.csv"))
