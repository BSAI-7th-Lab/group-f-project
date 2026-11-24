import math
import unittest

import networkx as nx

from dijkstra_astar import astar, dijkstra
from dispatch import Hospital, dispatch, select_hospital
from pso import pso
from tune_pso import synthetic_graph


class RoutingTests(unittest.TestCase):
    def test_parallel_edges_and_live_closure(self):
        graph = nx.MultiDiGraph()
        graph.add_node("a", x=73, y=33)
        graph.add_node("b", x=73.01, y=33)
        graph.add_edge("a", "b", key="fast", length=1000, travel_time=10)
        graph.add_edge("a", "b", key="slow", length=1200, travel_time=20)
        for route in (dijkstra(graph, "a", "b"), astar(graph, "a", "b"),
                      pso(graph, "a", "b", seed=4)[0]):
            self.assertEqual(route.edges, (("a", "b", "fast"),))
        graph["a"]["b"]["fast"]["blocked"] = True
        self.assertEqual(astar(graph, "a", "b").travel_time_s, 20)
        self.assertFalse(dijkstra(graph, "b", "a").success)
        self.assertTrue(math.isinf(pso(graph, "b", "a")[0].travel_time_s))
        self.assertEqual(pso(graph, "a", "a")[0].nodes, ("a",))

    def test_synthetic_baselines_and_seeded_swarm(self):
        graph = synthetic_graph()
        self.assertEqual(len(graph), 50)
        exact = dijkstra(graph, 0, 49)
        self.assertAlmostEqual(exact.travel_time_s, astar(graph, 0, 49).travel_time_s)
        route, history = pso(graph, 0, 49, seed=0)
        replicated, repeated_history = pso(graph, 0, 49, seed=0)
        self.assertTrue(route.success)
        self.assertEqual(route.edges, replicated.edges)
        self.assertEqual(history, repeated_history)
        self.assertGreaterEqual(route.travel_time_s + 1e-10, exact.travel_time_s)
        self.assertEqual(history[-1], route.travel_time_s)
        self.assertEqual(history, tuple(sorted(history, reverse=True)))

    def test_dispatch_capacity_and_unique_ambulances(self):
        graph = nx.DiGraph()
        for u, v, seconds in [
            ("a1", "i1", 2), ("a1", "i2", 3), ("a2", "i1", 4),
            ("i1", "h1", 5), ("i1", "h2", 8), ("i2", "h1", 1),
        ]:
            graph.add_edge(u, v, length=100, travel_time=seconds)
        graph.add_node("i3")
        hospitals = {"full": Hospital("i1", 0), "near": Hospital("h1", 1),
                     "far": Hospital("h2", 1)}
        self.assertEqual(select_hospital(graph, "i1", hospitals)[0], "near")
        result = dispatch(
            graph, {"A": "a1", "B": "a2"}, {"I1": "i1", "I2": "i2", "I3": "i3"},
            hospitals, seed=0,
        )
        self.assertEqual(result.unassigned, ("I3",))
        self.assertEqual(
            {(item.incident, item.ambulance, item.hospital) for item in result.assignments},
            {("I1", "B", "far"), ("I2", "A", "near")},
        )
        self.assertEqual(hospitals["near"].capacity, 1)


if __name__ == "__main__":
    unittest.main()
