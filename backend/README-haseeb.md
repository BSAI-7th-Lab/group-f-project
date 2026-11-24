# Haseeb's Swarm Project modules

Requires Python 3.10+ and NetworkX 3.x. Run from the backend directory.

```powershell
python -m pip install -r requirements-pso.txt
python -m unittest discover -s tests -v
python tune_pso.py
```

| Call | Result |
| --- | --- |
| `dijkstra(graph, source, target)` in `dijkstra_astar.py` | Timed shortest route |
| `astar(graph, source, target)` in `dijkstra_astar.py` | Timed shortest route using a Haversine lower bound |
| `pso(graph, source, target, seed=0)` in `pso.py` | Timed route and best travel time per swarm iteration, including initialization |
| `select_hospital(graph, incident, hospitals)` in `dispatch.py` | Fastest reachable hospital with capacity and its route, or `None` |
| `dispatch(graph, ambulances, incidents, hospitals, seed=0)` in `dispatch.py` | Assignments with response and transport routes, unassigned incident IDs, and runtime |

The graph can be a NetworkX directed graph or OSMnx MultiDiGraph. Required edge attributes are `length` in metres and `travel_time` in seconds. Pass `weight="your_attribute"` if the environment stores its effective time under another name. Edges marked `blocked=True` or assigned infinite effective time are excluded. For parallel edges, the fastest open edge is used and its original edge key is returned. A* uses longitude and latitude from node attributes `x` and `y` on an unprojected graph. Strip projected coordinates or transform to longitude and latitude before calling A*.

Each route has `success`, `nodes`, `edges` as (u, v, key) tuples, `travel_time_s`, `length_m`, and `runtime_s`. Route failure returns empty sequences and infinite time and length. Routing and dispatch read one traffic snapshot per call. Run them again after the environment updates traffic.

Ambulances and incidents are dictionaries mapping IDs to snapped graph nodes. Hospitals map IDs to `Hospital(node, capacity)`, where capacity is the count of available beds. Dispatch plans one batch without mutating availability. Its caller owns the state update if a plan is accepted.

```python
from dispatch import Hospital, dispatch
from pso import pso
from tune_pso import synthetic_graph

graph = synthetic_graph()
route, convergence = pso(graph, 0, 49, seed=0)
plan = dispatch(
    graph,
    {"A1": 0, "A2": 49},
    {"I1": 13, "I2": 35},
    {"H1": Hospital(9, 1), "H2": Hospital(40, 1)},
    seed=0,
)
for assignment in plan.assignments:
    print(assignment.incident, assignment.ambulance, assignment.hospital)
```

Hassan can import `synthetic_graph()` from `tune_pso.py` to prototype ACO on the same 50-node graph. See [report draft](docs/haseeb-report.md) for methods, tuning results, and literature.
