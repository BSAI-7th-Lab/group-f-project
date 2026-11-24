"""Capacity-aware batch dispatch on one traffic snapshot."""
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from dijkstra_astar import Route, _search, road_snapshot, route_result
from pso import integer_swarm


@dataclass(frozen=True)
class Hospital:
    node: object
    capacity: int


@dataclass(frozen=True)
class Assignment:
    incident: str
    ambulance: str
    hospital: str
    response: Route
    transport: Route


@dataclass(frozen=True)
class DispatchResult:
    assignments: tuple[Assignment, ...]
    unassigned: tuple[str, ...]
    runtime_s: float


def _hospital_routes(roads, incident, hospitals):
    options = []
    for name, hospital in hospitals.items():
        if hospital.capacity > 0:
            started = perf_counter()
            path = _search(roads, incident, hospital.node, "dijkstra")
            if path:
                options.append((name, route_result(roads, path, started)))
    return sorted(options, key=lambda item: item[1].travel_time_s)


def select_hospital(graph, incident, hospitals, *, weight="travel_time"):
    """Return (hospital ID, route) or None when no hospital can accept."""
    options = _hospital_routes(road_snapshot(graph, weight), incident, hospitals)
    return options[0] if options else None


def dispatch(graph, ambulances, incidents, hospitals, *, weight="travel_time",
             seed=None, swarm_size=30, iterations=40):
    """Assign each available ambulance once and respect available hospital beds.

    Inputs map IDs to graph nodes, except hospitals which map IDs to Hospital.
    Optimize served count, summed response time, then summed transport time.
    """
    started = perf_counter()
    roads = road_snapshot(graph, weight)
    options = []
    for incident, node in incidents.items():
        transports = _hospital_routes(roads, node, hospitals)
        choices = []
        if transports:
            for ambulance, source in ambulances.items():
                route_started = perf_counter()
                path = _search(roads, source, node, "dijkstra")
                if path:
                    response = route_result(roads, path, route_started)
                    choices.extend(
                        Assignment(incident, ambulance, hospital, response, transport)
                        for hospital, transport in transports
                    )
        choices.sort(key=lambda choice: (
            choice.response.travel_time_s, choice.transport.travel_time_s
        ))
        options.append(choices)

    def decode(position):
        used = set()
        occupied = Counter()
        assigned = []
        for choices, index in zip(options, position):
            if index == len(choices):
                continue
            # Try the encoded choice first, then repair a resource conflict.
            ordered = choices[index:] + choices[:index] if index else choices
            for choice in ordered:
                if (choice.ambulance not in used
                        and occupied[choice.hospital] < hospitals[choice.hospital].capacity):
                    assigned.append(choice)
                    used.add(choice.ambulance)
                    occupied[choice.hospital] += 1
                    break
        return tuple(assigned)

    def score(position):
        assignments = decode(position)
        return (
            -len(assignments),
            sum(item.response.travel_time_s for item in assignments),
            sum(item.transport.travel_time_s for item in assignments),
        )

    if options:
        greedy = (0,) * len(options)
        best, _ = integer_swarm(
            score, [len(choices) + 1 for choices in options],
            seed=seed, swarm_size=swarm_size, iterations=iterations,
        )
        assignments = decode(min(greedy, best, key=score))
    else:
        assignments = ()
    served = {item.incident for item in assignments}
    return DispatchResult(
        assignments, tuple(incident for incident in incidents if incident not in served),
        perf_counter() - started,
    )
