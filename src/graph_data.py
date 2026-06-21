"""City road network graph model.

This module defines a realistic road network between cities in southern Peru,
modeled as a weighted undirected graph. Edge weights represent approximate
driving distances in kilometers.

The graph is intentionally designed with:
- Multiple paths between source and destination (to show algorithm differences)
- At least one suboptimal "trap" route (shorter hop count but higher total cost)
- Realistic 2D coordinates for heuristic computation

City Coordinates:
    Approximate geographic positions (simplified to 2D plane in km units)
    used for Euclidean heuristic estimation. Coordinates are scaled from
    real-world latitude/longitude to a planar projection.
"""

from __future__ import annotations

from dataclasses import dataclass

from search_library import Graph


@dataclass(frozen=True)
class CityNetwork:
    """Encapsulates the city graph and associated spatial metadata.

    Attributes:
        graph: Weighted undirected graph where nodes are city names.
        coordinates: Mapping of city name to (x, y) position in km.
        start: Default origin city for path-finding.
        goal: Default destination city for path-finding.
    """

    graph: Graph[str]
    coordinates: dict[str, tuple[float, float]]
    start: str
    goal: str

    @property
    def cities(self) -> list[str]:
        """Return sorted list of all cities in the network."""
        return sorted(self.graph.nodes)

    @property
    def city_count(self) -> int:
        """Return total number of cities."""
        return self.graph.node_count


# ---------------------------------------------------------------------------
# 2D Coordinates (approximate planar positions in km)
# ---------------------------------------------------------------------------
# These represent a simplified projection of cities in southern Peru.
# The coordinate system uses km as units, enabling direct Euclidean
# distance computation that serves as an admissible heuristic.

CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "Arequipa": (0.0, 0.0),
    "Puno": (280.0, 50.0),
    "Juliaca": (260.0, 80.0),
    "Cusco": (380.0, 300.0),
    "Tacna": (200.0, -280.0),
    "Moquegua": (120.0, -160.0),
    "Ilo": (140.0, -220.0),
    "Nazca": (-350.0, 200.0),
    "Ayacucho": (-200.0, 400.0),
    "Abancay": (200.0, 350.0),
}

# ---------------------------------------------------------------------------
# Road connections with approximate distances (km)
# ---------------------------------------------------------------------------
# Each tuple: (city_a, city_b, distance_km)
# The graph is undirected — roads can be traveled in both directions.

ROAD_CONNECTIONS: list[tuple[str, str, float]] = [
    # Main corridor: Arequipa - Puno - Cusco
    ("Arequipa", "Puno", 300.0),
    ("Puno", "Juliaca", 45.0),
    ("Juliaca", "Cusco", 340.0),
    # Direct route: Arequipa - Juliaca (slightly longer than via Puno)
    ("Arequipa", "Juliaca", 290.0),
    # Southern corridor: Arequipa - Moquegua - Tacna
    ("Arequipa", "Moquegua", 220.0),
    ("Moquegua", "Tacna", 160.0),
    ("Moquegua", "Ilo", 90.0),
    ("Ilo", "Tacna", 120.0),
    # Northern corridor: Arequipa - Nazca - Ayacucho
    ("Arequipa", "Nazca", 560.0),
    ("Nazca", "Ayacucho", 400.0),
    # Cusco connections (mountain routes)
    ("Cusco", "Abancay", 190.0),
    ("Abancay", "Ayacucho", 450.0),
    ("Cusco", "Puno", 390.0),
    # Trap route: Arequipa -> Nazca -> Ayacucho -> Abancay -> Cusco
    # Total: 560 + 400 + 450 + 190 = 1600 km (much longer than optimal)
    # This demonstrates why uninformed search can find suboptimal paths.
]


def build_city_network(
    start: str = "Arequipa",
    goal: str = "Cusco",
) -> CityNetwork:
    """Build the complete city road network.

    Constructs a weighted undirected graph representing the road network
    between cities in southern Peru. Each edge weight represents the
    approximate driving distance in kilometers.

    Args:
        start: Origin city for path-finding (default: Arequipa).
        goal: Destination city for path-finding (default: Cusco).

    Returns:
        CityNetwork with the graph, coordinates, and search endpoints.

    Raises:
        ValueError: If start or goal city is not in the network.
    """
    graph: Graph[str] = Graph(directed=False)

    # Add all cities as nodes first
    for city in CITY_COORDINATES:
        graph.add_node(city)

    # Add road connections as weighted edges
    for city_a, city_b, distance in ROAD_CONNECTIONS:
        graph.add_edge(city_a, city_b, distance)

    # Validate start and goal
    if not graph.has_node(start):
        msg = f"Start city '{start}' not found in network. Available: {sorted(graph.nodes)}"
        raise ValueError(msg)
    if not graph.has_node(goal):
        msg = f"Goal city '{goal}' not found in network. Available: {sorted(graph.nodes)}"
        raise ValueError(msg)

    return CityNetwork(
        graph=graph,
        coordinates=CITY_COORDINATES.copy(),
        start=start,
        goal=goal,
    )
