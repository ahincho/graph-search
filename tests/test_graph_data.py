"""Tests for the city graph data module."""

from __future__ import annotations

import pytest
from graph_city_pathfinding.graph_data import (
    CITY_COORDINATES,
    ROAD_CONNECTIONS,
    CityNetwork,
    build_city_network,
)


class TestCityCoordinates:
    """Tests for coordinate data integrity."""

    def test_all_cities_have_coordinates(self) -> None:
        """Every city referenced in connections must have coordinates."""
        cities_in_connections = set()
        for city_a, city_b, _ in ROAD_CONNECTIONS:
            cities_in_connections.add(city_a)
            cities_in_connections.add(city_b)

        for city in cities_in_connections:
            assert city in CITY_COORDINATES, f"City '{city}' has no coordinates"

    def test_coordinates_are_numeric_tuples(self) -> None:
        """All coordinates must be (float, float) tuples."""
        for city, coords in CITY_COORDINATES.items():
            assert isinstance(coords, tuple), f"{city}: coordinates not a tuple"
            assert len(coords) == 2, f"{city}: coordinates must have 2 elements"
            assert isinstance(coords[0], (int, float)), f"{city}: x not numeric"
            assert isinstance(coords[1], (int, float)), f"{city}: y not numeric"

    def test_minimum_city_count(self) -> None:
        """Network must have at least 6 cities (requirement)."""
        assert len(CITY_COORDINATES) >= 6


class TestRoadConnections:
    """Tests for road connection data integrity."""

    def test_all_weights_positive(self) -> None:
        """All road distances must be positive."""
        for city_a, city_b, weight in ROAD_CONNECTIONS:
            assert weight > 0, f"Edge {city_a}-{city_b} has non-positive weight {weight}"

    def test_no_self_loops(self) -> None:
        """No city should connect to itself."""
        for city_a, city_b, _ in ROAD_CONNECTIONS:
            assert city_a != city_b, f"Self-loop found: {city_a}"

    def test_minimum_edge_count(self) -> None:
        """Network must have enough edges for interesting paths."""
        assert len(ROAD_CONNECTIONS) >= 10


class TestBuildCityNetwork:
    """Tests for the network builder function."""

    def test_default_build(self) -> None:
        """Default build should create a valid network."""
        network = build_city_network()
        assert isinstance(network, CityNetwork)
        assert network.start == "Arequipa"
        assert network.goal == "Cusco"

    def test_custom_endpoints(self) -> None:
        """Building with custom start/goal should work."""
        network = build_city_network(start="Puno", goal="Tacna")
        assert network.start == "Puno"
        assert network.goal == "Tacna"

    def test_invalid_start_raises(self) -> None:
        """Invalid start city should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            build_city_network(start="InvalidCity", goal="Cusco")

    def test_invalid_goal_raises(self) -> None:
        """Invalid goal city should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            build_city_network(start="Arequipa", goal="InvalidCity")

    def test_graph_is_undirected(self) -> None:
        """Graph should be undirected (edges in both directions)."""
        network = build_city_network()
        # Check that Arequipa->Puno implies Puno->Arequipa
        assert network.graph.has_edge("Arequipa", "Puno")
        assert network.graph.has_edge("Puno", "Arequipa")

    def test_graph_node_count(self) -> None:
        """Graph should contain all defined cities."""
        network = build_city_network()
        assert network.city_count == len(CITY_COORDINATES)

    def test_cities_property_sorted(self) -> None:
        """Cities property should return sorted list."""
        network = build_city_network()
        cities = network.cities
        assert cities == sorted(cities)

    def test_edge_weights_match_connections(self) -> None:
        """Edge weights in graph must match defined connections."""
        network = build_city_network()
        for city_a, city_b, expected_weight in ROAD_CONNECTIONS:
            actual_weight = network.graph.get_edge_weight(city_a, city_b)
            assert actual_weight == expected_weight, (
                f"Edge {city_a}-{city_b}: expected {expected_weight}, got {actual_weight}"
            )
