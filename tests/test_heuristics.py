"""Tests for the city Euclidean heuristic module."""

from __future__ import annotations

import pytest
from graph_city_path_finding.graph_data import CITY_COORDINATES, ROAD_CONNECTIONS
from graph_city_path_finding.heuristics import (
    CityEuclideanHeuristic,
    compute_euclidean_distance,
)


class TestCityEuclideanHeuristic:
    """Tests for the CityEuclideanHeuristic class."""

    @pytest.fixture
    def heuristic(self) -> CityEuclideanHeuristic:
        """Create a heuristic instance with real city coordinates."""
        return CityEuclideanHeuristic(CITY_COORDINATES)

    def test_same_city_zero_distance(self, heuristic: CityEuclideanHeuristic) -> None:
        """Distance from a city to itself should be zero."""
        for city in CITY_COORDINATES:
            assert heuristic.estimate(city, city) == 0.0

    def test_non_negative(self, heuristic: CityEuclideanHeuristic) -> None:
        """All distances should be non-negative."""
        cities = list(CITY_COORDINATES.keys())
        for i, city_a in enumerate(cities):
            for city_b in cities[i + 1 :]:
                assert heuristic.estimate(city_a, city_b) >= 0.0

    def test_symmetry(self, heuristic: CityEuclideanHeuristic) -> None:
        """Euclidean distance is symmetric: d(A,B) = d(B,A)."""
        cities = list(CITY_COORDINATES.keys())
        for i, city_a in enumerate(cities):
            for city_b in cities[i + 1 :]:
                assert heuristic.estimate(city_a, city_b) == pytest.approx(
                    heuristic.estimate(city_b, city_a)
                )

    def test_admissibility(self, heuristic: CityEuclideanHeuristic) -> None:
        """Heuristic must never overestimate actual edge cost (admissibility).

        For every direct road connection, the Euclidean distance between
        the two cities must be <= the road distance (edge weight).
        """
        for city_a, city_b, road_distance in ROAD_CONNECTIONS:
            euclidean = heuristic.estimate(city_a, city_b)
            assert euclidean <= road_distance + 1e-9, (
                f"Heuristic overestimates for {city_a}-{city_b}: "
                f"h={euclidean:.1f} > actual={road_distance:.1f}"
            )

    def test_known_distance(self) -> None:
        """Test with known coordinates for verification."""
        coords = {"A": (0.0, 0.0), "B": (3.0, 4.0)}
        h = CityEuclideanHeuristic(coords)
        assert h.estimate("A", "B") == pytest.approx(5.0)

    def test_unknown_city_returns_zero(self, heuristic: CityEuclideanHeuristic) -> None:
        """Unknown city should return 0.0 (safe fallback)."""
        assert heuristic.estimate("Arequipa", "UnknownCity") == 0.0
        assert heuristic.estimate("UnknownCity", "Cusco") == 0.0

    def test_empty_coordinates_raises(self) -> None:
        """Empty coordinates dict should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            CityEuclideanHeuristic({})

    def test_callable_interface(self, heuristic: CityEuclideanHeuristic) -> None:
        """Heuristic should be callable (via __call__ from base class)."""
        result = heuristic("Arequipa", "Cusco")
        assert result > 0.0
        assert result == heuristic.estimate("Arequipa", "Cusco")

    def test_coordinates_property_returns_copy(self, heuristic: CityEuclideanHeuristic) -> None:
        """Coordinates property should return a copy (immutability)."""
        coords = heuristic.coordinates
        coords["NewCity"] = (0.0, 0.0)
        assert "NewCity" not in heuristic.coordinates


class TestComputeEuclideanDistance:
    """Tests for the standalone distance utility function."""

    def test_basic_distance(self) -> None:
        """Test basic Euclidean distance computation."""
        coords = {"A": (0.0, 0.0), "B": (3.0, 4.0), "C": (1.0, 0.0)}
        assert compute_euclidean_distance("A", "B", coords) == pytest.approx(5.0)
        assert compute_euclidean_distance("A", "C", coords) == pytest.approx(1.0)

    def test_unknown_city_returns_zero(self) -> None:
        """Unknown cities should return 0.0."""
        coords = {"A": (0.0, 0.0)}
        assert compute_euclidean_distance("A", "Z", coords) == 0.0

    def test_consistency_with_heuristic_class(self) -> None:
        """Utility function should match heuristic class output."""
        h = CityEuclideanHeuristic(CITY_COORDINATES)
        for city_a in list(CITY_COORDINATES.keys())[:5]:
            for city_b in list(CITY_COORDINATES.keys())[:5]:
                assert compute_euclidean_distance(city_a, city_b, CITY_COORDINATES) == pytest.approx(
                    h.estimate(city_a, city_b)
                )


class TestTriangleInequality:
    """Tests verifying the triangle inequality for consistency."""

    def test_triangle_inequality_holds(self) -> None:
        """Euclidean distance must satisfy triangle inequality.

        For any three cities A, B, C:
            d(A, C) <= d(A, B) + d(B, C)

        This is required for the heuristic to be consistent (monotone).
        """
        h = CityEuclideanHeuristic(CITY_COORDINATES)
        cities = list(CITY_COORDINATES.keys())

        for a in cities:
            for b in cities:
                for c in cities:
                    d_ac = h.estimate(a, c)
                    d_ab = h.estimate(a, b)
                    d_bc = h.estimate(b, c)
                    assert d_ac <= d_ab + d_bc + 1e-9, (
                        f"Triangle inequality violated: "
                        f"d({a},{c})={d_ac:.1f} > d({a},{b})={d_ab:.1f} + d({b},{c})={d_bc:.1f}"
                    )
