"""Euclidean distance heuristic for city graph pathfinding.

This module implements a custom heuristic function compatible with
search-library's Heuristic[str] interface. It computes the straight-line
(Euclidean) distance between two cities based on their 2D coordinates.

Admissibility Proof:
    The Euclidean distance is always <= the actual road distance because:
    1. Roads are never perfectly straight (they curve, climb, detour)
    2. The straight-line distance is the theoretical minimum path length
    3. Therefore h(n) <= h*(n) for all nodes n

    This guarantees A* will find the optimal solution.

Consistency (Monotonicity):
    For any two adjacent cities A and B with road distance d(A,B):
        h(A) <= d(A,B) + h(B)

    This holds because the Euclidean distance satisfies the triangle
    inequality: ||A - G|| <= ||A - B|| + ||B - G|| for any point G.
    Since road distances >= Euclidean distances, consistency is preserved.
"""

from __future__ import annotations

import math

from search_library.heuristics.base import Heuristic


class CityEuclideanHeuristic(Heuristic[str]):
    """Euclidean distance heuristic for string-labeled city nodes.

    Maps city names to 2D coordinates and computes the straight-line
    distance as the heuristic estimate. This is admissible and consistent
    for road network problems where edge weights represent physical distances.

    Attributes:
        coordinates: Mapping of city names to (x, y) positions.
    """

    def __init__(self, coordinates: dict[str, tuple[float, float]]) -> None:
        """Initialize with city coordinate data.

        Args:
            coordinates: Dictionary mapping city name -> (x, y) in km.

        Raises:
            ValueError: If coordinates dictionary is empty.
        """
        if not coordinates:
            msg = "Coordinates dictionary must not be empty"
            raise ValueError(msg)
        self._coordinates = coordinates

    @property
    def coordinates(self) -> dict[str, tuple[float, float]]:
        """Return the coordinate mapping (read-only copy)."""
        return self._coordinates.copy()

    def estimate(self, state: str, goal: str) -> float:
        """Compute Euclidean distance between two cities.

        Args:
            state: Current city name.
            goal: Target city name.

        Returns:
            Straight-line distance in km between the two cities.
            Returns 0.0 if either city is not in the coordinate map
            (safe fallback that maintains admissibility).
        """
        if state not in self._coordinates or goal not in self._coordinates:
            return 0.0

        x1, y1 = self._coordinates[state]
        x2, y2 = self._coordinates[goal]

        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx * dx + dy * dy)


def compute_euclidean_distance(
    city_a: str,
    city_b: str,
    coordinates: dict[str, tuple[float, float]],
) -> float:
    """Standalone utility to compute Euclidean distance between two cities.

    Useful for verification and testing outside the Heuristic interface.

    Args:
        city_a: First city name.
        city_b: Second city name.
        coordinates: Coordinate mapping.

    Returns:
        Euclidean distance, or 0.0 if either city is not found.
    """
    if city_a not in coordinates or city_b not in coordinates:
        return 0.0

    x1, y1 = coordinates[city_a]
    x2, y2 = coordinates[city_b]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
