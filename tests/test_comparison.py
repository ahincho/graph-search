"""Tests for the algorithm comparison module."""

from __future__ import annotations

import pytest
from graph_city_path_finding.comparison import (
    ComparisonReport,
    format_academic_analysis,
    format_comparison_table,
    format_detailed_paths,
    run_comparison,
)
from graph_city_path_finding.graph_data import build_city_network


class TestRunComparison:
    """Tests for the comparison execution engine."""

    @pytest.fixture
    def report(self) -> ComparisonReport:
        """Run comparison once and return the report."""
        network = build_city_network()
        return run_comparison(network)

    def test_all_algorithms_present(self, report: ComparisonReport) -> None:
        """All expected algorithms should be in the report."""
        names = report.algorithm_names
        assert "Dijkstra" in names
        assert "A*" in names
        assert "BFS" in names
        assert "DFS" in names
        assert "Bidirectional" in names

    def test_all_algorithms_find_path(self, report: ComparisonReport) -> None:
        """All algorithms should find a path for this connected graph."""
        for result in report.results:
            assert result.success, f"{result.algorithm_name} failed to find a path"

    def test_dijkstra_is_optimal(self, report: ComparisonReport) -> None:
        """Dijkstra should always find the optimal path."""
        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        assert dijkstra.is_optimal

    def test_astar_is_optimal(self, report: ComparisonReport) -> None:
        """A* with admissible heuristic should find the optimal path."""
        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert astar.is_optimal

    def test_astar_matches_dijkstra_cost(self, report: ComparisonReport) -> None:
        """A* should find the same cost as Dijkstra."""
        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert astar.total_cost == pytest.approx(dijkstra.total_cost)

    def test_astar_explores_fewer_or_equal_nodes(self, report: ComparisonReport) -> None:
        """A* should explore at most as many nodes as Dijkstra."""
        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert astar.nodes_explored <= dijkstra.nodes_explored

    def test_paths_start_at_origin(self, report: ComparisonReport) -> None:
        """All paths should start at the origin city."""
        for result in report.results:
            if result.success:
                assert result.path[0] == report.start, (
                    f"{result.algorithm_name} path doesn't start at {report.start}"
                )

    def test_paths_end_at_goal(self, report: ComparisonReport) -> None:
        """All paths should end at the goal city."""
        for result in report.results:
            if result.success:
                assert result.path[-1] == report.goal, (
                    f"{result.algorithm_name} path doesn't end at {report.goal}"
                )

    def test_optimal_cost_positive(self, report: ComparisonReport) -> None:
        """Optimal cost should be positive (non-trivial problem)."""
        assert report.optimal_cost > 0

    def test_all_costs_at_least_optimal(self, report: ComparisonReport) -> None:
        """No algorithm should find a path cheaper than optimal."""
        for result in report.results:
            if result.success:
                assert result.total_cost >= report.optimal_cost - 1e-9, (
                    f"{result.algorithm_name} has cost {result.total_cost} < optimal {report.optimal_cost}"
                )

    def test_execution_times_non_negative(self, report: ComparisonReport) -> None:
        """All execution times should be non-negative."""
        for result in report.results:
            assert result.execution_time_ms >= 0

    def test_nodes_explored_positive(self, report: ComparisonReport) -> None:
        """All successful searches should explore at least 1 node."""
        for result in report.results:
            if result.success:
                assert result.nodes_explored > 0


class TestDifferentEndpoints:
    """Tests with various start/goal combinations."""

    @pytest.mark.parametrize(
        "start,goal",
        [
            ("Arequipa", "Tacna"),
            ("Puno", "Nazca"),
            ("Cusco", "Ilo"),
            ("Moquegua", "Ayacucho"),
        ],
    )
    def test_various_routes(self, start: str, goal: str) -> None:
        """Comparison should work for various start/goal pairs."""
        network = build_city_network(start=start, goal=goal)
        report = run_comparison(network)

        # A* and Dijkstra should agree on cost
        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert dijkstra.success
        assert astar.success
        assert astar.total_cost == pytest.approx(dijkstra.total_cost)


class TestFormatters:
    """Tests for output formatting functions."""

    @pytest.fixture
    def report(self) -> ComparisonReport:
        """Generate a report for formatting tests."""
        network = build_city_network()
        return run_comparison(network)

    def test_comparison_table_not_empty(self, report: ComparisonReport) -> None:
        """Formatted table should produce non-empty output."""
        table = format_comparison_table(report)
        assert len(table) > 0
        assert "Algorithm" in table
        assert report.start in table

    def test_detailed_paths_not_empty(self, report: ComparisonReport) -> None:
        """Detailed paths should produce non-empty output."""
        paths = format_detailed_paths(report)
        assert len(paths) > 0
        assert "DETAILED PATHS" in paths

    def test_academic_analysis_not_empty(self, report: ComparisonReport) -> None:
        """Academic analysis should produce meaningful output."""
        analysis = format_academic_analysis(report)
        assert len(analysis) > 0
        assert "ACADEMIC ANALYSIS" in analysis
        assert "heuristic" in analysis.lower() or "Heuristic" in analysis
