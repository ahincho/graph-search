"""Algorithm comparison engine for city path-finding.

This module provides structured comparison of search algorithms on the same
problem instance, collecting metrics and generating formatted output suitable
for academic analysis.

Metrics collected per algorithm:
- Path found (sequence of cities)
- Total cost (sum of edge weights = total km)
- Number of nodes explored (search efficiency)
- Execution time (wall-clock, optional)
- Optimality verdict (compared against known optimal cost)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from search_library import (
    SearchResult,
    astar_search,
    bfs_search,
    bidirectional_search,
    dfs_search,
    dijkstra_search,
)

from graph_city_path_finding.graph_data import CityNetwork
from graph_city_path_finding.heuristics import CityEuclideanHeuristic


@dataclass(frozen=True)
class AlgorithmResult:
    """Stores the result of running a single algorithm on the problem.

    Attributes:
        algorithm_name: Human-readable algorithm name.
        path: List of cities in the found path.
        total_cost: Total distance in km.
        nodes_explored: Number of nodes expanded during search.
        execution_time_ms: Wall-clock time in milliseconds.
        is_optimal: Whether this path has the minimum possible cost.
        success: Whether a path was found at all.
    """

    algorithm_name: str
    path: list[str]
    total_cost: float
    nodes_explored: int
    execution_time_ms: float
    is_optimal: bool
    success: bool


@dataclass
class ComparisonReport:
    """Aggregated comparison of all algorithms on the same problem.

    Attributes:
        start: Origin city.
        goal: Destination city.
        optimal_cost: The minimum achievable cost (from Dijkstra/A*).
        results: Individual results per algorithm.
    """

    start: str
    goal: str
    optimal_cost: float
    results: list[AlgorithmResult] = field(default_factory=list)

    @property
    def algorithm_names(self) -> list[str]:
        """Return list of algorithm names in comparison order."""
        return [r.algorithm_name for r in self.results]


def run_comparison(network: CityNetwork) -> ComparisonReport:
    """Execute all algorithms on the city network and compare results.

    Runs BFS, DFS, Dijkstra, and A* on the same problem instance,
    collecting performance metrics for each. The optimal cost is
    determined by Dijkstra (which is guaranteed optimal for non-negative
    edge weights without needing a heuristic).

    Args:
        network: The CityNetwork containing graph, coordinates, and endpoints.

    Returns:
        ComparisonReport with individual results and optimal cost reference.
    """
    heuristic = CityEuclideanHeuristic(network.coordinates)
    graph = network.graph
    start = network.start
    goal = network.goal

    results: list[AlgorithmResult] = []

    # --- 1. Dijkstra (baseline: guaranteed optimal) ---
    dijkstra_result = _run_timed(
        "Dijkstra",
        lambda: dijkstra_search(graph.to_search_problem(start, goal)),
    )
    results.append(dijkstra_result)
    optimal_cost = dijkstra_result.total_cost

    # --- 2. A* Search (informed: optimal with admissible heuristic) ---
    astar_result = _run_timed(
        "A*",
        lambda: astar_search(graph.to_search_problem(start, goal, heuristic)),
    )
    results.append(astar_result)

    # --- 3. BFS (uninformed: optimal by hop count, not by cost) ---
    bfs_result = _run_timed(
        "BFS",
        lambda: bfs_search(graph.to_search_problem(start, goal)),
    )
    results.append(bfs_result)

    # --- 4. DFS (uninformed: no optimality guarantee) ---
    dfs_result = _run_timed(
        "DFS",
        lambda: dfs_search(graph.to_search_problem(start, goal)),
    )
    results.append(dfs_result)

    # --- 5. Bidirectional BFS ---
    forward_problem = graph.to_search_problem(start, goal)
    reverse_problem = graph.to_search_problem(goal, start)
    bidir_result = _run_timed(
        "Bidirectional",
        lambda: bidirectional_search(
            forward_problem,
            reverse_problem=reverse_problem,
        ),
    )
    results.append(bidir_result)

    # Mark optimality for each result
    final_results = [
        AlgorithmResult(
            algorithm_name=r.algorithm_name,
            path=r.path,
            total_cost=r.total_cost,
            nodes_explored=r.nodes_explored,
            execution_time_ms=r.execution_time_ms,
            is_optimal=_is_close(r.total_cost, optimal_cost) if r.success else False,
            success=r.success,
        )
        for r in results
    ]

    return ComparisonReport(
        start=start,
        goal=goal,
        optimal_cost=optimal_cost,
        results=final_results,
    )


def format_comparison_table(report: ComparisonReport) -> str:
    """Format the comparison report as a readable ASCII table.

    Args:
        report: The ComparisonReport to format.

    Returns:
        Multi-line string with formatted comparison table.
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 90)
    lines.append(f"  ALGORITHM COMPARISON: {report.start} → {report.goal}")
    lines.append(f"  Optimal cost (Dijkstra baseline): {report.optimal_cost:.1f} km")
    lines.append("=" * 90)
    lines.append("")

    # Table header
    header = f"{'Algorithm':<15} {'Cost (km)':<12} {'Nodes':<8} {'Time (ms)':<12} {'Optimal':<9} {'Path'}"
    lines.append(header)
    lines.append("-" * 90)

    # Table rows
    for result in report.results:
        if result.success:
            path_str = " → ".join(result.path)
            if len(path_str) > 35:
                path_str = path_str[:32] + "..."
            optimal_mark = "✓" if result.is_optimal else "✗"
            row = (
                f"{result.algorithm_name:<15} "
                f"{result.total_cost:<12.1f} "
                f"{result.nodes_explored:<8} "
                f"{result.execution_time_ms:<12.4f} "
                f"{optimal_mark:<9} "
                f"{path_str}"
            )
        else:
            row = f"{result.algorithm_name:<15} {'N/A':<12} {'N/A':<8} {'N/A':<12} {'✗':<9} No path found"
        lines.append(row)

    lines.append("-" * 90)
    lines.append("")
    return "\n".join(lines)


def format_detailed_paths(report: ComparisonReport) -> str:
    """Format detailed path information for each algorithm.

    Args:
        report: The ComparisonReport to format.

    Returns:
        Multi-line string with detailed path breakdown.
    """
    lines: list[str] = []
    lines.append("DETAILED PATHS")
    lines.append("-" * 50)

    for result in report.results:
        lines.append(f"\n  {result.algorithm_name}:")
        if result.success:
            lines.append(f"    Path: {' → '.join(result.path)}")
            lines.append(f"    Hops: {len(result.path) - 1}")
            lines.append(f"    Cost: {result.total_cost:.1f} km")
            lines.append(f"    Explored: {result.nodes_explored} nodes")
        else:
            lines.append("    No path found")

    lines.append("")
    return "\n".join(lines)


def format_academic_analysis(report: ComparisonReport) -> str:
    """Generate academic analysis explaining the results.

    Args:
        report: The ComparisonReport to analyze.

    Returns:
        Multi-line string with academic explanation.
    """
    lines: list[str] = []
    lines.append("ACADEMIC ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    # Find specific results
    results_by_name = {r.algorithm_name: r for r in report.results}
    dijkstra = results_by_name.get("Dijkstra")
    astar = results_by_name.get("A*")
    bfs = results_by_name.get("BFS")
    dfs = results_by_name.get("DFS")

    # A* vs Dijkstra efficiency
    if astar and dijkstra and astar.success and dijkstra.success:
        efficiency_gain = dijkstra.nodes_explored - astar.nodes_explored
        if efficiency_gain > 0:
            pct = (efficiency_gain / dijkstra.nodes_explored) * 100
            lines.append("1. A* vs Dijkstra (Informed vs Uninformed Optimal Search)")
            lines.append(f"   Both find the optimal path (cost = {dijkstra.total_cost:.1f} km).")
            lines.append(f"   A* explored {astar.nodes_explored} nodes vs Dijkstra's {dijkstra.nodes_explored}.")
            lines.append(f"   A* is {pct:.0f}% more efficient due to heuristic guidance.")
            lines.append("   The Euclidean heuristic prunes branches that lead away from the goal.")
        else:
            lines.append("1. A* vs Dijkstra")
            lines.append(f"   Both find optimal path (cost = {dijkstra.total_cost:.1f} km).")
            lines.append("   In this small graph, both explore similar node counts.")
        lines.append("")

    # BFS limitations
    if bfs and dijkstra and bfs.success and dijkstra.success:
        lines.append("2. BFS Limitations in Weighted Graphs")
        if not bfs.is_optimal:
            cost_diff = bfs.total_cost - dijkstra.total_cost
            lines.append(f"   BFS found a path costing {bfs.total_cost:.1f} km (suboptimal by {cost_diff:.1f} km).")
            lines.append("   BFS minimizes HOP COUNT, not total cost.")
            lines.append("   It selects the path with fewest edges, ignoring edge weights.")
        else:
            lines.append(f"   BFS found the optimal path ({bfs.total_cost:.1f} km) by coincidence.")
            lines.append("   In general, BFS does NOT guarantee cost-optimality in weighted graphs.")
            lines.append("   It minimizes hop count, which happened to align with minimal cost here.")
        lines.append("")

    # DFS characteristics
    if dfs and dijkstra and dfs.success and dijkstra.success:
        lines.append("3. DFS: Completeness Without Optimality")
        if not dfs.is_optimal:
            lines.append(f"   DFS found a path costing {dfs.total_cost:.1f} km (non-optimal).")
        else:
            lines.append(f"   DFS found the optimal path ({dfs.total_cost:.1f} km) by luck.")
        lines.append("   DFS explores depth-first and returns the FIRST path found.")
        lines.append("   It has no mechanism to prefer shorter or cheaper paths.")
        lines.append("   Its result depends on successor ordering, not path quality.")
        lines.append("")

    # Heuristic role
    lines.append("4. Role of the Heuristic Function")
    lines.append("   h(n) = Euclidean distance from city n to the goal city.")
    lines.append("   Properties:")
    lines.append("     - Admissible: straight-line ≤ road distance (never overestimates)")
    lines.append("     - Consistent: satisfies triangle inequality")
    lines.append("     - Effect: guides A* toward the goal, reducing unnecessary exploration")
    lines.append("   Without a heuristic (h=0), A* degenerates into Dijkstra.")
    lines.append("")

    # Conclusion
    lines.append("CONCLUSION")
    lines.append("-" * 70)
    lines.append("  For weighted graph path-finding with spatial structure:")
    lines.append("  • A* is the best choice — optimal AND efficient with a good heuristic")
    lines.append("  • Dijkstra is correct but explores more nodes (no directional guidance)")
    lines.append("  • BFS is inappropriate — optimizes hop count, not distance")
    lines.append("  • DFS is unreliable — no optimality or efficiency guarantees")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_timed(
    name: str,
    search_fn: object,
) -> AlgorithmResult:
    """Run a search function with timing.

    Args:
        name: Algorithm name for labeling.
        search_fn: Callable that returns a SearchResult.

    Returns:
        AlgorithmResult with timing information.
    """
    start_time = time.perf_counter()
    result: SearchResult[str] = search_fn()  # type: ignore[operator]
    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000.0

    return AlgorithmResult(
        algorithm_name=name,
        path=list(result.path),
        total_cost=result.total_cost,
        nodes_explored=result.nodes_explored,
        execution_time_ms=elapsed_ms,
        is_optimal=False,  # Will be set later in run_comparison
        success=result.success,
    )


def _is_close(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    """Check if two floats are approximately equal.

    Args:
        a: First value.
        b: Second value.
        rel_tol: Relative tolerance.

    Returns:
        True if values are within tolerance.
    """
    return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1.0)
