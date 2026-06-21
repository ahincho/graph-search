"""Algorithm comparison engine for city path-finding.

This module provides structured comparison of search algorithms on the same
problem instance, collecting metrics and generating colored formatted output
suitable for academic analysis.

Metrics collected per algorithm:
- Path found (sequence of cities)
- Total cost (sum of edge weights = total km)
- Number of nodes explored (search efficiency)
- Execution time (wall-clock in milliseconds)
- Optimality verdict (compared against known optimal cost)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from colorama import Fore, Style
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

# ---------------------------------------------------------------------------
# Color constants (module-level for reuse across all format functions)
# ---------------------------------------------------------------------------

_R = Style.RESET_ALL          # reset all
_D = Fore.LIGHTBLACK_EX       # dark gray — secondary text (descriptions, separators)
_H = Style.BRIGHT              # bold — headers and key values
_CYAN = Fore.LIGHTCYAN_EX      # section titles, scenario names
_GREEN = Fore.LIGHTGREEN_EX    # optimal / success
_RED = Fore.LIGHTRED_EX        # suboptimal / failure
_YELLOW = Fore.LIGHTYELLOW_EX  # bullets, warnings


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Comparison Engine
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Formatting (colored)
# ---------------------------------------------------------------------------


def format_comparison_table(report: ComparisonReport) -> str:
    """Format the comparison report as a colored ASCII table.

    Padding rule: format plain string with f-spec first, then wrap in color.

    Args:
        report: The ComparisonReport to format.

    Returns:
        Multi-line colored string with formatted comparison table.
    """
    lines: list[str] = []

    # Header — format plain strings first, then wrap in color
    lines.append(f"{_CYAN}{'=' * 90}{_R}")
    lines.append(
        f"  {_CYAN}ALGORITHM COMPARISON:{_R} "
        f"{_H}{report.start} → {report.goal}{_R}"
    )
    lines.append(
        f"  {_D}Optimal cost (Dijkstra baseline):{_R} "
        f"{_GREEN}{report.optimal_cost:.1f} km{_R}"
    )
    lines.append(f"{_CYAN}{'=' * 90}{_R}")
    lines.append("")

    # Table header — format plain strings first, then wrap in bold
    h_algo  = f"{'Algorithm':<15}"
    h_cost  = f"{'Cost (km)':<12}"
    h_nodes = f"{'Nodes':<8}"
    h_time  = f"{'Time (ms)':<12}"
    h_opt   = f"{'Optimal':<9}"
    h_path  = "Path"
    lines.append(f"  {_H}{h_algo} {h_cost} {h_nodes} {h_time} {h_opt} {h_path}{_R}")
    lines.append(f"  {_D}{'-' * 88}{_R}")

    for result in report.results:
        # Format all fields as plain strings (preserves alignment), then color
        algo_str = f"{result.algorithm_name:<15}"
        if result.success:
            cost_str  = f"{result.total_cost:<12.1f}"
            nodes_str = f"{result.nodes_explored:<8}"
            time_str  = f"{result.execution_time_ms:<12.4f}"

            path_str = " → ".join(result.path)
            if len(path_str) > 35:
                path_str = path_str[:32] + "..."

            # Pad plain char first, then wrap in color (rule: no color in format spec)
            opt_char   = "✓" if result.is_optimal else "✗"
            opt_padded = f"{opt_char:<9}"
            opt_color  = _GREEN if result.is_optimal else _RED
            opt_field  = f"{opt_color}{opt_padded}{_R}"

            lines.append(
                f"  {algo_str} {cost_str} {nodes_str} {time_str} {opt_field} {path_str}"
            )
        else:
            # No-path rows in dim gray
            na_cost  = f"{'N/A':<12}"
            na_nodes = f"{'N/A':<8}"
            na_time  = f"{'N/A':<12}"
            na_opt   = f"{'—':<9}"
            lines.append(
                f"  {_D}{algo_str} {na_cost} {na_nodes} {na_time} {na_opt} No path found{_R}"
            )

    lines.append(f"  {_D}{'-' * 88}{_R}")
    lines.append("")
    return "\n".join(lines)


def format_detailed_paths(report: ComparisonReport) -> str:
    """Format detailed path information for each algorithm (colored).

    Args:
        report: The ComparisonReport to format.

    Returns:
        Multi-line colored string with detailed path breakdown.
    """
    lines: list[str] = []
    lines.append(f"{_CYAN}DETAILED PATHS{_R}")
    lines.append(f"{_D}{'-' * 50}{_R}")

    for result in report.results:
        lines.append(f"\n  {_H}{result.algorithm_name}{_R}:")
        if result.success:
            lines.append(f"    Path: {' → '.join(result.path)}")
            lines.append(f"    Hops: {_H}{len(result.path) - 1}{_R}")
            cost_color = _GREEN if result.is_optimal else _RED
            lines.append(f"    Cost: {cost_color}{result.total_cost:.1f} km{_R}")
            lines.append(f"    Explored: {_H}{result.nodes_explored}{_R} nodes")
        else:
            lines.append(f"    {_D}No path found{_R}")

    lines.append("")
    return "\n".join(lines)


def format_academic_analysis(report: ComparisonReport) -> str:
    """Generate colored academic analysis explaining the results.

    Args:
        report: The ComparisonReport to analyze.

    Returns:
        Multi-line colored string with academic explanation.
    """
    lines: list[str] = []
    lines.append(f"{_CYAN}{'=' * 70}{_R}")
    lines.append(f"  {_CYAN}ACADEMIC ANALYSIS{_R}")
    lines.append(f"{_CYAN}{'=' * 70}{_R}")
    lines.append("")

    results_by_name = {r.algorithm_name: r for r in report.results}
    dijkstra = results_by_name.get("Dijkstra")
    astar    = results_by_name.get("A*")
    bfs      = results_by_name.get("BFS")
    dfs      = results_by_name.get("DFS")

    # --- 1. A* vs Dijkstra ---
    if astar and dijkstra and astar.success and dijkstra.success:
        lines.append(f"  {_H}1. A* vs Dijkstra (Informed vs Uninformed Optimal Search){_R}")
        lines.append(f"  {_D}{'-' * 60}{_R}")
        efficiency_gain = dijkstra.nodes_explored - astar.nodes_explored
        if efficiency_gain > 0:
            pct = (efficiency_gain / dijkstra.nodes_explored) * 100
            savings_color = _GREEN if pct >= 50 else _YELLOW
            lines.append(
                f"   Both find the optimal path "
                f"(cost = {_GREEN}{dijkstra.total_cost:.1f} km{_R})."
            )
            lines.append(
                f"   A* explored {_H}{astar.nodes_explored}{_R} nodes "
                f"vs Dijkstra's {_H}{dijkstra.nodes_explored}{_R}."
            )
            lines.append(
                f"   A* is {savings_color}{pct:.0f}% more efficient{_R} due to heuristic guidance."
            )
            lines.append(
                "   The Euclidean heuristic prunes branches that lead away from the goal."
            )
        else:
            lines.append(
                f"   Both find optimal path (cost = {_GREEN}{dijkstra.total_cost:.1f} km{_R})."
            )
            lines.append("   In this small graph, both explore similar node counts.")
        lines.append("")

    # --- 2. BFS limitations ---
    if bfs and dijkstra and bfs.success and dijkstra.success:
        lines.append(f"  {_H}2. BFS Limitations in Weighted Graphs{_R}")
        lines.append(f"  {_D}{'-' * 60}{_R}")
        if not bfs.is_optimal:
            cost_diff = bfs.total_cost - dijkstra.total_cost
            lines.append(
                f"   BFS found a path costing {_RED}{bfs.total_cost:.1f} km{_R} "
                f"(suboptimal by {_RED}{cost_diff:.1f} km{_R})."
            )
            lines.append("   BFS minimizes HOP COUNT, not total cost.")
            lines.append("   It selects the path with fewest edges, ignoring edge weights.")
        else:
            lines.append(
                f"   BFS found the optimal path ({_GREEN}{bfs.total_cost:.1f} km{_R}) by coincidence."
            )
            lines.append("   In general, BFS does NOT guarantee cost-optimality in weighted graphs.")
            lines.append("   It minimizes hop count, which happened to align with minimal cost here.")
        lines.append("")

    # --- 3. DFS characteristics ---
    if dfs and dijkstra and dfs.success and dijkstra.success:
        lines.append(f"  {_H}3. DFS: Completeness Without Optimality{_R}")
        lines.append(f"  {_D}{'-' * 60}{_R}")
        if not dfs.is_optimal:
            ratio = dfs.total_cost / dijkstra.total_cost if dijkstra.total_cost > 0 else 1.0
            dfs_color = _RED if ratio > 2 else _YELLOW
            lines.append(
                f"   DFS found a path costing {dfs_color}{dfs.total_cost:.1f} km{_R} "
                f"({dfs_color}{ratio:.2f}x optimal{_R})."
            )
        else:
            lines.append(
                f"   DFS found the optimal path ({_GREEN}{dfs.total_cost:.1f} km{_R}) by luck."
            )
        lines.append("   DFS explores depth-first and returns the FIRST path found.")
        lines.append("   It has no mechanism to prefer shorter or cheaper paths.")
        lines.append("   Its result depends on successor ordering, not path quality.")
        lines.append("")

    # --- 4. Heuristic role ---
    lines.append(f"  {_H}4. Role of the Heuristic Function{_R}")
    lines.append(f"  {_D}{'-' * 60}{_R}")
    lines.append("   h(n) = Euclidean distance from city n to the goal city.")
    lines.append("   Properties:")
    lines.append(f"     {_YELLOW}•{_R} Admissible: straight-line ≤ road distance (never overestimates)")
    lines.append(f"     {_YELLOW}•{_R} Consistent: satisfies triangle inequality")
    lines.append(f"     {_YELLOW}•{_R} Effect: guides A* toward the goal, reducing unnecessary exploration")
    lines.append("   Without a heuristic (h=0), A* degenerates into Dijkstra.")
    lines.append("")

    # --- Conclusion ---
    lines.append(f"  {_CYAN}{'=' * 60}{_R}")
    lines.append(f"  {_CYAN}CONCLUSION — For weighted graph path-finding:{_R}")
    lines.append(f"  {_CYAN}{'=' * 60}{_R}")
    lines.append(f"  {_YELLOW}•{_R} {_GREEN}A*{_R}           — optimal AND efficient with a good heuristic")
    lines.append(f"  {_YELLOW}•{_R} Dijkstra     — correct but explores more nodes (no directional guidance)")
    lines.append(f"  {_YELLOW}•{_R} BFS          — inappropriate: optimizes hop count, not distance")
    lines.append(f"  {_YELLOW}•{_R} DFS          — unreliable: no optimality or efficiency guarantees")
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
