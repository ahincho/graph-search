"""Main entry point for the city path-finding demonstration.

Orchestrates the complete example:
1. Builds the city road network graph
2. Runs all search algorithms on the same problem
3. Displays comparison table and detailed analysis
4. Prints academic conclusions

Usage:
    uv run city-path-finding
    # or
    uv run python -m graph_city_path_finding.main
"""

from __future__ import annotations

from graph_city_path_finding.comparison import (
    format_academic_analysis,
    format_comparison_table,
    format_detailed_paths,
    run_comparison,
)
from graph_city_path_finding.graph_data import build_city_network


def main() -> None:
    """Run the complete city path-finding demonstration."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     CITY PATH-FINDING — Search Algorithm Comparison Demo           ║")
    print("║     search-library v1.0.0                                          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # --- Step 1: Build the city network ---
    print("▶ Building city road network...")
    network = build_city_network(start="Arequipa", goal="Cusco")
    print(f"  Cities: {network.city_count}")
    print(f"  Network: {', '.join(network.cities)}")
    print(f"  Problem: Find optimal route from {network.start} to {network.goal}")
    print()

    # --- Step 2: Display graph structure ---
    print("▶ Graph structure (adjacency):")
    _print_graph_structure(network)
    print()

    # --- Step 3: Run algorithm comparison ---
    print("▶ Running search algorithms...")
    print()
    report = run_comparison(network)

    # --- Step 4: Display results ---
    print(format_comparison_table(report))
    print(format_detailed_paths(report))

    # --- Step 5: Academic analysis ---
    print(format_academic_analysis(report))


def _print_graph_structure(network: object) -> None:
    """Print the graph adjacency structure in a readable format.

    Args:
        network: CityNetwork instance.
    """
    from graph_city_path_finding.graph_data import CityNetwork

    if not isinstance(network, CityNetwork):
        return

    for city in sorted(network.graph.nodes):
        neighbors = network.graph.neighbors(city)
        neighbor_strs = [f"{n}({w:.0f}km)" for n, w in sorted(neighbors)]
        print(f"  {city}: {', '.join(neighbor_strs)}")


if __name__ == "__main__":
    main()
