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

import sys

import colorama
from colorama import Fore, Style

from graph_city_path_finding.comparison import (
    format_academic_analysis,
    format_comparison_table,
    format_detailed_paths,
    run_comparison,
)
from graph_city_path_finding.graph_data import build_city_network

# ---------------------------------------------------------------------------
# Color constants (local shortcuts)
# ---------------------------------------------------------------------------

_R = Style.RESET_ALL          # reset all
_D = Fore.LIGHTBLACK_EX       # dark gray — secondary text
_H = Style.BRIGHT              # bold — emphasis
_CYAN = Fore.LIGHTCYAN_EX      # headers, banners, step markers
_GREEN = Fore.LIGHTGREEN_EX    # success, optimal
_YELLOW = Fore.LIGHTYELLOW_EX  # list items, city names


def _configure_encoding() -> None:
    """Configure stdout for UTF-8 and initialize colorama for ANSI support."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    colorama.init()


def _banner() -> None:
    """Print the colored application banner."""
    w = 76
    border_top = f"{_CYAN}╔{'═' * (w - 2)}╗{_R}"
    border_bot = f"{_CYAN}╚{'═' * (w - 2)}╝{_R}"
    pad = w - 4  # inner width excluding "║ " and " ║"
    line1 = "CITY PATH-FINDING — Search Algorithm Comparison Demo"
    line2 = "search-library v1.0.0 | graph-city-path-finding v1.0.0"
    print(border_top)
    print(f"{_CYAN}║ {_H}{line1:<{pad}}{_R}{_CYAN} ║{_R}")
    print(f"{_CYAN}║ {_D}{line2:<{pad}}{_R}{_CYAN} ║{_R}")
    print(border_bot)


def _step(text: str) -> None:
    """Print a colored step marker (▶)."""
    print(f"{_CYAN}▶{_R} {_H}{text}{_R}")


def _section(title: str) -> None:
    """Print a colored section divider."""
    print(f"{_CYAN}{'=' * 80}{_R}")
    print(f"  {_CYAN}{title}{_R}")
    print(f"{_CYAN}{'=' * 80}{_R}")


def main() -> None:
    """Run the complete city path-finding demonstration."""
    _configure_encoding()

    print()
    _banner()
    print()

    # --- Step 1: Build the city network ---
    _step("Building city road network...")
    network = build_city_network(start="Arequipa", goal="Cusco")
    print(f"  Cities:  {_H}{network.city_count}{_R}")
    print(f"  Network: {_D}{', '.join(network.cities)}{_R}")
    print(
        f"  Problem: Find optimal route from "
        f"{_GREEN}{network.start}{_R} to {_GREEN}{network.goal}{_R}"
    )
    print()

    # --- Step 2: Display graph structure ---
    _step("Graph structure (adjacency):")
    _print_graph_structure(network)
    print()

    # --- Step 3: Run algorithm comparison ---
    _step("Running search algorithms...")
    print()
    report = run_comparison(network)

    # --- Step 4: Display results ---
    print(format_comparison_table(report))
    print(format_detailed_paths(report))

    # --- Step 5: Academic analysis ---
    print(format_academic_analysis(report))


def _print_graph_structure(network: object) -> None:
    """Print the graph adjacency structure in a readable colored format.

    Args:
        network: CityNetwork instance.
    """
    from graph_city_path_finding.graph_data import CityNetwork

    if not isinstance(network, CityNetwork):
        return

    for city in sorted(network.graph.nodes):
        neighbors = network.graph.neighbors(city)
        neighbor_strs = [
            f"{_YELLOW}{n}{_R}{_D}({w:.0f}km){_R}"
            for n, w in sorted(neighbors)
        ]
        print(f"  {_H}{city}{_R}: {', '.join(neighbor_strs)}")


if __name__ == "__main__":
    main()
