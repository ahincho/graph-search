# City Pathfinding — Search Algorithm Comparison

Academic demonstration of search algorithms applied to optimal route finding
in a weighted city road network graph using [search-library](https://github.com/ahincho/search-library).

## Problem Description

Given a network of cities connected by roads with known distances (in km),
find the **optimal route** (minimum total distance) between a source and
destination city.

This is a classic **graph search problem** with the following properties:

- **State space**: Set of cities (nodes)
- **Actions**: Travel along a road to an adjacent city
- **Transition cost**: Road distance in kilometers (edge weight)
- **Goal**: Reach the destination city with minimum total cost

### Why This Problem?

Road network pathfinding is a canonical application of graph search because:

1. It has **non-uniform edge costs** (unlike grid problems with unit cost)
2. Multiple paths exist between endpoints, enabling algorithm comparison
3. Geographic coordinates provide a natural **admissible heuristic**
4. Results are intuitive and easy to verify

---

## Graph Model

### Network Structure

The graph represents cities in **southern Peru** connected by major roads:

```
                    Ayacucho
                   /        \
              Nazca          Abancay
             /                    \
    Arequipa ---- Juliaca ---- Cusco
         \        /      \      /
          \      /         Puno
           \    /
         Moquegua
          /    \
        Ilo    Tacna
```

### Cities (Nodes)

| City | Coordinates (x, y) km | Role in Network |
|------|----------------------|-----------------|
| Arequipa | (0, 0) | Origin — major hub |
| Puno | (280, 50) | Intermediate — lake region |
| Juliaca | (260, 80) | Junction — multiple routes |
| Cusco | (380, 300) | Destination — mountain city |
| Tacna | (200, -280) | Southern terminus |
| Moquegua | (120, -160) | Southern intermediate |
| Ilo | (140, -220) | Coastal city |
| Nazca | (-350, 200) | Western corridor |
| Ayacucho | (-200, 400) | Northern corridor |
| Abancay | (200, 350) | Mountain junction |

### Road Connections (Edges)

| Route | Distance (km) | Corridor |
|-------|--------------|----------|
| Arequipa — Puno | 300 | Main eastern |
| Puno — Juliaca | 45 | Regional |
| Juliaca — Cusco | 340 | Main eastern |
| Arequipa — Juliaca | 290 | Direct northern |
| Arequipa — Moquegua | 220 | Southern |
| Moquegua — Tacna | 160 | Southern |
| Moquegua — Ilo | 90 | Coastal |
| Ilo — Tacna | 120 | Coastal |
| Arequipa — Nazca | 560 | Western |
| Nazca — Ayacucho | 400 | Western |
| Cusco — Abancay | 190 | Mountain |
| Abancay — Ayacucho | 450 | Mountain |
| Cusco — Puno | 390 | Southern highland |

### Design Rationale

The graph includes a deliberate **trap route**:

> Arequipa → Nazca → Ayacucho → Abancay → Cusco = **1,600 km**

This path has only 4 hops but costs significantly more than the optimal route.
It demonstrates why **hop count ≠ optimal cost** and why BFS fails on weighted graphs.

---

## Heuristic Function

### Euclidean Distance (Straight-Line)

The heuristic `h(n)` computes the straight-line distance from city `n` to the goal:

```
h(n) = √((x_n - x_goal)² + (y_n - y_goal)²)
```

### Admissibility Proof

A heuristic is **admissible** if it never overestimates the true cost:

```
∀n: h(n) ≤ h*(n)
```

where `h*(n)` is the actual shortest-path cost from `n` to the goal.

**Proof**: The Euclidean distance is the shortest possible path between two
points in 2D space (a straight line). Since actual roads:
- Curve around terrain
- Climb and descend mountains
- Follow non-straight corridors

The road distance is always ≥ the Euclidean distance. Therefore,
the heuristic never overestimates. ∎

### Consistency (Monotonicity)

A heuristic is **consistent** if for every node `n` and successor `n'`:

```
h(n) ≤ c(n, n') + h(n')
```

**Proof**: By the triangle inequality of Euclidean distance:

```
||n - goal|| ≤ ||n - n'|| + ||n' - goal||
```

Since `c(n, n') ≥ ||n - n'||` (road distance ≥ straight-line distance):

```
h(n) ≤ ||n - n'|| + h(n') ≤ c(n, n') + h(n')
```

This guarantees A* never re-expands nodes. ∎

---

## Algorithms Compared

### 1. A* Search

- **Strategy**: Best-first search using `f(n) = g(n) + h(n)`
- **Optimality**: ✓ (with admissible heuristic)
- **Completeness**: ✓
- **Key advantage**: Explores fewer nodes than Dijkstra by using heuristic guidance

### 2. Dijkstra's Algorithm

- **Strategy**: Best-first search using `f(n) = g(n)` (equivalent to A* with h=0)
- **Optimality**: ✓
- **Completeness**: ✓
- **Key property**: Explores nodes in order of increasing cost from source

### 3. Breadth-First Search (BFS)

- **Strategy**: Level-order expansion (FIFO queue)
- **Optimality**: ✗ for weighted graphs (optimizes hop count, not cost)
- **Completeness**: ✓
- **Limitation**: Ignores edge weights; finds fewest-edge path

### 4. Depth-First Search (DFS)

- **Strategy**: Explore deepest unexpanded node first (LIFO stack)
- **Optimality**: ✗
- **Completeness**: ✓ (in finite graphs)
- **Limitation**: Returns first path found; result depends on expansion order

### 5. Bidirectional Search

- **Strategy**: BFS from both start and goal, meeting in the middle
- **Optimality**: ✗ for weighted graphs (BFS-based)
- **Completeness**: ✓
- **Key advantage**: Reduces search space from O(b^d) to O(b^(d/2))

---

## Expected Results

For the route **Arequipa → Cusco**:

| Algorithm | Cost (km) | Nodes Explored | Optimal |
|-----------|-----------|----------------|---------|
| A* | 630 | ≤6 | ✓ |
| Dijkstra | 630 | ≥6 | ✓ |
| BFS | varies | varies | ✗ |
| DFS | varies | varies | ✗ |
| Bidirectional | varies | varies | ✗ |

**Optimal path**: Arequipa → Juliaca → Cusco (290 + 340 = 630 km)

---

## Running the Example

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
cd graph-search
uv sync
```

### Run the demonstration

```bash
uv run city-pathfinding
```

### Run tests

```bash
uv run pytest
```

### Run with verbose output

```bash
uv run pytest -v
```

---

## Project Structure

```
graph-search/
├── pyproject.toml                          # Project config (uv + setuptools)
├── README.md                              # This documentation
├── src/
│   ├── __init__.py                        # Package definition
│   ├── main.py                            # Entry point — orchestrates demo
│   ├── graph_data.py                      # City network model
│   ├── heuristics.py                      # Euclidean heuristic implementation
│   └── comparison.py                      # Algorithm comparison engine
└── tests/
    ├── __init__.py
    ├── test_graph_data.py                 # Graph integrity tests
    ├── test_heuristics.py                 # Admissibility & consistency tests
    └── test_comparison.py                 # Algorithm correctness tests
```

---

## Academic Conclusions

1. **A\* is optimal AND efficient**: It finds the shortest path while exploring
   fewer nodes than Dijkstra, thanks to heuristic guidance toward the goal.

2. **Dijkstra is correct but blind**: Without directional guidance, it explores
   nodes uniformly in all directions, including away from the goal.

3. **BFS is inappropriate for weighted graphs**: It minimizes edge count, not
   total weight. In road networks where distances vary, BFS will often find
   suboptimal routes.

4. **DFS provides no guarantees**: It returns the first complete path found,
   which may be arbitrarily expensive. Its behavior depends entirely on the
   order in which successors are generated.

5. **The heuristic is the key differentiator**: The Euclidean distance provides
   just enough information to guide A* toward the goal without sacrificing
   optimality. This is the fundamental insight of informed search.

### Complexity Analysis

For a graph with V vertices and E edges:

| Algorithm | Time | Space | Optimal |
|-----------|------|-------|---------|
| A* | O((V+E) log V) | O(V) | ✓ |
| Dijkstra | O((V+E) log V) | O(V) | ✓ |
| BFS | O(V+E) | O(V) | ✗ (weighted) |
| DFS | O(V+E) | O(V) | ✗ |
| Bidirectional | O(b^(d/2)) | O(b^(d/2)) | ✗ (weighted) |

---

## References

- Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Chapter 3: Solving Problems by Searching.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.
- Dijkstra, E. W. (1959). A Note on Two Problems in Connexion with Graphs. *Numerische Mathematik*, 1, 269-271.
