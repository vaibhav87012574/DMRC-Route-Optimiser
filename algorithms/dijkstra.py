"""
Dijkstra's Algorithm Implementation for Delhi Metro Shortest Path Finder
=========================================================================
This module implements Dijkstra's algorithm to find the shortest path
between any two stations in the Delhi Metro network.

Dijkstra's Algorithm Overview:
- Uses a priority queue (min-heap) to always process the nearest unvisited node
- Maintains a distance table for all nodes (initialized to infinity)
- Updates distances when a shorter path is discovered
- Records the step-by-step process for educational visualization
"""

import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional


def dijkstra(
    graph: Dict[str, Dict[str, float]],
    source: str,
    destination: str
) -> Tuple[List[str], float, List[dict]]:
    """
    Find the shortest path between source and destination using Dijkstra's algorithm.

    Args:
        graph: Adjacency dict {node: {neighbor: weight}}
        source: Starting station name
        destination: Target station name

    Returns:
        Tuple of:
          - path (list of station names)
          - total_distance (float)
          - steps (list of dicts describing each algorithm step)
    """

    # --- Initialization ---
    # Distance to every node starts at infinity (unknown)
    distances: Dict[str, float] = {node: float('inf') for node in graph}
    distances[source] = 0.0  # Source has distance 0

    # To reconstruct the path, track where we came from
    predecessors: Dict[str, Optional[str]] = {node: None for node in graph}

    # Priority queue: (distance, node)
    # heapq is a min-heap: always pops the smallest distance first
    priority_queue: List[Tuple[float, str]] = [(0.0, source)]

    # Track visited nodes so we don't re-process them
    visited: set = set()

    # Record each step for the educational visualizer
    steps: List[dict] = []

    # Step counter
    step_num = 1

    while priority_queue:
        # Pop the node with the smallest current known distance
        current_dist, current_node = heapq.heappop(priority_queue)

        # Skip already-visited nodes (stale entries in heap)
        if current_node in visited:
            continue

        # Mark as visited
        visited.add(current_node)

        # Build the priority queue snapshot (top 5 items for display)
        pq_snapshot = sorted(
            [(d, n) for d, n in priority_queue if n not in visited],
            key=lambda x: x[0]
        )[:5]

        # Record this step for the visualizer
        step_info = {
            "step": step_num,
            "current_node": current_node,
            "current_distance": current_dist,
            "visited_nodes": list(visited),
            "updates": [],
            "priority_queue": [(round(d, 2), n) for d, n in pq_snapshot],
            "distances_snapshot": {
                k: (round(v, 2) if v != float('inf') else "∞")
                for k, v in sorted(distances.items())
                if v != float('inf')  # Only show reachable nodes
            }
        }

        # If we've reached the destination, we can stop early
        if current_node == destination:
            steps.append(step_info)
            break

        # --- Relaxation Step ---
        # Explore all neighbors of the current node
        if current_node in graph:
            for neighbor, weight in graph[current_node].items():
                if neighbor in visited:
                    continue

                # Calculate tentative distance through current node
                tentative_dist = current_dist + weight

                # If this path is shorter, update it
                if tentative_dist < distances[neighbor]:
                    old_dist = distances[neighbor]
                    distances[neighbor] = tentative_dist
                    predecessors[neighbor] = current_node

                    # Push updated distance to priority queue
                    heapq.heappush(priority_queue, (tentative_dist, neighbor))

                    # Record this update for educational display
                    step_info["updates"].append({
                        "node": neighbor,
                        "old_distance": round(old_dist, 2) if old_dist != float('inf') else "∞",
                        "new_distance": round(tentative_dist, 2),
                        "via": current_node
                    })

        steps.append(step_info)
        step_num += 1

    # --- Path Reconstruction ---
    # Trace back from destination to source using predecessors
    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = predecessors.get(node)

    path.reverse()  # Reverse to get source → destination order

    # If path doesn't start with source, no path exists
    if not path or path[0] != source:
        return [], float('inf'), steps

    total_distance = distances[destination]
    return path, total_distance, steps


def bfs_path(
    graph: Dict[str, Dict[str, float]],
    source: str,
    destination: str
) -> Tuple[List[str], float]:
    """
    Find a path using Breadth-First Search (BFS).
    BFS finds the path with fewest hops (stations), not minimum distance.
    Used for comparison with Dijkstra's result.

    Args:
        graph: Adjacency dict {node: {neighbor: weight}}
        source: Starting station name
        destination: Target station name

    Returns:
        Tuple of (path as list, total_distance as float)
    """
    if source not in graph:
        return [], float('inf')

    # BFS queue: each entry is (current_node, path_so_far)
    queue = deque([(source, [source])])
    visited = {source}

    while queue:
        current_node, path = queue.popleft()

        if current_node == destination:
            # Calculate total distance for this path
            total_dist = 0.0
            for i in range(len(path) - 1):
                total_dist += graph[path[i]].get(path[i + 1], 0)
            return path, round(total_dist, 2)

        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return [], float('inf')  # No path found


def calculate_fare(distance: float) -> int:
    """
    Calculate Delhi Metro fare based on distance slabs (DMRC fare chart).

    Args:
        distance: Distance in kilometers

    Returns:
        Fare in Indian Rupees (₹)
    """
    if distance <= 2:
        return 10
    elif distance <= 5:
        return 20
    elif distance <= 12:
        return 30
    elif distance <= 21:
        return 40
    elif distance <= 32:
        return 50
    else:
        return 60


def calculate_travel_time(path: List[str], connections_df) -> int:
    """
    Calculate total travel time for a given path using connection data.

    Args:
        path: List of station names
        connections_df: DataFrame with travel time information

    Returns:
        Total travel time in minutes (includes interchange time)
    """
    import pandas as pd

    total_time = 0

    # Create a lookup dict for fast access: (source, dest) -> travel_time
    time_lookup = {}
    for _, row in connections_df.iterrows():
        time_lookup[(row['source'], row['destination'])] = row['travel_time']

    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        travel_time = time_lookup.get((src, dst), 2)  # Default 2 mins
        total_time += travel_time

    # Add interchange penalty (3 mins per interchange)
    interchanges = count_interchanges(path)
    total_time += interchanges * 3

    return total_time


def count_interchanges(path: List[str]) -> int:
    """
    Count the number of line changes (interchanges) in a path.
    An interchange occurs at multi-line stations like Rajiv Chowk, Kashmere Gate.

    Args:
        path: List of station names

    Returns:
        Number of interchanges
    """
    # Multi-line stations (interchange points)
    interchange_stations = {
        "Rajiv Chowk", "Kashmere Gate", "Central Secretariat",
        "INA", "Hauz Khas", "Kirti Nagar", "Azadpur", "New Delhi",
        "Yamuna Bank", "Netaji Subhash Place", "Inderlok"
    }

    interchange_count = 0
    for station in path[1:-1]:  # Exclude source and destination
        if station in interchange_stations:
            interchange_count += 1

    return interchange_count
