"""
Graph Builder for Delhi Metro Network
======================================
Builds a NetworkX graph from the stations and connections CSV files.
Provides utility functions for querying the metro network.
"""

import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional


# Line color mapping for visualization
LINE_COLORS = {
    "Red": "#e74c3c",
    "Yellow": "#f1c40f",
    "Blue": "#2980b9",
    "Green": "#27ae60",
    "Violet": "#8e44ad",
    "Orange": "#e67e22",
    "Magenta": "#d63031",
    "Pink": "#fd79a8",
    "Airport": "#636e72",
    "Blue/Yellow": "#2980b9",
    "Blue/Pink": "#2980b9",
    "Blue/Green": "#2980b9",
    "Yellow/Airport": "#f1c40f",
}


def load_data(stations_path: str, connections_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load station and connection data from CSV files.

    Args:
        stations_path: Path to stations.csv
        connections_path: Path to connections.csv

    Returns:
        Tuple of (stations_df, connections_df)
    """
    stations_df = pd.read_csv(stations_path)
    connections_df = pd.read_csv(connections_path)
    return stations_df, connections_df


def build_graph(stations_df: pd.DataFrame, connections_df: pd.DataFrame) -> nx.Graph:
    """
    Build a NetworkX undirected weighted graph from the metro data.

    Each node represents a station with attributes:
      - line: metro line color
      - latitude, longitude: geographic coordinates

    Each edge represents a track with attributes:
      - distance: in km
      - travel_time: in minutes

    Args:
        stations_df: DataFrame with station info
        connections_df: DataFrame with connection info

    Returns:
        NetworkX Graph object
    """
    G = nx.Graph()

    # Add all stations as nodes
    for _, row in stations_df.iterrows():
        G.add_node(
            row['station_name'],
            line=row['line'],
            latitude=row['latitude'],
            longitude=row['longitude'],
            station_id=row['station_id']
        )

    # Add all connections as weighted edges
    for _, row in connections_df.iterrows():
        src = row['source']
        dst = row['destination']

        # Only add edge if both stations exist in our graph
        if src in G.nodes and dst in G.nodes:
            # Use distance as edge weight for Dijkstra
            G.add_edge(
                src, dst,
                distance=row['distance'],
                travel_time=row['travel_time'],
                weight=row['distance']  # Primary weight for shortest path
            )

    return G


def build_adjacency_dict(
    connections_df: pd.DataFrame,
    weight_column: str = 'distance'
) -> Dict[str, Dict[str, float]]:
    """
    Build a plain adjacency dictionary for our custom Dijkstra implementation.

    Format: {station: {neighbor: weight}}

    Args:
        connections_df: DataFrame with connection info
        weight_column: Column to use as edge weight ('distance' or 'travel_time')

    Returns:
        Adjacency dictionary
    """
    adj: Dict[str, Dict[str, float]] = {}

    for _, row in connections_df.iterrows():
        src = row['source']
        dst = row['destination']
        weight = float(row[weight_column])

        if src not in adj:
            adj[src] = {}
        if dst not in adj:
            adj[dst] = {}

        adj[src][dst] = weight
        adj[dst][src] = weight  # Bidirectional

    return adj


def get_station_line(station_name: str, stations_df: pd.DataFrame) -> str:
    """
    Get the metro line for a given station.

    Args:
        station_name: Name of the station
        stations_df: Stations DataFrame

    Returns:
        Line name as string
    """
    match = stations_df[stations_df['station_name'] == station_name]
    if not match.empty:
        return match.iloc[0]['line']
    return "Unknown"


def get_station_color(station_name: str, stations_df: pd.DataFrame) -> str:
    """
    Get the display color for a station based on its metro line.

    Args:
        station_name: Name of the station
        stations_df: Stations DataFrame

    Returns:
        Hex color string
    """
    line = get_station_line(station_name, stations_df)
    return LINE_COLORS.get(line, "#636e72")


def get_path_lines(path: List[str], stations_df: pd.DataFrame) -> List[str]:
    """
    Get the metro lines for each station in a path.

    Args:
        path: List of station names
        stations_df: Stations DataFrame

    Returns:
        List of line names corresponding to path stations
    """
    return [get_station_line(station, stations_df) for station in path]


def get_interchange_stations(path: List[str], stations_df: pd.DataFrame) -> List[str]:
    """
    Identify interchange stations in a path where passengers must change lines.

    Args:
        path: List of station names
        stations_df: Stations DataFrame

    Returns:
        List of interchange station names
    """
    lines = get_path_lines(path, stations_df)
    interchanges = []

    for i in range(1, len(path) - 1):
        # A station is an interchange if the line changes at it
        if lines[i] != lines[i - 1] or lines[i] != lines[i + 1]:
            if path[i] not in interchanges:
                interchanges.append(path[i])

    return interchanges


def get_graph_statistics(G: nx.Graph) -> dict:
    """
    Get basic statistics about the metro graph.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary of statistics
    """
    return {
        "total_stations": G.number_of_nodes(),
        "total_connections": G.number_of_edges(),
        "is_connected": nx.is_connected(G),
        "average_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        "total_network_distance": round(
            sum(d['distance'] for _, _, d in G.edges(data=True)), 2
        )
    }
