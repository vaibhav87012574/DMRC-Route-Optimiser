"""
Network Graph Visualization for Delhi Metro
=============================================
Builds interactive Plotly figures for the metro network,
highlighting the shortest path and selected stations.
"""

import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from typing import List, Optional

# Metro line colors for visualization
LINE_COLORS = {
    "Red": "#e74c3c",
    "Yellow": "#f1c40f",
    "Blue": "#3498db",
    "Green": "#2ecc71",
    "Violet": "#9b59b6",
    "Orange": "#e67e22",
    "Magenta": "#e91e63",
    "Pink": "#fd79a8",
    "Airport": "#95a5a6",
    "Blue/Yellow": "#3498db",
    "Blue/Pink": "#3498db",
    "Blue/Green": "#3498db",
    "Yellow/Airport": "#f1c40f",
}


def build_metro_figure(
    G: nx.Graph,
    stations_df: pd.DataFrame,
    path: Optional[List[str]] = None,
    source: Optional[str] = None,
    destination: Optional[str] = None,
    dark_mode: bool = True
) -> go.Figure:
    """
    Build an interactive Plotly figure of the Delhi Metro network.

    Args:
        G: NetworkX graph of the metro network
        stations_df: DataFrame with station lat/lon info
        path: List of stations in the shortest path (optional)
        source: Source station name (highlighted green)
        destination: Destination station name (highlighted red)
        dark_mode: Whether to use dark background theme

    Returns:
        Plotly Figure object
    """
    # Build position dictionary: station_name -> (longitude, latitude)
    pos = {}
    for _, row in stations_df.iterrows():
        pos[row['station_name']] = (row['longitude'], row['latitude'])

    # Determine which stations are on the path
    path_set = set(path) if path else set()
    path_edges = set()
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i + 1]))
            path_edges.add((path[i + 1], path[i]))

    fig = go.Figure()

    # ── Draw background edges (non-path connections) ──────────────────────────
    # Group edges by line for color coding
    line_edge_traces = {}

    for u, v, data in G.edges(data=True):
        # Skip edges that are on the highlighted path
        if (u, v) in path_edges:
            continue
        if u not in pos or v not in pos:
            continue

        # Determine line for color (use source station's line)
        u_line = stations_df[stations_df['station_name'] == u]['line'].values
        line = u_line[0] if len(u_line) > 0 else "Blue"
        # Normalize compound line names
        primary_line = line.split('/')[0]

        color = LINE_COLORS.get(primary_line, "#636e72")

        if line not in line_edge_traces:
            line_edge_traces[line] = {
                'x': [], 'y': [], 'color': color, 'primary': primary_line
            }

        x0, y0 = pos[u]
        x1, y1 = pos[v]
        line_edge_traces[line]['x'].extend([x0, x1, None])
        line_edge_traces[line]['y'].extend([y0, y1, None])

    # Add each line's edges as a trace
    for line_name, trace_data in line_edge_traces.items():
        fig.add_trace(go.Scatter(
            x=trace_data['x'],
            y=trace_data['y'],
            mode='lines',
            line=dict(
                color=trace_data['color'],
                width=2.5,
            ),
            opacity=0.55,
            name=f"{line_name} Line",
            hoverinfo='none',
            showlegend=True
        ))

    # ── Draw highlighted path edges ───────────────────────────────────────────
    if path and len(path) > 1:
        path_x, path_y = [], []
        for i in range(len(path) - 1):
            if path[i] in pos and path[i + 1] in pos:
                x0, y0 = pos[path[i]]
                x1, y1 = pos[path[i + 1]]
                path_x.extend([x0, x1, None])
                path_y.extend([y0, y1, None])

        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='lines',
            line=dict(color='#00d2ff', width=5),
            name='Shortest Path',
            hoverinfo='none',
            showlegend=True
        ))

    # ── Draw regular stations (non-path nodes) ────────────────────────────────
    regular_x, regular_y, regular_text, regular_hover = [], [], [], []

    for node in G.nodes():
        if node in path_set or node == source or node == destination:
            continue
        if node not in pos:
            continue

        x, y = pos[node]
        regular_x.append(x)
        regular_y.append(y)
        regular_text.append('')
        node_data = G.nodes[node]
        regular_hover.append(
            f"<b>{node}</b><br>"
            f"Line: {node_data.get('line', 'N/A')}<br>"
            f"Lat: {node_data.get('latitude', 0):.4f}<br>"
            f"Lon: {node_data.get('longitude', 0):.4f}"
        )

    fig.add_trace(go.Scatter(
        x=regular_x,
        y=regular_y,
        mode='markers',
        marker=dict(
            size=7,
            color='#bdc3c7',
            line=dict(width=1, color='#7f8c8d')
        ),
        text=regular_hover,
        hovertemplate='%{text}<extra></extra>',
        name='Stations',
        showlegend=False
    ))

    # ── Draw path stations ────────────────────────────────────────────────────
    if path:
        path_station_x, path_station_y, path_hover = [], [], []
        for station in path:
            if station == source or station == destination:
                continue
            if station not in pos:
                continue
            x, y = pos[station]
            path_station_x.append(x)
            path_station_y.append(y)
            node_data = G.nodes.get(station, {})
            path_hover.append(
                f"<b>🚇 {station}</b><br>"
                f"Line: {node_data.get('line', 'N/A')}<br>"
                f"(On shortest path)"
            )

        fig.add_trace(go.Scatter(
            x=path_station_x,
            y=path_station_y,
            mode='markers+text',
            marker=dict(
                size=12,
                color='#00d2ff',
                line=dict(width=2, color='#ffffff'),
                symbol='circle'
            ),
            text=[s.split(' ')[0] for s in path[1:-1] if s != source and s != destination],
            textposition='top center',
            textfont=dict(size=9, color='#00d2ff'),
            hovertext=path_hover,
            hovertemplate='%{hovertext}<extra></extra>',
            name='Path Stations',
            showlegend=False
        ))

    # ── Draw source station (green) ───────────────────────────────────────────
    if source and source in pos:
        x, y = pos[source]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(
                size=18,
                color='#00b894',
                symbol='star',
                line=dict(width=2, color='#ffffff')
            ),
            text=[source],
            textposition='top center',
            textfont=dict(size=11, color='#00b894', family='Arial Black'),
            hovertemplate=f"<b>🟢 SOURCE</b><br>{source}<extra></extra>",
            name=f'Source: {source}',
            showlegend=True
        ))

    # ── Draw destination station (red) ────────────────────────────────────────
    if destination and destination in pos:
        x, y = pos[destination]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(
                size=18,
                color='#d63031',
                symbol='star',
                line=dict(width=2, color='#ffffff')
            ),
            text=[destination],
            textposition='bottom center',
            textfont=dict(size=11, color='#d63031', family='Arial Black'),
            hovertemplate=f"<b>🔴 DESTINATION</b><br>{destination}<extra></extra>",
            name=f'Destination: {destination}',
            showlegend=True
        ))

    # ── Layout and styling ────────────────────────────────────────────────────
    bg_color = '#0f1117' if dark_mode else '#f8f9fa'
    paper_bg = '#0f1117' if dark_mode else '#ffffff'
    font_color = '#ecf0f1' if dark_mode else '#2c3e50'
    grid_color = '#2c3e50' if dark_mode else '#dfe6e9'

    fig.update_layout(
        title=dict(
            text='🚇 Delhi Metro Network',
            font=dict(size=20, color=font_color, family='Arial Black'),
            x=0.5
        ),
        paper_bgcolor=paper_bg,
        plot_bgcolor=bg_color,
        height=600,
        hovermode='closest',
        legend=dict(
            bgcolor='rgba(0,0,0,0.4)' if dark_mode else 'rgba(255,255,255,0.8)',
            bordercolor='#636e72',
            borderwidth=1,
            font=dict(color=font_color, size=11),
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=0.5,
            zeroline=False,
            showticklabels=False,
            title='Longitude'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=0.5,
            zeroline=False,
            showticklabels=False,
            title='Latitude'
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        dragmode='pan'
    )

    return fig


def build_algorithm_trace_figure(steps: List[dict], dark_mode: bool = True) -> go.Figure:
    """
    Build a Plotly figure showing the distance progression during Dijkstra's execution.

    Args:
        steps: List of step dictionaries from dijkstra()
        dark_mode: Whether to use dark theme

    Returns:
        Plotly Figure
    """
    if not steps:
        return go.Figure()

    step_nums = [s['step'] for s in steps]
    visited_counts = [len(s['visited_nodes']) for s in steps]
    current_dists = [s['current_distance'] for s in steps]
    nodes_visited = [s['current_node'] for s in steps]

    bg_color = '#0f1117' if dark_mode else '#f8f9fa'
    paper_bg = '#1a1a2e' if dark_mode else '#ffffff'
    font_color = '#ecf0f1' if dark_mode else '#2c3e50'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=step_nums,
        y=current_dists,
        mode='lines+markers',
        name='Distance at Current Node',
        line=dict(color='#00d2ff', width=2),
        marker=dict(size=8, color='#00d2ff'),
        text=nodes_visited,
        hovertemplate='Step %{x}<br>Station: %{text}<br>Distance: %{y:.2f} km<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=step_nums,
        y=visited_counts,
        name='Nodes Visited',
        marker_color='rgba(0, 184, 148, 0.4)',
        yaxis='y2',
        hovertemplate='Step %{x}<br>Visited: %{y} nodes<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='Dijkstra Algorithm Progress',
            font=dict(size=16, color=font_color),
            x=0.5
        ),
        paper_bgcolor=paper_bg,
        plot_bgcolor=bg_color,
        height=300,
        font=dict(color=font_color),
        legend=dict(
            bgcolor='rgba(0,0,0,0.4)' if dark_mode else 'rgba(255,255,255,0.8)',
            font=dict(color=font_color)
        ),
        xaxis=dict(
            title='Algorithm Step',
            gridcolor='#2c3e50',
            showgrid=True
        ),
        yaxis=dict(
            title='Distance (km)',
            gridcolor='#2c3e50',
            showgrid=True,
            side='left'
        ),
        yaxis2=dict(
            title='Nodes Visited',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig
