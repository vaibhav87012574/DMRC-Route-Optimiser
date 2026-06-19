"""
Delhi Metro Shortest Path Visualizer
======================================
Main Streamlit application entry point.

A professional, interactive tool for exploring the Delhi Metro network,
finding shortest paths using Dijkstra's algorithm, and understanding
how graph algorithms work in real-world transit systems.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import os
import sys

# Add project root to path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.dijkstra import (
    dijkstra, bfs_path, calculate_fare, calculate_travel_time, count_interchanges
)
from utils.graph_builder import (
    load_data, build_graph, build_adjacency_dict,
    get_interchange_stations, get_graph_statistics, LINE_COLORS
)
from visualizations.network_graph import build_metro_figure, build_algorithm_trace_figure

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi Metro Path Visualizer",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Custom CSS ────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), 'styles', 'main.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Global Dark Theme Override ─────────────────────────────────────────────────
st.markdown("""
<style>
/* Core dark theme */
.stApp { background-color: #0f1117; color: #e8eaf0; }
.stApp > header { background-color: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #2d3561;
}
[data-testid="stSidebar"] .stMarkdown { color: #8892b0; }
.stSelectbox label, .stRadio label, .stSlider label { color: #8892b0 !important; }
.stSelectbox > div > div {
    background: #1a1f35 !important;
    border: 1px solid #2d3561 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}
.stButton > button {
    background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
    color: #0f1117 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.05em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0, 210, 255, 0.35) !important;
}
.stExpander {
    background: #1a1f35 !important;
    border: 1px solid #2d3561 !important;
    border-radius: 12px !important;
}
.stExpander > div > div > div { color: #e8eaf0 !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
hr { border-color: #2d3561; }
.stAlert { border-radius: 10px; }
[data-testid="stMetricLabel"] { color: #8892b0 !important; }
[data-testid="stMetricValue"] { color: #00d2ff !important; }
h1, h2, h3 { color: #e8eaf0 !important; }
p { color: #8892b0; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading (cached for performance) ──────────────────────────────────────
@st.cache_data
def load_metro_data():
    """Load and cache metro network data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stations_path = os.path.join(base_dir, 'data', 'stations.csv')
    connections_path = os.path.join(base_dir, 'data', 'connections.csv')
    return load_data(stations_path, connections_path)

@st.cache_resource
def build_metro_graph(stations_df, connections_df):
    """Build and cache the NetworkX graph."""
    return build_graph(stations_df, connections_df)

@st.cache_data
def build_adj_dict(connections_df_hash, weight='distance'):
    """Build and cache the adjacency dictionary."""
    return build_adjacency_dict(connections_df, weight)

# Load data
stations_df, connections_df = load_metro_data()
G = build_metro_graph(stations_df, connections_df)
adj_dict = build_adjacency_dict(connections_df, 'distance')
adj_dict_time = build_adjacency_dict(connections_df, 'travel_time')

# Get sorted station list
all_stations = sorted(stations_df['station_name'].tolist())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <span class="sidebar-logo">🚇</span>
        <div class="sidebar-title">DMRC PathFinder</div>
        <div class="sidebar-subtitle">Delhi Metro Visualizer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗺️ Route Finder")

    # Source and destination dropdowns
    source_station = st.selectbox(
        "🟢 Source Station",
        all_stations,
        index=all_stations.index("Rajiv Chowk") if "Rajiv Chowk" in all_stations else 0,
        key="source"
    )

    destination_station = st.selectbox(
        "🔴 Destination Station",
        all_stations,
        index=all_stations.index("Botanical Garden") if "Botanical Garden" in all_stations else 1,
        key="destination"
    )

    # Swap button
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("⇄", help="Swap source and destination"):
            # Swap by rerunning with swapped values
            st.session_state['source'], st.session_state['destination'] = (
                st.session_state['destination'], st.session_state['source']
            )
            st.rerun()

    st.markdown("---")

    # Find route button
    find_route = st.button("🔍 Find Shortest Route", use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚙️ Options")

    show_algo = st.checkbox("Show Dijkstra Steps", value=True)
    show_comparison = st.checkbox("Show Route Comparison", value=True)
    show_ticket = st.checkbox("Show Metro Ticket", value=True)
    max_steps = st.slider("Max Steps to Display", 5, 30, 15,
                          help="Number of Dijkstra algorithm steps to show")

    st.markdown("---")

    # Network statistics in sidebar
    stats = get_graph_statistics(G)
    st.markdown("### 📊 Network Stats")
    st.metric("Total Stations", stats['total_stations'])
    st.metric("Total Connections", stats['total_connections'])
    st.metric("Network Distance", f"{stats['total_network_distance']} km")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#636e72; text-align:center; line-height:1.8'>
        Built with ❤️ using<br>
        Streamlit · NetworkX · Plotly<br>
        <span style='color:#00d2ff'>Dijkstra's Algorithm</span>
    </div>
    """, unsafe_allow_html=True)

# ── Main Content Area ──────────────────────────────────────────────────────────
st.markdown("""
<h1 style='
    text-align: center;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00b894);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
'>🚇 Delhi Metro Shortest Path Visualizer</h1>
<p style='text-align:center; color:#8892b0; font-size:1rem; margin-bottom:28px'>
    Dijkstra's Algorithm · Interactive Graph · Real DMRC Network
</p>
""", unsafe_allow_html=True)

# ── Initialize session state ───────────────────────────────────────────────────
if 'route_computed' not in st.session_state:
    st.session_state['route_computed'] = False
if 'path' not in st.session_state:
    st.session_state['path'] = []
if 'total_dist' not in st.session_state:
    st.session_state['total_dist'] = 0
if 'steps' not in st.session_state:
    st.session_state['steps'] = []
if 'bfs_path_result' not in st.session_state:
    st.session_state['bfs_path_result'] = []
if 'bfs_dist' not in st.session_state:
    st.session_state['bfs_dist'] = 0

# ── Compute Route When Button Clicked ─────────────────────────────────────────
if find_route:
    if source_station == destination_station:
        st.warning("⚠️ Source and destination are the same station!")
    else:
        with st.spinner("Computing shortest path using Dijkstra's algorithm..."):
            # Run Dijkstra
            path, total_dist, steps = dijkstra(adj_dict, source_station, destination_station)

            if not path:
                st.error(f"❌ No path found between **{source_station}** and **{destination_station}**")
            else:
                # Run BFS for comparison
                bfs_result, bfs_dist = bfs_path(adj_dict, source_station, destination_station)

                # Store results in session state
                st.session_state['route_computed'] = True
                st.session_state['path'] = path
                st.session_state['total_dist'] = total_dist
                st.session_state['steps'] = steps
                st.session_state['bfs_path_result'] = bfs_result
                st.session_state['bfs_dist'] = bfs_dist
                st.session_state['source_station'] = source_station
                st.session_state['destination_station'] = destination_station

# ── Display Results ────────────────────────────────────────────────────────────
if st.session_state.get('route_computed', False):
    path = st.session_state['path']
    total_dist = st.session_state['total_dist']
    steps = st.session_state['steps']
    bfs_result = st.session_state['bfs_path_result']
    bfs_dist = st.session_state['bfs_dist']
    src = st.session_state.get('source_station', source_station)
    dst = st.session_state.get('destination_station', destination_station)

    # Compute additional metrics
    travel_time = calculate_travel_time(path, connections_df)
    fare = calculate_fare(total_dist)
    num_interchanges = count_interchanges(path)
    interchange_stations_list = get_interchange_stations(path, stations_df)

    # ── Success Banner ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='
        background: linear-gradient(90deg, rgba(0,184,148,0.15), rgba(0,210,255,0.1));
        border: 1px solid rgba(0,184,148,0.3);
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    '>
        <span style='font-size:1.5rem'>✅</span>
        <div>
            <div style='font-weight:700; color:#00b894; font-size:1rem'>Route Found!</div>
            <div style='color:#8892b0; font-size:0.85rem'>
                <b style='color:#e8eaf0'>{src}</b> → <b style='color:#e8eaf0'>{dst}</b>
                &nbsp;·&nbsp; {len(path)} stations &nbsp;·&nbsp;
                {round(total_dist, 2)} km &nbsp;·&nbsp; ~{travel_time} mins
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Route Metric Cards ──────────────────────────────────────────────────────
    st.markdown("### 📊 Route Analytics")
    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        (c1, "🚉", str(len(path)), "Stations", "#00d2ff"),
        (c2, "⏱️", f"{travel_time}", "Minutes", "#00b894"),
        (c3, "📏", f"{round(total_dist, 2)}", "Kilometres", "#fdcb6e"),
        (c4, "💰", f"₹{fare}", "Fare", "#a29bfe"),
        (c5, "🔄", str(num_interchanges), "Interchanges", "#fd79a8"),
    ]

    for col, icon, value, label, color in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent-color: {color}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Optimal Route Path Display ──────────────────────────────────────────────
    st.markdown("### 🗺️ Optimal Route")
    path_html = ""
    for i, station in enumerate(path):
        css_class = "path-station"
        if station == src:
            css_class += " source"
        elif station == dst:
            css_class += " destination"
        elif station in interchange_stations_list:
            css_class += " interchange"
        path_html += f'<span class="{css_class}">{"🔄 " if station in interchange_stations_list else ""}{station}</span>'
        if i < len(path) - 1:
            path_html += '<span style="color:#2d3561; margin: 0 2px;">→</span>'

    st.markdown(f"""
    <div class="path-display">
        <div style='font-size:0.7rem; color:#8892b0; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.1em'>
            Route (🔄 = Interchange)
        </div>
        {path_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive Network Visualization ───────────────────────────────────────
    st.markdown("### 🌐 Interactive Metro Network")
    fig = build_metro_figure(G, stations_df, path=path, source=src, destination=dst)
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'scrollZoom': True,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'displaylogo': False
    })

    # ── Two-column: Ticket + Comparison ────────────────────────────────────────
    ticket_col, compare_col = st.columns([1, 1])

    # Metro Ticket Card
    if show_ticket:
        with ticket_col:
            st.markdown("### 🎫 Metro Ticket")
            bfs_stations = len(bfs_result)
            bfs_time = calculate_travel_time(bfs_result, connections_df) if bfs_result else 0
            st.markdown(f"""
            <div class="ticket-card">
                <div class="ticket-header">
                    <div class="ticket-header-text">🚇 DELHI METRO</div>
                    <div class="ticket-logo">🎫</div>
                </div>
                <div class="ticket-body">
                    <div class="ticket-route">
                        <div class="ticket-station">
                            <div class="ticket-station-label">FROM</div>
                            <div class="ticket-station-name">🟢 {src}</div>
                        </div>
                        <div class="ticket-arrow">→</div>
                        <div class="ticket-station">
                            <div class="ticket-station-label">TO</div>
                            <div class="ticket-station-name">🔴 {dst}</div>
                        </div>
                    </div>
                    <hr class="ticket-divider">
                    <div class="ticket-stats">
                        <div class="ticket-stat">
                            <div class="ticket-stat-value">{len(path)}</div>
                            <div class="ticket-stat-label">Stations</div>
                        </div>
                        <div class="ticket-stat">
                            <div class="ticket-stat-value">{travel_time} min</div>
                            <div class="ticket-stat-label">Travel Time</div>
                        </div>
                        <div class="ticket-stat">
                            <div class="ticket-stat-value">{round(total_dist, 1)} km</div>
                            <div class="ticket-stat-label">Distance</div>
                        </div>
                        <div class="ticket-stat">
                            <div class="ticket-stat-value">₹{fare}</div>
                            <div class="ticket-stat-label">Fare</div>
                        </div>
                    </div>
                    <hr class="ticket-divider">
                    <div style="text-align:center; margin-top:8px">
                        <span style="font-size:0.75rem; color:#8892b0">Interchanges: </span>
                        <span style="font-size:0.85rem; color:#fd79a8; font-weight:600">{num_interchanges}</span>
                        &nbsp;·&nbsp;
                        <span style="font-size:0.75rem; color:#8892b0">Algorithm: </span>
                        <span style="font-size:0.75rem; color:#00d2ff; font-weight:600">Dijkstra</span>
                    </div>
                </div>
                <div class="ticket-footer">
                    Valid for single journey · DMRC Network · Non-Transferable
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Route Comparison
    if show_comparison and bfs_result:
        with compare_col:
            st.markdown("### ⚖️ Route Comparison")
            dijk_better_dist = total_dist <= bfs_dist
            dijk_better_stations = len(path) <= len(bfs_result)

            st.markdown(f"""
            <table style='width:100%; border-collapse:separate; border-spacing:0 8px;'>
                <thead>
                    <tr>
                        <th style='color:#8892b0; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; padding:8px 12px; text-align:left'>Metric</th>
                        <th style='color:#00d2ff; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; padding:8px 12px; text-align:center'>Dijkstra</th>
                        <th style='color:#fdcb6e; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; padding:8px 12px; text-align:center'>BFS</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style='background:#1a1f35; border-radius:8px'>
                        <td style='padding:10px 12px; color:#8892b0; border-radius:8px 0 0 8px'>Distance</td>
                        <td style='padding:10px 12px; text-align:center; font-weight:700; color:{"#00b894" if dijk_better_dist else "#e8eaf0"}'>
                            {round(total_dist, 2)} km {"🏆" if dijk_better_dist else ""}
                        </td>
                        <td style='padding:10px 12px; text-align:center; font-weight:700; color:{"#00b894" if not dijk_better_dist else "#e8eaf0"}; border-radius:0 8px 8px 0'>
                            {round(bfs_dist, 2)} km {"🏆" if not dijk_better_dist else ""}
                        </td>
                    </tr>
                    <tr style='background:#1a1f35'>
                        <td style='padding:10px 12px; color:#8892b0; border-radius:8px 0 0 8px'>Stations</td>
                        <td style='padding:10px 12px; text-align:center; font-weight:700; color:{"#00b894" if dijk_better_stations else "#e8eaf0"}'>
                            {len(path)} {"🏆" if dijk_better_stations else ""}
                        </td>
                        <td style='padding:10px 12px; text-align:center; font-weight:700; color:{"#00b894" if not dijk_better_stations else "#e8eaf0"}; border-radius:0 8px 8px 0'>
                            {len(bfs_result)} {"🏆" if not dijk_better_stations else ""}
                        </td>
                    </tr>
                    <tr style='background:#1a1f35'>
                        <td style='padding:10px 12px; color:#8892b0; border-radius:8px 0 0 8px'>Travel Time</td>
                        <td style='padding:10px 12px; text-align:center; font-weight:700; color:#e8eaf0'>
                            {travel_time} min
                        </td>
                        <td style='padding:10px 12px; text-align:center; font-weight:700; color:#e8eaf0; border-radius:0 8px 8px 0'>
                            {calculate_travel_time(bfs_result, connections_df)} min
                        </td>
                    </tr>
                    <tr style='background:#1a1f35'>
                        <td style='padding:10px 12px; color:#8892b0; border-radius:8px 0 0 8px'>Fare</td>
                        <td style='padding:10px 12px; text-align:center; font-weight:700; color:#a29bfe'>₹{fare}</td>
                        <td style='padding:10px 12px; text-align:center; font-weight:700; color:#a29bfe; border-radius:0 8px 8px 0'>
                            ₹{calculate_fare(bfs_dist)}
                        </td>
                    </tr>
                </tbody>
            </table>
            <div style='background:#1a1f35; border-radius:10px; padding:12px 16px; margin-top:12px; border-left:3px solid #00d2ff; font-size:0.8rem; color:#8892b0'>
                💡 <b style='color:#e8eaf0'>Dijkstra</b> minimizes total distance (km),
                while <b style='color:#fdcb6e'>BFS</b> minimizes the number of stops (stations).
                They may find different routes!
            </div>
            """, unsafe_allow_html=True)

    # ── Dijkstra Algorithm Steps ────────────────────────────────────────────────
    if show_algo and steps:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 How Dijkstra Computed This Route")
        st.markdown("""
        <div style='background:#1a1f35; border-radius:12px; padding:14px 18px; margin-bottom:16px; border-left:4px solid #00d2ff; font-size:0.88rem; color:#8892b0'>
            Each step below shows: which node was processed, its distance from the source,
            and which neighboring stations had their distances updated (relaxation step).
        </div>
        """, unsafe_allow_html=True)

        # Algorithm progress chart
        trace_fig = build_algorithm_trace_figure(steps)
        st.plotly_chart(trace_fig, use_container_width=True, config={'displaylogo': False})

        # Step-by-step expandable cards
        display_steps = steps[:max_steps]
        for step in display_steps:
            is_path_node = step['current_node'] in set(path)
            border_color = "#00b894" if is_path_node else "#2d3561"
            icon = "✅" if is_path_node else "⬜"

            with st.expander(
                f"{icon} Step {step['step']}: Processing **{step['current_node']}** "
                f"(dist = {round(step['current_distance'], 2)} km)",
                expanded=(step['step'] <= 3)
            ):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"""
                    **Current Node:** `{step['current_node']}`
                    **Distance from Source:** `{round(step['current_distance'], 2)} km`
                    **Nodes Visited So Far:** `{len(step['visited_nodes'])}`
                    """)

                    if step.get('updates'):
                        st.markdown("**Distance Updates (Relaxation):**")
                        for update in step['updates']:
                            arrow = "∞" if update['old_distance'] == "∞" else f"{update['old_distance']}"
                            st.markdown(f"""
                            <div class="step-update">
                                🔄 <code>{update['node']}</code>:
                                {arrow} → <b style='color:#00b894'>{update['new_distance']} km</b>
                                <span style='color:#8892b0; font-size:0.8rem'>(via {update['via']})</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("*No updates (neighbors already visited or distances unchanged)*")

                with col_b:
                    if step.get('priority_queue'):
                        st.markdown("**Priority Queue (Top 5):**")
                        pq_df = pd.DataFrame(
                            step['priority_queue'],
                            columns=['Distance (km)', 'Station']
                        )
                        st.dataframe(
                            pq_df,
                            use_container_width=True,
                            hide_index=True
                        )

        if len(steps) > max_steps:
            st.info(f"⚡ Showing first {max_steps} of {len(steps)} steps. Increase the slider in the sidebar to see more.")

    # ── Distance Table ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Segment-by-Segment Route Details"):
        segment_data = []
        time_lookup = {}
        dist_lookup = {}
        for _, row in connections_df.iterrows():
            time_lookup[(row['source'], row['destination'])] = row['travel_time']
            dist_lookup[(row['source'], row['destination'])] = row['distance']

        cumulative_dist = 0
        cumulative_time = 0
        for i in range(len(path) - 1):
            seg_dist = dist_lookup.get((path[i], path[i + 1]),
                       dist_lookup.get((path[i + 1], path[i]), 0))
            seg_time = time_lookup.get((path[i], path[i + 1]),
                       time_lookup.get((path[i + 1], path[i]), 2))
            cumulative_dist += seg_dist
            cumulative_time += seg_time

            line = stations_df[stations_df['station_name'] == path[i + 1]]['line'].values
            line_name = line[0] if len(line) > 0 else "N/A"

            segment_data.append({
                'From': path[i],
                'To': path[i + 1],
                'Line': line_name,
                'Segment Dist (km)': round(seg_dist, 2),
                'Seg Time (min)': seg_time,
                'Cumulative Dist (km)': round(cumulative_dist, 2),
                'Cumulative Time (min)': cumulative_time
            })

        seg_df = pd.DataFrame(segment_data)
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

else:
    # ── Landing State (no route computed yet) ──────────────────────────────────
    # Show the metro network map without a path
    st.markdown("### 🌐 Delhi Metro Network")
    fig = build_metro_figure(G, stations_df)
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'scrollZoom': True,
        'displaylogo': False
    })

    # Quick start guide
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1a1f35 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 24px 32px;
        margin: 20px 0;
    '>
        <h3 style='color:#00d2ff; margin-bottom:16px'>🚀 Get Started</h3>
        <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:16px'>
            <div style='text-align:center; padding:16px'>
                <div style='font-size:2rem'>📍</div>
                <div style='color:#e8eaf0; font-weight:600; margin:8px 0'>Select Stations</div>
                <div style='color:#8892b0; font-size:0.85rem'>Choose your source and destination from the sidebar dropdowns</div>
            </div>
            <div style='text-align:center; padding:16px'>
                <div style='font-size:2rem'>🔍</div>
                <div style='color:#e8eaf0; font-weight:600; margin:8px 0'>Find Route</div>
                <div style='color:#8892b0; font-size:0.85rem'>Click "Find Shortest Route" to run Dijkstra's algorithm</div>
            </div>
            <div style='text-align:center; padding:16px'>
                <div style='font-size:2rem'>📊</div>
                <div style='color:#e8eaf0; font-weight:600; margin:8px 0'>Explore Results</div>
                <div style='color:#8892b0; font-size:0.85rem'>View the path on the map, algorithm steps, and ticket summary</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Algorithm Explanation Section ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📚 Understanding Dijkstra's Algorithm")

with st.expander("🧩 What is Dijkstra's Algorithm? (Click to expand)", expanded=False):
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        **Dijkstra's Algorithm** (pronounced "DIKE-stra") is a graph traversal algorithm
        invented by Dutch computer scientist Edsger W. Dijkstra in 1956.

        It finds the **shortest path** from a single source node to all other nodes in a
        weighted graph — making it perfect for metro route planning!

        #### How it works:
        """)

        steps_explanation = [
            ("Initialize", "Set distance to source = 0, all others = ∞. Add source to priority queue."),
            ("Extract Min", "Pop the node with the smallest distance from the priority queue."),
            ("Relax Edges", "For each neighbor, check if going through the current node gives a shorter path."),
            ("Update", "If shorter path found, update the distance and add to priority queue."),
            ("Repeat", "Repeat steps 2–4 until the destination is reached or queue is empty."),
            ("Reconstruct", "Trace back through predecessors to build the actual path."),
        ]

        for i, (title, desc) in enumerate(steps_explanation, 1):
            st.markdown(f"""
            <div class="algo-step">
                <div class="algo-step-num">{i}</div>
                <div class="algo-step-text">
                    <b style='color:#e8eaf0'>{title}:</b> {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        #### ⚡ Complexity

        | | Value |
        |---|---|
        | **Time** | O((V + E) log V) |
        | **Space** | O(V) |
        | **V** | Vertices (stations) |
        | **E** | Edges (connections) |

        #### ✅ Why Dijkstra for Metro?
        - **Optimal**: Guarantees shortest path
        - **Efficient**: Works with weighted edges
        - **Scalable**: Handles large networks
        - **Real-world**: Used in Google Maps, GPS

        #### ⚠️ Limitations
        - Doesn't work with negative weights
        - BFS is better for unweighted graphs
        - A* is faster with a good heuristic
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='
    text-align: center;
    padding: 24px;
    margin-top: 40px;
    border-top: 1px solid #2d3561;
    color: #636e72;
    font-size: 0.8rem;
'>
    🚇 Delhi Metro Shortest Path Visualizer &nbsp;·&nbsp;
    Built with Streamlit, NetworkX & Plotly &nbsp;·&nbsp;
    <span style='color:#00d2ff'>Dijkstra's Algorithm</span> for optimal routing
</div>
""", unsafe_allow_html=True)
