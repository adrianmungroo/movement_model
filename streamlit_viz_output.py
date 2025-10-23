import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from streamlit_folium import st_folium
from shapely.geometry import Point
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Edge Time Series Viz", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# CONFIG: set your file paths
# -------------------------------------------------------------
PARQUET_PATH = r"data/edge_time_series.parquet"
EDGES_PATH = r"data/walk_edges_clean.geojson"

# -------------------------------------------------------------
# LOAD DATA (no checks, assumes correct)
# -------------------------------------------------------------
edge_time_series = pd.read_parquet(PARQUET_PATH)   # index = time, columns = edge OBJECTIDs
edges = gpd.read_file(EDGES_PATH)                  # must have 'OBJECTID' and 'geometry'

# Set CRS to EPSG:4326 (WGS84) if not already set, then project to EPSG:2240 for accurate distance calculations
if edges.crs is None:
    edges.set_crs("EPSG:4326", inplace=True)
edges_projected = edges.to_crs("EPSG:2240")

# -------------------------------------------------------------
# UI SETUP
# -------------------------------------------------------------
st.write("# GT Spatio-Temporal Movement Visualization")
st.caption(f"Made by Adrian Mungroo, Simon Ramdath and Isaac Lo")
st.write("This app allows you to visualize the movement of people within the GT campus. It combines GIS data from campus building geometries & walkways and timeseries WiFi data collected between April 14th and 15th, 2025.")

# Initialize session state for selected edge
if 'selected_objectid' not in st.session_state:
    st.session_state.selected_objectid = None

n = len(edge_time_series)
if n == 0:
    st.stop()

# Get time values for selection
time_values = edge_time_series.index
default_time = time_values[0]

# Sidebar controls
st.sidebar.header("Visualization Settings")

cmap_name = st.sidebar.selectbox("Colormap", ["Blues","Reds"], index=0)
line_weight = st.sidebar.slider("Line weight", 1, 7, 3)
clip_quantiles = st.sidebar.checkbox("Clip color scale to 1–99% quantiles", value=True)
use_log1p = st.sidebar.checkbox("log1p transform (display only)", value=False)

# Main area: Time selection (outside sidebar)
st.write("---")
st.write("### Select Timestep")
time_choice = st.select_slider(
    "Time", 
    options=time_values, 
    value=default_time,
    label_visibility="collapsed"
)

# Get the index for this time
idx = time_values.get_loc(time_choice)

# -------------------------------------------------------------
# MERGE DATA FOR SELECTED TIMESTEP
# -------------------------------------------------------------
row = edge_time_series.iloc[idx]
row.name = "count"
t = edges.merge(row.rename("count"), left_on="OBJECTID", right_index=True, how="left")
t["count"] = t["count"].fillna(0.0)

# Create projected version with same counts
t_projected = edges_projected.merge(row.rename("count"), left_on="OBJECTID", right_index=True, how="left")
t_projected["count"] = t_projected["count"].fillna(0.0)

vals = np.log1p(t["count"].to_numpy()) if use_log1p else t["count"].to_numpy()

# Optional clipping for color scale
if clip_quantiles:
    vmin = float(np.quantile(vals, 0.01))
    vmax = float(np.quantile(vals, 0.99))
else:
    vmin = float(vals.min())
    vmax = float(vals.max())
if vmin == vmax:
    vmin = 0.0

# -------------------------------------------------------------
# BUILD AND DISPLAY MAP
# -------------------------------------------------------------
m = t.explore(
    column="count",
    cmap=cmap_name,
    tiles="CartoDB Positron",
    style_kwds=dict(weight=line_weight),
    tooltip=["OBJECTID", "count"],
    vmin=vmin,
    vmax=vmax
)

# Display map and capture click events
c1, c2 = st.columns(2)
with c1:
    st.write("## Map of Selected Time")
    map_data = st_folium(m, width=None, height=500, returned_objects=["last_object_clicked"])

# Find nearest edge to clicked point and update session state
if map_data["last_object_clicked"]:
    clicked_point = map_data["last_object_clicked"]
    
    # Extract lat/lon from clicked point
    if "lat" in clicked_point and "lng" in clicked_point:
        lat = clicked_point["lat"]
        lon = clicked_point["lng"]
        
        # Check if this is a new click (different from current selection)
        current_objid = st.session_state.selected_objectid
        
        # Create Point geometry in WGS84
        clicked_geom = Point(lon, lat)
        clicked_gdf = gpd.GeoDataFrame([1], geometry=[clicked_geom], crs="EPSG:4326")
        
        # Project to EPSG:2240 for accurate distance calculation
        clicked_gdf_projected = clicked_gdf.to_crs("EPSG:2240")
        clicked_geom_projected = clicked_gdf_projected.geometry.iloc[0]
        
        # Calculate distance from clicked point to all edges (in EPSG:2240 units)
        distances = t_projected.geometry.distance(clicked_geom_projected)
        
        # Find the index of the nearest edge
        nearest_idx = distances.idxmin()
        nearest_edge = t.loc[nearest_idx]  # Use original CRS version for display
        nearest_distance = distances.loc[nearest_idx]
        
        # Update session state with selected edge
        selected_edge = nearest_edge.to_dict()
        new_objid = selected_edge["OBJECTID"]
        
        # Only update and rerun if this is a different edge
        if new_objid != current_objid:
            st.session_state.selected_objectid = new_objid
            st.rerun()

# Display time series in right column
with c2:
    st.write("## Time Series of Selected Edge")
    if st.session_state.selected_objectid is None:
        st.write(" Please click on the map to select an edge ")
    else:
        st.write(f"**Selected Edge: OBJECTID {st.session_state.selected_objectid}**")
        
        # Display time series
        plt.figure(figsize=(10,4))
        edge_time_series[st.session_state.selected_objectid].plot(figsize=(10,4), grid=True)
        # draw a vertical line at the selected time
        plt.axvline(x=time_choice, color='red', linestyle='--')
        st.pyplot(plt.gcf())

# -------------------------------------------------------------
# SIDEBAR SUMMARY
# -------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.write(f"Row index: **{idx}**")
st.sidebar.write(f"vmin–vmax: **{vmin:.3g} – {vmax:.3g}**")
st.sidebar.write(f"Nonzero edges: **{int((t['count']>0).sum())}** / {len(t)}")
