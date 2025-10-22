import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="Edge Time Series Viz", layout="wide")

# -------------------------------------------------------------
# CONFIG: set your file paths
# -------------------------------------------------------------
PARQUET_PATH = "data\edge_time_series.parquet"
EDGES_PATH = "data\walk_edges_clean.geojson"

# -------------------------------------------------------------
# LOAD DATA (no checks, assumes correct)
# -------------------------------------------------------------
edge_time_series = pd.read_parquet(PARQUET_PATH)   # index = time, columns = edge OBJECTIDs
edges = gpd.read_file(EDGES_PATH)                  # must have 'OBJECTID' and 'geometry'

# -------------------------------------------------------------
# UI SETUP
# -------------------------------------------------------------
st.title("Edge Time Series Visualization")

n = len(edge_time_series)
if n == 0:
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")

# Fast index slider instead of datetime slider
default_idx = n // 10 if n > 1 else 0
idx = st.sidebar.slider("Select timestep (row index)", 0, n - 1, default_idx, step=1)

time_choice = edge_time_series.index[idx]
st.sidebar.write(f"Selected time: **{time_choice}**")

cmap_name = st.sidebar.selectbox("Colormap", ["Blues","Reds"], index=0)
line_weight = st.sidebar.slider("Line weight", 1, 7, 3)
clip_quantiles = st.sidebar.checkbox("Clip color scale to 1–99% quantiles", value=True)
use_log1p = st.sidebar.checkbox("log1p transform (display only)", value=False)

# -------------------------------------------------------------
# MERGE DATA FOR SELECTED TIMESTEP
# -------------------------------------------------------------
row = edge_time_series.iloc[idx]
row.name = "count"
t = edges.merge(row.rename("count"), left_on="OBJECTID", right_index=True, how="left")
t["count"] = t["count"].fillna(0.0)

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

st_html(m._repr_html_(), height=720)

# -------------------------------------------------------------
# SIDEBAR SUMMARY
# -------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.write(f"Row index: **{idx}**")
st.sidebar.write(f"vmin–vmax: **{vmin:.3g} – {vmax:.3g}**")
st.sidebar.write(f"Nonzero edges: **{int((t['count']>0).sum())}** / {len(t)}")
