import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import geopandas as gpd
import folium
import numpy as np
from scipy.interpolate import splprep, splev
from shapely import wkt
from shapely.geometry import box, LineString
from streamlit_folium import folium_static

# ==================== Custom CSS for Aesthetic Improvements ====================
st.markdown(
    """
    <style>
    /* Overall background for the main container */
    .reportview-container {
        background: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Style for the title */
    h1 {
        color: #2e7bcf;
        font-weight: 600;
    }
    /* Sidebar style changed to pastel purple */
    [data-testid="stSidebar"] {
        background: #C8A2C8;
        color: black;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] .sidebar-content {
        color: black;
    }
    /* File uploader styling */
    .stFileUploader {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================== Streamlit Page Title ====================
st.title("Road KSI Map - Essex")

# -------------------- Step 1: Upload and Load Data --------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

def load_data(file):
    try:
        df = pd.read_csv(file)
        if 'geometry' not in df.columns:
            st.error("Error: The uploaded file must contain a 'geometry' column.")
            return gpd.GeoDataFrame()
        
        # Convert WKT to geometry and filter for LineString types
        df['geometry'] = df['geometry'].apply(wkt.loads)
        df['geometry_type'] = df['geometry'].apply(lambda geom: geom.geom_type)
        df = df[df['geometry_type'] == 'LineString']

        if df.empty:
            st.error("No valid LineString geometries found in the dataset.")
            return gpd.GeoDataFrame()

        df['Road Number'] = df['Road Number'].fillna("Unknown").astype(str).str.strip()
        df['speed_limit'] = df['speed_limit'].fillna(0).astype(int)
        df['KSI Count'] = df['KSI Count'].fillna(0).astype(int)
        return gpd.GeoDataFrame(df, geometry='geometry')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return gpd.GeoDataFrame()

if uploaded_file:
    gdf = load_data(uploaded_file)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# -------------------- Step 2: Detect and Apply Correct CRS --------------------
if not gdf.empty:
    min_x, max_x = gdf.geometry.bounds.minx.min(), gdf.geometry.bounds.maxx.max()
    min_y, max_y = gdf.geometry.bounds.miny.min(), gdf.geometry.bounds.maxy.max()
    
    if min_x > -180 and max_x < 180 and min_y > -90 and max_y < 90:
        gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf.set_crs("EPSG:32631", inplace=True)
        gdf = gdf.to_crs("EPSG:4326")
else:
    st.stop()

# -------------------- Step 3: Define Essex Bounding Box and Filter Out KSI Count 0 --------------------
bbox_coords = {"south": 51.2, "north": 52.3, "west": -0.2, "east": 1.5}
essex_bbox = box(bbox_coords["west"], bbox_coords["south"], bbox_coords["east"], bbox_coords["north"])

# Apply bounding box filter and remove rows with KSI Count of 0
gdf = gdf[gdf.intersects(essex_bbox)]
gdf = gdf[gdf['KSI Count'] != 0]

# -------------------- Step 4: Streamlit UI Sidebar Filters --------------------
st.sidebar.header("Filters")

# Road Number Filter
all_road_numbers = sorted(gdf["Road Number"].unique())
all_roads = st.sidebar.checkbox("Select All Road Numbers", True)
selected_roads = all_road_numbers if all_roads else st.sidebar.multiselect(
    "Select Road Numbers", all_road_numbers, default=all_road_numbers[:5]
)

# Speed Limit Filter
speed_limits = sorted(gdf['speed_limit'].unique())
all_speeds = st.sidebar.checkbox("Select All Speed Limits", True)
selected_speed = speed_limits if all_speeds else st.sidebar.multiselect(
    "Select Speed Limits", speed_limits, default=speed_limits[:5]
)

gdf_filtered = gdf[(gdf['Road Number'].isin(selected_roads)) & (gdf['speed_limit'].isin(selected_speed))]

# -------------------- Step 5: KSI Count Filter & Legend in Sidebar --------------------
st.sidebar.header("KSI Count Filter & Legend")

ksi_legend_html = """
<div style="background: white; padding: 10px; border: 1px solid #cccccc; border-radius: 5px; margin-bottom: 10px;">
  <strong>KSI Count Legend</strong><br>
  <span style="background: yellow; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></span>1–4<br>
  <span style="background: orange; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></span>5–7<br>
  <span style="background: red; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></span>8+<br>
</div>
"""
st.sidebar.markdown(ksi_legend_html, unsafe_allow_html=True)

filter_1_4 = st.sidebar.checkbox("Include KSI 1–4", True)
filter_5_7 = st.sidebar.checkbox("Include KSI 5–7", True)
filter_8   = st.sidebar.checkbox("Include KSI 8+", True)

if not gdf_filtered.empty:
    ksi_conditions = []
    if filter_1_4:
        ksi_conditions.append((gdf_filtered['KSI Count'] >= 1) & (gdf_filtered['KSI Count'] <= 4))
    if filter_5_7:
        ksi_conditions.append((gdf_filtered['KSI Count'] >= 5) & (gdf_filtered['KSI Count'] <= 7))
    if filter_8:
        ksi_conditions.append(gdf_filtered['KSI Count'] >= 8)
    
    if ksi_conditions:
        combined_ksi = ksi_conditions.pop(0)
        for cond in ksi_conditions:
            combined_ksi |= cond
        gdf_filtered = gdf_filtered[combined_ksi]

# -------------------- Step 6: Smooth the Road Geometries --------------------
def smooth_geometry(geom, smoothing_factor=0, num_points=500):
    """
    Smooth a LineString using spline interpolation.
    Returns a new LineString with more interpolated points.
    """
    try:
        if geom.geom_type == "LineString" and len(geom.coords) > 2:
            x, y = geom.xy
            x, y = list(x), list(y)
            tck, u = splprep([x, y], s=smoothing_factor)  # s=0 for exact interpolation
            unew = np.linspace(0, 1.0, num=num_points)
            out = splev(unew, tck)
            return LineString(list(zip(out[0], out[1])))
        else:
            return geom
    except Exception as e:
        return geom

if not gdf_filtered.empty:
    # Optionally, create a copy to avoid SettingWithCopyWarning:
    gdf_filtered = gdf_filtered.copy()
    gdf_filtered['geometry'] = gdf_filtered['geometry'].apply(
        lambda geom: smooth_geometry(geom, smoothing_factor=0, num_points=500)
    )

# -------------------- Step 7: Create Map with Color Coding --------------------
def get_ksi_color(ksi):
    """Return a color based on the KSI Count."""
    if 1 <= ksi <= 4:
        return 'yellow'
    elif 5 <= ksi <= 7:
        return 'orange'
    else:
        return 'red'

if not gdf_filtered.empty:
    center = essex_bbox.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="cartodbpositron")
    folium.GeoJson(
        gdf_filtered.to_json(),
        name='Road Segments',
        style_function=lambda feature: {
            'color': get_ksi_color(feature['properties']['KSI Count']),
            'weight': 6,
            'opacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['Road Number', 'KSI Count', 'speed_limit'],
            aliases=['Road Number:', 'KSI Count:', 'Speed Limit:'],
            localize=True
        ),
    ).add_to(m)
    st.markdown("<style>.block-container { padding: 1rem 1rem; }</style>", unsafe_allow_html=True)
    folium_static(m, width=1600, height=800)
else:
    st.warning("No data matches the selected filters or there is an issue rendering the map.")
