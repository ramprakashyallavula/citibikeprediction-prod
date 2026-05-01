import sys
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st
import time
import streamlit as st

# Add project root to PYTHONPATH
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
from src.inference import fetch_hourly_rides, fetch_predictions

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title='🛠️ Monitoring Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- HEADER ---
st.markdown("# 🛠️ Monitoring Dashboard")
st.markdown("**Analyze actual vs. predicted Citi Bike rides in real-time.**")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
time_window = st.sidebar.slider(
    "Select Time Window (hours)",
    min_value=1,
    max_value=24 * 28,
    value=24,
    step=1
)
# Preload merged data to populate station list
@st.cache_data(ttl=300)
def get_station_list(hours):
    dp = fetch_predictions(hours)
    dr = fetch_hourly_rides(hours)
    merged = pd.merge(dp, dr, on=['pickup_location_id','pickup_hour'], how='inner')
    return merged['pickup_location_id'].unique().tolist()
stations = get_station_list(time_window)
selected_station = st.sidebar.selectbox("Station ID", stations)
st.sidebar.markdown("---")

# --- DATA LOADING & PROCESSING ---
@st.cache_data(ttl=300)
def load_station_data(station_id, hours):
    df_pred = fetch_predictions(hours)
    df_obs = fetch_hourly_rides(hours)
    df = pd.merge(
        df_pred, df_obs,
        on=['pickup_location_id','pickup_hour'],
        how='inner'
    )
    df = df[df['pickup_location_id'] == station_id]
    df['timestamp'] = pd.to_datetime(df['pickup_hour'])
    df.sort_values('timestamp', inplace=True)
    return df

data = load_station_data(selected_station, time_window)
if data.empty:
    st.warning("No data for this station in the selected window.")
    st.stop()

# Compute errors and average MAE
data['error'] = (data['predicted_demand'] - data['rides']).abs()
avg_error = data['error'].mean()

# --- METRICS ---
st.subheader(f"Station {selected_station} — Last {time_window} h")
col1, col2, col3 = st.columns(3)
col1.metric("Average Error", f"{avg_error:.2f}")
col2.metric("Max Error", f"{data['error'].max():.2f}")
col3.metric("Data Points", len(data))
st.markdown("---")

# --- COMPARISON CHART ---
st.subheader("Actual vs. Predicted Rides")
comparison = alt.Chart(data).transform_fold(
    ['rides','predicted_demand'],
    as_=['Type','Value']
).mark_line(point=True).encode(
    x=alt.X('timestamp:T', title='Time'),
    y=alt.Y('Value:Q', title='Ride Count'),
    color='Type:N',
    tooltip=['timestamp:T','Type:N','Value:Q']
).properties(width=800, height=400).interactive()
st.altair_chart(comparison, use_container_width=True)

st.markdown("---")

# --- ERROR OVER TIME WITH MAE HIGHLIGHT ---
st.subheader("Error Over Time with MAE Highlight")
# Base area chart
base = alt.Chart(data).mark_area(opacity=0.3, color='#F77F00').encode(
    x=alt.X('timestamp:T', title='Time'),
    y=alt.Y('error:Q', title='Absolute Error'),
    tooltip=['timestamp:T','error:Q']
)
# MAE horizontal rule
mae_rule = alt.Chart(pd.DataFrame({'MAE':[avg_error]})).mark_rule(color='red', strokeWidth=3).encode(
    y='MAE:Q'
)
# Text label for MAE
mae_text = alt.Chart(pd.DataFrame({'MAE':[avg_error]})).mark_text(
    align='left', dx=5, dy=-5, color='red', fontWeight='bold'
).encode(
    y='MAE:Q',
    x=alt.value(5),
    text=alt.Text('MAE:Q', format=".2f")
)
# Combine charts
err_chart = alt.layer(base, mae_rule, mae_text).properties(width=800, height=250)
st.altair_chart(err_chart, use_container_width=True)
