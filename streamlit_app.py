import json
import time
import queue
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import joblib
import altair as alt
import paho.mqtt.client as mqtt

# --- Configuration ---
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_DATA = "net4think/air_quality/data"
TOPIC_PRED = "net4think/air_quality/prediction"
MODEL_FILE = "air_quality_rf_model.joblib"

st.set_page_config(
    page_title="Air Quality AI Monitor",
    page_icon="🍃",
    layout="wide"
)

# --- State Management ---
if 'mqtt_queue' not in st.session_state:
    st.session_state.mqtt_queue = queue.Queue()

if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {
        "temp": 0, "hum": 0, "gas": 0, "timestamp": "-"
    }

if 'pred_result' not in st.session_state:
    st.session_state.pred_result = {
        "label": "Menunggu...", "confidence": 0
    }

if 'history' not in st.session_state:
    st.session_state.history = []

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load the machine learning model, label encoder, and feature names."""
    try:
        artifact = joblib.load(MODEL_FILE)
        return artifact["model"], artifact["label_encoder"], artifact["features"]
    except Exception as e:
        print(f"Model Load Error: {e}")
        return None, None, None

model, label_encoder, features = load_model()

if model is None:
    st.error("⚠️ Gagal memuat model Machine Learning. Pastikan file model kompatibel.")

# --- MQTT Setup ---
import uuid

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback for when the client receives a CONNACK response from the server."""
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(TOPIC_DATA)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Callback for MQTT message reception."""
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic
        if userdata is not None:
            userdata.put((topic, payload))
    except Exception as e:
        print(f"MQTT Thread Error: {e}")

@st.cache_resource
def start_mqtt():
    """Initialize and start the MQTT client in a background thread."""
    # Use a unique client ID to avoid conflicts
    unique_id = f"Streamlit_AI_{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=unique_id, clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"MQTT Connection Error: {e}")
        return None

mqtt_client = start_mqtt()

if mqtt_client:
    mqtt_client.user_data_set(st.session_state.mqtt_queue)


# --- Data Processing (Main Thread) ---
while not st.session_state.mqtt_queue.empty():
    topic, payload = st.session_state.mqtt_queue.get()
    
    if topic == TOPIC_DATA:
        # Parse Data
        temp = float(payload.get("temperature", 0))
        hum = float(payload.get("humidity", 0))
        gas = float(payload.get("gas_ppm", 0))
        
        # Get WIB Time (UTC+7)
        wib_time = datetime.now(timezone.utc) + timedelta(hours=7)
        timestamp = payload.get("timestamp", wib_time.strftime("%H:%M:%S"))

        # Update Current State
        st.session_state.sensor_data = {
            "temp": temp, "hum": hum, "gas": gas, "timestamp": timestamp
        }
        
        # Inference
        pred_label = "N/A"
        confidence = 0
        if model is not None:
            input_df = pd.DataFrame([[temp, hum, gas]], columns=features)
            pred_idx = model.predict(input_df)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            proba = model.predict_proba(input_df)[0]
            confidence = round(proba[pred_idx] * 100, 1)

            st.session_state.pred_result = {
                "label": pred_label, "confidence": confidence
            }

            # Publish Result
            if mqtt_client is not None:
                resp = {
                    "label": pred_label,
                    "confidence": confidence,
                    "device_id": "Streamlit-Cloud"
                }
                mqtt_client.publish(TOPIC_PRED, json.dumps(resp))
        
        # Update History
        new_record = {
            "time": timestamp, 
            "Temp": temp, 
            "Hum": hum, 
            "Gas": gas,
            "Status": pred_label 
        }
        st.session_state.history.append(new_record)
        
        # Maintain history buffer size
        if len(st.session_state.history) > 1000: 
            st.session_state.history.pop(0)

# --- UI Application ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
        border: 1px solid #e2e8f0;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-card {
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .chart-wrapper {
        background: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    div[data-testid="stExpander"] {
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stExpander"] > div[role="button"] {
        color: #1e3a8a;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🍃 Air Quality AI Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Real-time Environmental Sensing</div>', unsafe_allow_html=True)

# --- Hero Section (Status) ---
lbl = st.session_state.pred_result['label']
conf = st.session_state.pred_result['confidence']
timestamp = st.session_state.sensor_data['timestamp']

STATUS_CONFIG = {
    "Baik": {"color": "#10b981", "icon": "😊", "msg": "Kualitas udara sangat baik. Nikmati harimu!"},
    "Sedang": {"color": "#f59e0b", "icon": "😐", "msg": "Kualitas udara cukup. Sensitif? Hati-hati."},
    "Tidak_Sehat": {"color": "#f97316", "icon": "😷", "msg": "Udara kotor. Kurangi aktivitas luar ruangan."},
    "Berbahaya": {"color": "#ef4444", "icon": "☠️", "msg": "BAHAYA! Gunakan masker N95 atau tetap di dalam."}
}

current_state = STATUS_CONFIG.get(lbl, {"color": "#64748b", "icon": "❓", "msg": "Menunggu Data..."})
bg_color = current_state["color"]

st.markdown(f"""
    <div class="status-card" style="background: linear-gradient(135deg, {bg_color}, {bg_color}cc);">
        <div style="font-size: 5rem; margin-bottom: 0px; text-shadow: 0 4px 10px rgba(0,0,0,0.2);">{current_state['icon']}</div>
        <h3 style="margin:0; font-weight:400; opacity:0.9; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;">Status Udara</h3>
        <h1 style="margin: 5px 0 15px 0; font-size: 3.5rem; font-weight:800; letter-spacing: 1px;">{lbl.replace("_", " ")}</h1>
        <p style="font-size: 1.1rem; opacity: 0.95; font-style: italic; background: rgba(0,0,0,0.1); display: inline-block; padding: 5px 15px; border-radius: 15px; margin-bottom: 25px;">"{current_state['msg']}"</p>
        <div style="background: rgba(255, 255, 255, 0.25); border-radius: 16px; padding: 20px; text-align: left; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid rgba(255,255,255,0.2);">
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;">
                <span style="font-size: 0.95rem; font-weight: 500; opacity: 0.9; letter-spacing: 0.5px;">🤖 AI CONFIDENCE</span>
                <span style="font-size: 1.8rem; font-weight: 700; line-height: 1;">{conf}<span style="font-size: 1rem;">%</span></span>
            </div>
            <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.3); border-radius: 5px; overflow: hidden; margin-bottom: 12px;">
                <div style="width: {conf}%; height: 100%; background: #ffffff; border-radius: 5px; box-shadow: 0 0 10px rgba(255,255,255,0.5); transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.85;">
                <span>Updating live...</span>
                <span>⏱️ {timestamp}</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Metrics Section ---
c1, c2, c3 = st.columns(3)

def metric_card(label, value, unit, icon):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value} <span style="font-size:1rem; color:#94a3b8;">{unit}</span></div>
    </div>
    """

with c1:
    st.markdown(metric_card("Temperature", st.session_state.sensor_data['temp'], "°C", "🌡️"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Humidity", st.session_state.sensor_data['hum'], "%", "💧"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Gas Level", st.session_state.sensor_data['gas'], "ppm", "💨"), unsafe_allow_html=True)

# --- Analytics Section ---
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    df[['Gas', 'Temp', 'Hum']] = df[['Gas', 'Temp', 'Hum']].apply(pd.to_numeric)
    
    # Generate sequential step for charts
    df = df.reset_index(names='step')

    with st.expander("📄 View Raw Data", expanded=True):
        st.dataframe(
            df[['time', 'Temp', 'Hum', 'Gas', 'Status']], 
            use_container_width=True,
            column_config={
                "time": "Timestamp",
                "Temp": st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
                "Hum": st.column_config.NumberColumn("Humidity (%)", format="%.1f"),
                "Gas": st.column_config.NumberColumn("Gas (ppm)", format="%.1f"),
                "Status": "AI Status"
            },
            hide_index=True
        )
        
        st.download_button(
            label="📥 Download Full Log (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"air_quality_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )

    st.subheader("📊 Analytics Overview")
    with st.container():
        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
        
        tab_gas, tab_env = st.tabs(["💨 Gas Quality Trend", "🌡️ Environment Data"])
        
        with tab_gas:
            chart_gas = alt.Chart(df).mark_area(
                line={'color':'#f59e0b'},
                interpolate='monotone',
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='rgba(245, 158, 11, 0.1)', offset=0),
                           alt.GradientStop(color='rgba(245, 158, 11, 0.6)', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x=alt.X('step', axis=None), # Hide X axis labels for cleaner look or keep them
                y=alt.Y('Gas', title='Gas Concentration (ppm)', scale=alt.Scale(zero=False, padding=1)),
                tooltip=[
                    alt.Tooltip('time', title='Time'),
                    alt.Tooltip('Gas', title='Gas ppm', format='.1f'),
                    alt.Tooltip('Status', title='Status')
                ]
            ).properties(height=350)
            
            # Add points for better hover interaction
            points = chart_gas.mark_circle(size=60, color='#f59e0b').encode(
                opacity=alt.condition(alt.value(0), alt.value(1), alt.value(0)) # Hidden unless hovered? No, let's keep them hidden or small
            )

            st.altair_chart(chart_gas.interactive(), use_container_width=True)
            
        with tab_env:
            # Base chart
            base = alt.Chart(df).encode(
                x=alt.X('step', axis=alt.Axis(title='History Steps'), title=None),
                tooltip=[alt.Tooltip('time', title='Time')]
            )

            # Temperature Line (Left Axis, Red)
            line_temp = base.mark_line(color='#ef4444', interpolate='monotone', strokeWidth=3).encode(
                y=alt.Y('Temp', title='Temperature (°C)', axis=alt.Axis(titleColor='#ef4444'), scale=alt.Scale(zero=False, padding=1)),
                tooltip=['time', alt.Tooltip('Temp', title='Temperature', format='.1f')]
            )

            # Humidity Line (Right Axis, Blue)
            line_hum = base.mark_line(color='#3b82f6', interpolate='monotone', strokeWidth=3).encode(
                y=alt.Y('Hum', title='Humidity (%)', axis=alt.Axis(titleColor='#3b82f6'), scale=alt.Scale(zero=False, padding=1)),
                tooltip=['time', alt.Tooltip('Hum', title='Humidity', format='.1f')]
            )

            # Combine with independent y scales
            chart_env = alt.layer(line_temp, line_hum).resolve_scale(
                y='independent'
            ).properties(height=350).interactive()
            
            st.altair_chart(chart_env, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("⏳ Waiting for data stream...")

# Auto-refresh loop
time.sleep(2)
st.rerun()