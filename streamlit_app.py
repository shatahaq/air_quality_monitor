import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import joblib
import time
from datetime import datetime
import queue # Library tambahan untuk antrian thread-safe

# ================= KONFIGURASI =================
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_DATA = "net4think/air_quality/data"
TOPIC_PRED = "net4think/air_quality/prediction"
MODEL_FILE = "air_quality_rf_model.joblib"

st.set_page_config(page_title="Air Quality AI Monitor", page_icon="🍃", layout="wide")

# ================= QUEUE SYSTEM (SOLUSI THREADING) =================
# Kita buat "kotak surat" yang bisa diakses oleh MQTT dan Streamlit dengan aman
if 'mqtt_queue' not in st.session_state:
    st.session_state.mqtt_queue = queue.Queue()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        artifact = joblib.load(MODEL_FILE)
        return artifact["model"], artifact["label_encoder"], artifact["features"]
    except Exception as e:
        return None, None, None

model, label_encoder, features = load_model()

# ================= SESSION STATE =================
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {"temp": 0, "hum": 0, "gas": 0, "timestamp": "-"}
if 'pred_result' not in st.session_state:
    st.session_state.pred_result = {"label": "Menunggu...", "confidence": 0}
if 'history' not in st.session_state:
    st.session_state.history = []

# ================= MQTT LOGIC =================
def on_message(client, userdata, msg):
    # userdata adalah queue
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic
        if userdata is not None:
            userdata.put((topic, payload))
    except Exception as e:
        print(f"Error di thread MQTT: {e}")

@st.cache_resource
def start_mqtt():
    client = mqtt.Client(client_id="Streamlit_AI_Cloud_V2", clean_session=True)
    # user_data akan diset di main thread
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
        client.subscribe(TOPIC_DATA)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"MQTT Error: {e}")
        return None

mqtt_client = start_mqtt()

if mqtt_client:
    mqtt_client.user_data_set(st.session_state.mqtt_queue)

# ================= PROCESS QUEUE (MAIN THREAD) =================
# Di sini kita bongkar "kotak surat" dari MQTT dan update tampilan
# Loop ini akan mengambil SEMUA pesan yang menumpuk di antrian
while not st.session_state.mqtt_queue.empty():
    topic, payload = st.session_state.mqtt_queue.get()
    
    if topic == TOPIC_DATA:
        # 1. Ambil Data
        temp = float(payload.get("temperature", 0))
        hum = float(payload.get("humidity", 0))
        gas = float(payload.get("gas_ppm", 0))
        timestamp = payload.get("timestamp", datetime.now().strftime("%H:%M:%S"))

        # Update Session State
        st.session_state.sensor_data = {"temp": temp, "hum": hum, "gas": gas, "timestamp": timestamp}
        
        # Update History
        new_record = {"time": timestamp, "Temp": temp, "Hum": hum, "Gas": gas}
        st.session_state.history.append(new_record)
        if len(st.session_state.history) > 50: st.session_state.history.pop(0)

        # 2. LAKUKAN PREDIKSI (INFERENCE)
        if model is not None:
            input_df = pd.DataFrame([[temp, hum, gas]], columns=features)
            
            pred_idx = model.predict(input_df)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            proba = model.predict_proba(input_df)[0]
            confidence = round(proba[pred_idx] * 100, 1)

            st.session_state.pred_result = {"label": pred_label, "confidence": confidence}

            # 3. KIRIM BALIK KE ESP32
            if mqtt_client is not None:
                resp = {
                    "label": pred_label,
                    "confidence": confidence,
                    "device_id": "Streamlit-Cloud"
                }
                mqtt_client.publish(TOPIC_PRED, json.dumps(resp))

# ================= TAMPILAN UI =================
import altair as alt

# ================= TAMPILAN UI =================
# Custom CSS untuk styling yang lebih modern & Estetik
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
        color: #1e3a8a; /* Dark Blue */
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Card Styling */
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
    
    /* Hero Status Card */
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
    .status-bg-icon {
        position: absolute;
        top: -20px;
        right: -20px;
        font-size: 10rem;
        opacity: 0.1;
    }
    
    /* Charts */
    .chart-wrapper {
        background: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">🍃 Air Quality AI Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Real-time Environmental Sensing</div>', unsafe_allow_html=True)

# ================= STATUS UTAMA (HERO SECTION) =================
lbl = st.session_state.pred_result['label']
conf = st.session_state.pred_result['confidence']
timestamp = st.session_state.sensor_data['timestamp']

# Definisi Warna & Icon
states = {
    "Baik": {"color": "#10b981", "icon": "😊", "msg": "Kualitas udara sangat baik. Nikmati harimu!"},
    "Sedang": {"color": "#f59e0b", "icon": "😐", "msg": "Kualitas udara cukup. Sensitif? Hati-hati."},
    "Tidak_Sehat": {"color": "#f97316", "icon": "😷", "msg": "Udara kotor. Kurangi aktivitas luar ruangan."},
    "Berbahaya": {"color": "#ef4444", "icon": "☠️", "msg": "BAHAYA! Gunakan masker N95 atau tetap di dalam."}
}

current_state = states.get(lbl, {"color": "#64748b", "icon": "❓", "msg": "Menunggu Data..."})
bg_color = current_state["color"]

st.markdown(f"""
    <div class="status-card" style="background: linear-gradient(135deg, {bg_color}, {bg_color}cc);">
        <div style="font-size: 5rem; margin-bottom: 0px; text-shadow: 0 4px 10px rgba(0,0,0,0.2);">{current_state['icon']}</div>
        <h3 style="margin:0; font-weight:400; opacity:0.9; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;">Status Udara</h3>
        <h1 style="margin: 5px 0 15px 0; font-size: 3.5rem; font-weight:800; letter-spacing: 1px;">{lbl.replace("_", " ")}</h1>
        <p style="font-size: 1.1rem; opacity: 0.95; font-style: italic; background: rgba(0,0,0,0.1); display: inline-block; padding: 5px 15px; border-radius: 15px;">"{current_state['msg']}"</p>
        <div style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
            Confidence: <b>{conf}%</b> &nbsp;•&nbsp; Last Update: {timestamp}
        </div>
    </div>
""", unsafe_allow_html=True)

# ================= METRICS ROW =================
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

# ================= ALTAIR CHARTS =================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📊 Analytics Overview")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    # Ensure numerical types
    df['Gas'] = pd.to_numeric(df['Gas'])
    df['Temp'] = pd.to_numeric(df['Temp'])
    df['Hum'] = pd.to_numeric(df['Hum'])
    
    # Reset index to get a sequential 'step' for charting if time is string
    df = df.reset_index(names='step')

    with st.container():
        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
        
        tab_gas, tab_env = st.tabs(["💨 Gas Quality Trend", "🌡️ Environment Data"])
        
        with tab_gas:
            # Area Chart for Gas
            chart_gas = alt.Chart(df).mark_area(
                line={'color':'#f59e0b'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='#fef3c7', offset=0),
                           alt.GradientStop(color='#f59e0b', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x=alt.X('step', title='Time Steps'),
                y=alt.Y('Gas', title='Gas (ppm)', scale=alt.Scale(zero=False)),
                tooltip=['time', 'Gas']
            ).properties(height=300).interactive()
            
            st.altair_chart(chart_gas, use_container_width=True)
            
        with tab_env:
            # Dual Line Chart
            base = alt.Chart(df).encode(x=alt.X('step', title='Time Steps'))
            
            line_temp = base.mark_line(color='#ef4444', text='Temp').encode(
                y=alt.Y('Temp', title='Temperature (°C)', scale=alt.Scale(zero=False)),
                tooltip=['time', 'Temp']
            )
            
            line_hum = base.mark_line(color='#3b82f6').encode(
                y=alt.Y('Hum', title='Humidity (%)', scale=alt.Scale(zero=False)),
                tooltip=['time', 'Hum']
            )
            
            st.altair_chart((line_temp + line_hum).interactive(), use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("⏳ Menunggu data streaming dari perangkat...")

# Auto Refresh logic
time.sleep(2)
st.rerun()