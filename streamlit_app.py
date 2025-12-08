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
# Custom CSS untuk styling yang lebih modern
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stMetric label {
        color: #6c757d;
        font-weight: 500;
    }
    .stMetric .css-1wivap2 {
        font-size: 24px;
        color: #212529;
        font-weight: 700;
    }
    .status-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    h3 {
        color: #34495e;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🍃 Air Quality AI Monitor")
st.markdown("### Intelligent Environmental Sensing System")

# ================= STATUS UTAMA (HERO SECTION) =================
lbl = st.session_state.pred_result['label']
conf = st.session_state.pred_result['confidence']
timestamp = st.session_state.sensor_data['timestamp']

colors = {
    "Baik": ("#28a745", "😊", "Udara Bersih"),
    "Sedang": ("#ffc107", "😐", "Cukup Baik"),
    "Tidak_Sehat": ("#fd7e14", "😷", "Kurangi Aktivitas Luar"),
    "Berbahaya": ("#dc3545", "☠️", "BAHAYA! Pakai Masker")
}

color, icon, suggestion = colors.get(lbl, ("#6c757d", "❓", "Menunggu Data..."))

st.markdown(f"""
    <div class="status-card" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
        <h3 style="color:white; margin:0; opacity:0.9;">Kualitas Udara Saat Ini</h3>
        <h1 style="color:white; font-size: 3.5rem; margin: 10px 0;">{icon} {lbl}</h1>
        <p style="font-size: 1.2rem; margin:0;">Confidence AI: <b>{conf}%</b></p>
        <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
        <p style="font-size: 1.1rem; font-style: italic;">"{suggestion}"</p>
        <p style="font-size: 0.8rem; margin-top: 10px; opacity: 0.8;">Last Update: {timestamp}</p>
    </div>
""", unsafe_allow_html=True)

# ================= METRICS ROW =================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡️ Temperature", f"{st.session_state.sensor_data['temp']} °C", delta=None)

with col2:
    st.metric("💧 Humidity", f"{st.session_state.sensor_data['hum']} %", delta=None)

with col3:
    st.metric("💨 Gas Level", f"{st.session_state.sensor_data['gas']} ppm", delta_color="inverse")

# ================= GRAFIK DETIL =================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📈 Real-time Trend Analysis")

if st.session_state.history:
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Konversi ke DataFrame
        df = pd.DataFrame(st.session_state.history)
        
        # Custom Chart Tabs
        tab1, tab2 = st.tabs(["Gas Levels", "Environmental Ops"])
        
        with tab1:
            st.line_chart(df.set_index("time")["Gas"], color="#fd7e14", height=250)
            
        with tab2:
            st.line_chart(df.set_index("time")[["Temp", "Hum"]], color=["#dc3545", "#007bff"], height=250)
            
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("⏳ Menunggu stream data dari perangkat IoT...")

# Auto Refresh logic
time.sleep(2)
st.rerun()