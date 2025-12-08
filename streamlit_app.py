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
def start_mqtt(q):
    client = mqtt.Client(client_id="Streamlit_AI_Cloud_V2", clean_session=True)
    client.user_data_set(q)
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
        client.subscribe(TOPIC_DATA)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"MQTT Error: {e}")
        return None

mqtt_client = start_mqtt(st.session_state.mqtt_queue)

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
st.title("🍃 AI Air Quality Monitoring")
st.markdown("### Random Forest Inference on Streamlit Cloud")
st.markdown("---")

# Kolom Metrik & Status
col_metrics, col_status = st.columns([2, 1])

with col_metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("🌡️ Temperature", f"{st.session_state.sensor_data['temp']} °C")
    c2.metric("💧 Humidity", f"{st.session_state.sensor_data['hum']} %")
    c3.metric("💨 Gas PPM", f"{st.session_state.sensor_data['gas']} ppm")
    st.caption(f"Last Update: {st.session_state.sensor_data['timestamp']}")

with col_status:
    lbl = st.session_state.pred_result['label']
    conf = st.session_state.pred_result['confidence']
    
    colors = {
        "Baik": "#28a745",          # Hijau
        "Sedang": "#ffc107",        # Kuning
        "Tidak_Sehat": "#fd7e14",   # Oranye
        "Berbahaya": "#dc3545"      # Merah
    }
    bg_color = colors.get(lbl, "#6c757d")
    
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <h2 style="margin:0;">{lbl}</h2>
            <p style="margin:0;">Confidence: {conf}%</p>
        </div>
    """, unsafe_allow_html=True)

# Grafik
st.subheader("📈 Real-time Data")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.line_chart(df, x="time", y=["Gas", "Temp", "Hum"])
else:
    st.info("Menunggu data dari ESP32...")

# Auto Refresh (Penting untuk mengambil data dari Queue)
time.sleep(2)
st.rerun()