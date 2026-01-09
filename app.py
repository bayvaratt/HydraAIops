import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
import sys
import os

# Add the src folder to Python's path so we can import our new modules
sys.path.append('src')

# Import our custom modules
# Wrap imports in try-except to prevent crashing if files aren't ready yet
try:
    from models.predictive_head import train_link_prediction, predict_anomalies
    from wisdom.core import HeadOfWisdom
except ImportError:
    pass # Will handle inside the app logic if needed

import networkx as nx
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="HYDRA: Neuro-Symbolic AIOps", layout="wide")
st.title("🐍 HYDRA: Cyber Defense Platform")

# --- SIDEBAR ---
st.sidebar.header("Select Module")
# FIXED: Added the third option to this list so you can actually click it!
app_mode = st.sidebar.selectbox(
    "Choose Analysis Head", 
    [
        "Statistical Head (Network)", 
        "Semantic Head (Logs)", 
        "Predictive Head (Lateral Movement)"
    ]
)

# --- 1. STATISTICAL HEAD (Network) ---
if app_mode == "Statistical Head (Network)":
    st.subheader("🛡️ Network Anomaly Detection")
    st.write("Upload a CSV of network flows (TON_IoT format) to detect DDoS, Backdoors, etc.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload Network CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load Model
            model = joblib.load('src/models/statistical_head_rf.pkl')
            
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {df.shape[0]} flows.")
            
            # --- Quick Preprocessing ---
            cols_to_drop = ['src_ip', 'dst_ip', 'ts', 'date', 'time', 'label', 'type']
            existing_drop = [c for c in cols_to_drop if c in df.columns]
            X = df.drop(columns=existing_drop)
            
            # Select numeric only for the demo
            X_numeric = X.select_dtypes(include=[np.number])
            
            if st.button("Analyze Traffic"):
                predictions = model.predict(X_numeric)
                df['Prediction'] = predictions
                
                # Highlight Threats
                threats = df[df['Prediction'] != 'normal']
                
                st.metric("Threats Detected", len(threats), delta_color="inverse")
                
                if not threats.empty:
                    st.warning("⚠️ Malicious Traffic Found!")
                    st.dataframe(threats.head(20))
                    st.bar_chart(threats['Prediction'].value_counts())
                else:
                    st.success("✅ System Clean. No anomalies detected.")
                    
        except Exception as e:
            st.error(f"Error loading model or data: {e}")
            st.info("Tip: Run the '01_statistical_exploration' notebook to train and save the model first.")

# --- 2. SEMANTIC HEAD (Logs) ---
elif app_mode == "Semantic Head (Logs)":
    st.subheader("📜 Log Anomaly Detection")
    st.write("Upload a .log file (HDFS format) to detect rare/anomalous system events.")
    
    uploaded_log = st.file_uploader("Upload Log File", type=["log", "txt"])
    
    def normalize_log(text):
        text = re.sub(r'blk_-?\d+', 'BLK_ID', text)
        text = re.sub(r'\d+\.\d+\.\d+\.\d+(:\d+)?', 'IP_ADDR', text)
        text = re.sub(r'\d+', 'NUM', text)
        return text

    if uploaded_log is not None:
        try:
            vectorizer = joblib.load('src/models/log_vectorizer.pkl')
            model = joblib.load('src/models/semantic_head_iso.pkl')
            
            stringio = uploaded_log.getvalue().decode("utf-8")
            lines = stringio.splitlines()
            
            if st.button("Scan Logs"):
                data = []
                log_pattern = re.compile(r'.*?(INFO|WARN|ERROR)\s+(.*)')
                
                for line in lines:
                    match = log_pattern.search(line)
                    if match:
                        content = match.group(2)
                        data.append({'raw': line, 'content': content})
                
                df_logs = pd.DataFrame(data)
                df_logs['template'] = df_logs['content'].apply(normalize_log)
                X_vec = vectorizer.transform(df_logs['template'])
                
                df_logs['score'] = model.predict(X_vec)
                anomalies = df_logs[df_logs['score'] == -1]
                
                st.metric("Anomalous Logs", len(anomalies), delta_color="inverse")
                
                if not anomalies.empty:
                    st.error("⚠️ Abnormal Log Patterns Detected")
                    st.dataframe(anomalies[['raw', 'template']])
                else:
                    st.success("✅ Logs look normal.")
                    
        except Exception as e:
            st.error(f"Error: {e}")

# --- 3. PREDICTIVE HEAD (Lateral Movement) ---
elif app_mode == "Predictive Head (Lateral Movement)":
    st.subheader("🔮 Lateral Movement Prediction")
    st.write("Analyzes user authentication graphs to detect unauthorized lateral movement.")

    data_source = st.radio("Select Data Source", ["Dummy Data (HR vs Eng Cluster)", "Real LANL Data"])
    
    # Check if file exists
    auth_file = 'data/raw/auth.txt.gz'
    
    if st.button("Build Graph & Detect Anomalies"):
        if not os.path.exists(auth_file):
            st.error("❌ Data file not found! Please run 'src/utils/generate_dummy_lanl.py' first.")
        else:
            try:
                with st.spinner('Training Link Prediction Model...'):
                    G, df_test = train_link_prediction(auth_file)
                    anomalies = predict_anomalies(G, df_test)
                
                if anomalies.empty:
                    st.success("✅ No lateral movement anomalies detected.")
                else:
                    st.error(f"⚠️ Detected {len(anomalies)} Suspicious Connections!")
                    
                    st.subheader("🧠 Head of Wisdom Analysis")
                    st.info("Applying symbolic logic and policy constraints to explain the AI detections...")
                    
                    wisdom = HeadOfWisdom()
                    
                    for index, row in anomalies.iterrows():
                        enrichment = wisdom.enrich_alert(row)
                        with st.expander(f"🚨 ALERT: {enrichment['Original_Event']} ({enrichment['Severity']})", expanded=True):
                            st.markdown(f"**Context:** {enrichment['Context']}")
                            st.markdown(f"**Violation:** `{enrichment['Explanation']}`")
                            st.markdown(f"**Severity:** **{enrichment['Severity']}**")
                    
                    st.subheader("🕸️ Attack Path Visualization")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pos = nx.spring_layout(G, k=0.5, seed=42)
                    
                    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue', alpha=0.7, ax=ax)
                    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                    
                    anomaly_edges = list(zip(anomalies['src'], anomalies['dst']))
                    nx.draw_networkx_edges(G, pos, edgelist=anomaly_edges, edge_color='red', width=2.5, ax=ax)
                    
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")