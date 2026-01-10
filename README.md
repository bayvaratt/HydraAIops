# 🐍 HYDRA: Neuro-Symbolic AIOps & Cyber Defense

**HYDRA** is a hybrid intrusion detection system (IDS) that combines **Machine Learning (Neuro)** with **Knowledge Graph policies (Symbolic)** to detect, predict, and explain cyber threats. It addresses the "black box" problem in AIOps by enforcing logic constraints and context-aware reasoning on top of deep learning alerts.

---

## 🏗️ Architecture

HYDRA operates using a "Multi-Head" architecture where different AI models specialize in specific data types, unified by a neuro-symbolic reasoning engine.

1.  **Statistical Head (Network Flows):**
    * **Data:** TON_IoT (Netflow/Telemetry).
    * **Model:** Random Forest / XGBoost.
    * **Goal:** Detect high-volume volumetric attacks (DDoS, Scanning, Backdoors).

2.  **Semantic Head (System Logs):**
    * **Data:** HDFS / LogHub (Unstructured Text).
    * **Model:** TF-IDF + Isolation Forest (Unsupervised).
    * **Goal:** Detect rare system states, error sequences, and unknown "zero-day" anomalies.

3.  **Predictive Head (Lateral Movement):**
    * **Data:** LANL Unified Host & Network (Authentication Graphs).
    * **Model:** Graph Link Prediction (Adamic-Adar / Jaccard).
    * **Goal:** Predict and detect unauthorized user movement between network clusters (e.g., HR $\to$ Engineering).

4.  **🧠 Head of Wisdom (Reasoning Core):**
    * **Tech:** Symbolic Logic & Policy Engine.
    * **Goal:** Contextualizes alerts. Instead of just flagging an anomaly, it checks asset criticality and user roles to generate human-readable explanations (e.g., *"Intern violating Policy-002 by accessing Critical Server"*).

---

## 📂 Project Structure

```bash
hydra-aiops/
├── data/
│   ├── raw/                 # Place datasets here (auth.txt.gz, HDFS.log, Train_Test_Network.csv)
│   └── processed/           # Cleaned data (if applicable)
├── notebooks/
│   ├── 01_statistical_exploration.ipynb  # Training the Network Model
│   ├── 02_semantic_exploration.ipynb     # Training the Log Parser & Anomaly Detector
│   └── 03_predictive_exploration.ipynb   # Graph visualization & Link Prediction experiments
├── src/
│   ├── models/              # Saved .pkl models and training scripts
│   │   ├── predictive_head.py
│   │   └── ...
│   ├── wisdom/
│   │   └── core.py          # The Neuro-Symbolic Logic Engine
│   └── utils/
│       └── generate_dummy_lanl.py  # Script to generate synthetic graph data for testing
├── app.py                   # Main Streamlit Dashboard
└── requirements.txt         # Python dependencies
