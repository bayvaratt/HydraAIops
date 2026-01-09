from sys import displayhook
import pandas as pd
import networkx as nx
import gzip

def train_link_prediction(auth_file):
    print(f"Training Link Predictor on {auth_file}...")
    
    # 1. Load Data (Normal traffic only for training)
    # In a real system, we'd split by time (Train = Day 1-20, Test = Day 21)
    # Here we just take the first 90% as "History"
    df = pd.read_csv(auth_file, names=['time', 'user', 'src', 'dst'])
    
    # Simple Train/Test Split
    split_idx = int(len(df) * 0.9)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    # 2. Build the "Normal" Graph
    G = nx.Graph() # Undirected for Link Prediction heuristics
    edges = list(zip(df_train['src'], df_train['dst']))
    G.add_edges_from(edges)
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    return G, df_test

def predict_anomalies(G, df_test):
    print("\nScanning for Lateral Movement (Anomalous Links)...")
    
    alerts = []
    
    # Check every new event in the Test set
    for index, row in df_test.iterrows():
        u, v = row['src'], row['dst']
        
        # Skip if nodes don't exist in our history (New computers)
        if not G.has_node(u) or not G.has_node(v):
            continue
            
        # 3. Calculate Adamic-Adar Score
        # This checks: "Do these two computers share common neighbors?"
        # Higher score = More legitimate connection.
        # Score 0 = They are total strangers (Suspicious!)
        try:
            preds = nx.adamic_adar_index(G, [(u, v)])
            score = next(preds)[2]
            
            # 4. THRESHOLDING
            # If the score is 0.0, it means there is NO historical reason 
            # for these computers to talk. It is likely a lateral jump.
            if score == 0.0:
                alerts.append(row)
                
        except Exception:
            pass
            
    return pd.DataFrame(alerts)

# --- RUNNER ---
if __name__ == "__main__":
    # Point this to your dummy file for now
    AUTH_FILE = 'data/raw/auth.txt.gz'
    
    G, df_test = train_link_prediction(AUTH_FILE)
    anomalies = predict_anomalies(G, df_test)
    
    print("-" * 30)
    print(f"DETECTED {len(anomalies)} ANOMALIES")
    
    if not anomalies.empty:
        print("Top suspicious lateral movements:")
        # Look for the 'User666' from your dummy data!
        displayhook(anomalies.head())
    else:
        print("No anomalies found (Try generating more dummy Red Team events).")