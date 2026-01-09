import pandas as pd

class HeadOfWisdom:
    def __init__(self):
        # --- 1. THE KNOWLEDGE GRAPH (Symbolic Layer) ---
        # In a real app, this comes from Neo4j or a CMDB
        self.assets = {
            "Comp1": {"role": "Workstation", "dept": "HR", "criticality": "Low"},
            "Comp55": {"role": "Server", "dept": "Engineering", "criticality": "HIGH"},
        }
        
        self.users = {
            "User5@DOM1": {"role": "Intern", "dept": "HR"},
            "User666@DOM1": {"role": "Unknown", "dept": "External"},
        }
        
        # --- 2. THE POLICIES ---
        self.policies = [
            {
                "id": "POL-001",
                "desc": "Inter-Department Access Violation",
                "condition": lambda u, src, dst: u['dept'] != dst['dept']
            },
            {
                "id": "POL-002",
                "desc": "Intern Accessing Critical Asset",
                "condition": lambda u, src, dst: u['role'] == 'Intern' and dst['criticality'] == 'HIGH'
            }
        ]

    def enrich_alert(self, anomaly_row):
        """
        Takes a raw AI alert and adds 'Wisdom' (Context & Explanations)
        """
        user_id = anomaly_row['user']
        src_id = anomaly_row['src']
        dst_id = anomaly_row['dst']
        
        # 1. Fetch Context
        # (Use .get() to handle unknown/new nodes gracefully)
        user_ctx = self.users.get(user_id, {"role": "Unknown", "dept": "Unknown"})
        src_ctx = self.assets.get(src_id, {"role": "Unknown", "dept": "Unknown", "criticality": "Low"})
        dst_ctx = self.assets.get(dst_id, {"role": "Unknown", "dept": "Unknown", "criticality": "Low"})
        
        explanations = []
        severity = "Low"
        
        # 2. Apply Logic Rules (Symbolic Reasoning)
        for policy in self.policies:
            if policy["condition"](user_ctx, src_ctx, dst_ctx):
                explanations.append(f"VIOLATION: {policy['desc']}")
                
        # 3. Determine Severity
        if dst_ctx['criticality'] == 'HIGH':
            severity = "CRITICAL"
        elif explanations:
            severity = "MEDIUM"
            
        return {
            "Original_Event": f"{user_id} connected to {dst_id}",
            "Context": f"User is {user_ctx['role']} ({user_ctx['dept']}) accessing {dst_ctx['role']}",
            "Explanation": " | ".join(explanations),
            "Severity": severity
        }

# --- TEST THE WISDOM ---
if __name__ == "__main__":
    # Simulate the anomaly you just caught in Phase 3
    raw_anomaly = pd.Series({
        "time": 6000, 
        "user": "User5@DOM1", 
        "src": "Comp1", 
        "dst": "Comp55" # The HR -> Eng Jump
    })
    
    wisdom = HeadOfWisdom()
    enrichment = wisdom.enrich_alert(raw_anomaly)
    
    print("\n--- HYDRA HEAD OF WISDOM REPORT ---")
    for k, v in enrichment.items():
        print(f"{k}: {v}")