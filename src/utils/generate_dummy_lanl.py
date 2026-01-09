import random
import gzip
import os

# --- CONFIGURATION ---
OUTPUT_DIR = 'data/raw'
AUTH_FILE = os.path.join(OUTPUT_DIR, 'auth.txt.gz')
REDTEAM_FILE = os.path.join(OUTPUT_DIR, 'redteam.txt.gz')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Generating CLUSTERED dummy LANL data...")

# 1. Define Clusters (Silos)
# HR Department: Users 1-20, Computers 1-10
hr_users = [f"User{i}" for i in range(1, 21)]
hr_comps = [f"Comp{i}" for i in range(1, 11)]

# Engineering Department: Users 50-70, Computers 50-60
eng_users = [f"User{i}" for i in range(50, 71)]
eng_comps = [f"Comp{i}" for i in range(50, 61)]

data = []

# 2. Generate NORMAL Traffic (Strictly inside clusters)
# HR stays in HR, Eng stays in Eng
for i in range(5000):
    # HR Traffic
    data.append(f"{i},{random.choice(hr_users)}@DOM1,{random.choice(hr_comps)},{random.choice(hr_comps)}")
    # Eng Traffic
    data.append(f"{i},{random.choice(eng_users)}@DOM1,{random.choice(eng_comps)},{random.choice(eng_comps)}")

# 3. Inject LATERAL MOVEMENT (The Attack)
# An HR User (User5) jumps into an Engineering Computer (Comp55)
# This is anomalous because HR and Eng computers have NEVER talked before.
attacks = [
    "6000,User5@DOM1,Comp1,Comp55",  # The Jump!
    "6001,User5@DOM1,Comp55,Comp56" # Moving around inside Eng
]
data.extend(attacks)

# Sort by time
data.sort(key=lambda x: int(x.split(',')[0]))

# 4. Save Files
print(f"Writing {len(data)} lines to {AUTH_FILE}...")
with gzip.open(AUTH_FILE, 'wt') as f:
    for line in data:
        f.write(line + '\n')

# Save Red Team labels
with open(REDTEAM_FILE.replace('.gz', ''), 'w') as f:
    for line in attacks:
        f.write(line + '\n')
        
# Compress redteam
with open(REDTEAM_FILE.replace('.gz', ''), 'rb') as f_in:
    with gzip.open(REDTEAM_FILE, 'wb') as f_out:
        f_out.writelines(f_in)

print("✅ Clustered Data Created. Run the model again!")