import pandas as pd
import numpy as np

# -----------------------------
# 📂 STEP 1: Load dataset
# -----------------------------
df = pd.read_csv("LDAP.csv", low_memory=False)

print("✅ Dataset loaded")

# -----------------------------
# 🔥 STEP 2: Fix column names
# -----------------------------
# Remove leading/trailing spaces
df.columns = df.columns.str.strip()

print("\nColumns after cleaning:")
print(df.columns.tolist())


# -----------------------------
# 🧹 STEP 3: Drop useless columns
# -----------------------------
drop_cols = [
    'Unnamed: 0',
    'Flow ID',
    'Source IP',
    'Destination IP',
    'Timestamp',
    'SimillarHTTP'   # optional, causes warning
]

df.drop(columns=drop_cols, inplace=True, errors='ignore')


# -----------------------------
# 🧹 STEP 4: Handle bad values
# -----------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("\n✅ After cleaning NaN/inf:", df.shape)


# -----------------------------
# 🎯 STEP 5: Define features
# -----------------------------
features = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Mean',
    'Packet Length Std',
    'Max Packet Length',
    'Min Packet Length',
    'SYN Flag Count',
    'ACK Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'Down/Up Ratio',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Flow IAT Mean'
]


# -----------------------------
# 🔍 STEP 6: Check missing columns
# -----------------------------
missing = [col for col in features if col not in df.columns]

if missing:
    print("\n❌ Missing columns:")
    print(missing)
    print("\n👉 Check column names above and fix if needed.")
    exit()


# -----------------------------
# 📊 STEP 7: Keep only needed data
# -----------------------------
df = df[features + ['Label']]

print("\n✅ Selected features")


# -----------------------------
# ⚖️ STEP 8: Check class distribution
# -----------------------------
print("\nLabel distribution BEFORE sampling:")
print(df['Label'].value_counts())


# -----------------------------
# ⚖️ STEP 9: Balanced sampling
# -----------------------------
# Adjust sample size if needed
sample_size = 20000

df_benign = df[df['Label'] == 'BENIGN']
df_ldap = df[df['Label'] == 'LDAP']
df_netbios = df[df['Label'] == 'NetBIOS']

# Safe sampling
df_benign = df_benign.sample(n=min(sample_size, len(df_benign)), random_state=42)
df_ldap = df_ldap.sample(n=min(sample_size, len(df_ldap)), random_state=42)
df_netbios = df_netbios.sample(n=min(sample_size, len(df_netbios)), random_state=42)

# Combine
df_final = pd.concat([df_benign, df_ldap, df_netbios])

# Shuffle
df_final = df_final.sample(frac=1, random_state=42)


# -----------------------------
# 💾 STEP 10: Save cleaned dataset
# -----------------------------
df_final.to_csv("cleaned_ddos_data.csv", index=False)

print("\n✅ CLEANED DATASET SAVED AS: cleaned_ddos_data.csv")

print("\nFinal dataset shape:", df_final.shape)
print("\nFinal label distribution:")
print(df_final['Label'].value_counts())
