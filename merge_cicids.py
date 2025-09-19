import os
import pandas as pd

DATA_FOLDER = "data"   
OUTPUT_FILE = "data/cicids2017_full.csv"

# Collect files with .csv or .pcap_ISCX
files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith((".csv", ".pcap_iscx"))]

if not files:
    print(" No CSV or .pcap_ISCX files found in", DATA_FOLDER)
    exit(1)

print("📂 Found files:", files)

dfs = []
for file in files:
    path = os.path.join(DATA_FOLDER, file)
    print(f" Loading {file} ...")
    try:
        df = pd.read_csv(path)
        dfs.append(df)
    except Exception as e:
        print(f" Skipping {file}: {e}")

if not dfs:
    print(" No data loaded, exiting.")
    exit(1)

full_df = pd.concat(dfs, ignore_index=True)
full_df.to_csv(OUTPUT_FILE, index=False)
print(f" Merged dataset saved to {OUTPUT_FILE}, shape = {full_df.shape}")
