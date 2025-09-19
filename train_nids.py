import os
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Features from CICIDS2017 (after stripping spaces in column names)
EXPECTED_NUMERIC = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow IAT Mean",
    "Fwd IAT Mean",
    "Bwd IAT Mean"
]

# CICIDS2017 does not have a Protocol column
EXPECTED_CATEG = []  

# Possible target names
EXPECTED_TARGET = [
    "Label",
    "label",
    "Attack",
    "class",
    "Category",
    "Attack category",
    "Attack_label",
    "Label Name"
]

def normalize_label(y):
    return y.apply(lambda v: 0 if str(v).lower() in ["normal", "benign"] else 1)

def train_pipeline(csv_path, out_dir):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    print(f"📂 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # Strip spaces in column names
    df.columns = df.columns.str.strip()

    # Find target column
    target_col = None
    for c in EXPECTED_TARGET:
        if c in df.columns:
            target_col = c
            break
    if not target_col:
        print(" No label column found. Available columns are:")
        print(df.columns.tolist())
        raise ValueError(f"No label column found in {EXPECTED_TARGET}")

    print(f" Using target column: {target_col}")

    # Features + target
    missing = [c for c in EXPECTED_NUMERIC if c not in df.columns]
    if missing:
        print(" Warning: Missing expected numeric features:", missing)

    X = df[[c for c in EXPECTED_NUMERIC if c in df.columns]].copy()
    y = normalize_label(df[target_col])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [c for c in EXPECTED_NUMERIC if c in df.columns])
        ]
    )

    # Model
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    # Pipeline
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # Train
    print(" Training model...")
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, "nids_pipeline.joblib"))
    meta = {
        "numeric_features": EXPECTED_NUMERIC,
        "categorical_features": EXPECTED_CATEG,
        "accuracy": acc,
        "target_column": target_col
    }
    joblib.dump(meta, os.path.join(out_dir, "meta.joblib"))

    print(" Training complete")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print(f"Saved pipeline → {os.path.join(out_dir, 'nids_pipeline.joblib')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Monday-WorkingHours.pcap_ISCX", help="Path to CSV dataset")
    parser.add_argument("--out", default="model_artifacts", help="Output directory")
    args = parser.parse_args()
    train_pipeline(args.data, args.out)
