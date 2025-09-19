import os
import threading
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from scapy.all import sniff, IP, TCP, UDP
from collections import deque

app = Flask(__name__)

PIPELINE = joblib.load("model_artifacts/nids_pipeline.joblib")
META = joblib.load("model_artifacts/meta.joblib")

PACKET_BUFFER = deque(maxlen=200)
STATS = {"total": 0, "attacks": 0, "normal": 0}

def extract_features(pkt):
    if IP in pkt:
        feat = {
            "src": pkt[IP].src,
            "dst": pkt[IP].dst,
            "proto": pkt[IP].proto,
            "length": len(pkt),
            "sport": pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0),
            "dport": pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0),
        }

        # Minimal feature set for pipeline
        df = pd.DataFrame([{
            "Flow Duration": 0,
            "Total Fwd Packets": 1,
            "Total Backward Packets": 0,
            "Total Length of Fwd Packets": feat["length"],
            "Total Length of Bwd Packets": 0,
            "Flow IAT Mean": 0,
            "Fwd IAT Mean": 0,
            "Bwd IAT Mean": 0,
            "Protocol": str(feat["proto"]),
        }])

        try:
            pred = PIPELINE.predict(df)[0]
            label = "Attack" if pred == 1 else "Normal"
            prob = float(PIPELINE.predict_proba(df)[0][1])
        except Exception:
            label, prob = "Unknown", 0.0

        STATS["total"] += 1
        if label == "Attack":
            STATS["attacks"] += 1
        elif label == "Normal":
            STATS["normal"] += 1

        # Save to buffer
        PACKET_BUFFER.appendleft({
            "src": feat["src"],
            "dst": feat["dst"],
            "sport": feat["sport"],
            "dport": feat["dport"],
            "proto": feat["proto"],
            "length": feat["length"],
            "label": label,
            "prob": round(prob, 2)
        })

# Background Sniffer
def start_sniffer():
    sniff(prn=extract_features, store=False)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/data")
def data():
    return jsonify({"stats": STATS, "packets": list(PACKET_BUFFER)})

@app.route("/offline", methods=["GET", "POST"])
def offline():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(file)
            elif ext == ".pcap":
                from scapy.all import rdpcap
                packets = rdpcap(file)
                rows = []
                for pkt in packets:
                    if IP in pkt:
                        rows.append({
                            "Flow Duration": 0,
                            "Total Fwd Packets": 1,
                            "Total Backward Packets": 0,
                            "Total Length of Fwd Packets": len(pkt),
                            "Total Length of Bwd Packets": 0,
                            "Flow IAT Mean": 0,
                            "Fwd IAT Mean": 0,
                            "Bwd IAT Mean": 0,
                            "Protocol": str(pkt[IP].proto),
                        })
                df = pd.DataFrame(rows)
            else:
                return " Unsupported file format. Upload .csv or .pcap"

            #  Clean + enforce expected columns
            expected_cols = [
                "Flow Duration",
                "Total Fwd Packets",
                "Total Backward Packets",
                "Total Length of Fwd Packets",
                "Total Length of Bwd Packets",
                "Flow IAT Mean",
                "Fwd IAT Mean",
                "Bwd IAT Mean",
                "Protocol"
            ]

            # Strip spaces from CSV headers
            df.columns = df.columns.str.strip()

            # Add missing columns with default 0
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0

            # Reorder to match training
            df = df[expected_cols]

            preds = PIPELINE.predict(df)
            probs = PIPELINE.predict_proba(df)

            details = []
            for i, p in enumerate(preds):
                details.append({
                    "proto": df.iloc[i]["Protocol"],
                    "length": df.iloc[i]["Total Length of Fwd Packets"],
                    "label": "Attack" if p == 1 else "Normal",
                    "prob": float(probs[i][1]),
                })

            result = {
                "total": len(preds),
                "attacks": int(sum(preds)),
                "normal": int(len(preds) - sum(preds)),
                "details": details
            }

    return render_template("offline.html", result=result)

@app.route("/remedy", methods=["POST"])
def remedy():
    data = request.json
    action = data.get("action")
    target = data.get("target")

    message = f" Simulated remedy: {action} applied on {target}"
    print(message)

    return jsonify({"status": "ok", "message": message})

if __name__ == "__main__":
    t = threading.Thread(target=start_sniffer, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=True)
