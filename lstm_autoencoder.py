import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import json

FEATURES = [
    "login_hour", "session_duration_min", "bytes_transferred_mb",
    "failed_logins", "endpoints_accessed", "sensitive_endpoints_hit",
    "unique_ips", "commands_executed"
]
SEQ_LEN = 10
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.001
HIDDEN = 64
LATENT = 16

# ── MITRE ATT&CK mapping ───────────────────────────────────────────────────
MITRE_MAP = {
    "brute_force":          {"id": "T1110", "name": "Brute Force",
                              "tactic": "Credential Access",
                              "action": "Block source IP, force MFA, alert SOC"},
    "data_exfiltration":    {"id": "T1041", "name": "Exfiltration Over C2",
                              "tactic": "Exfiltration",
                              "action": "Throttle egress, isolate session, preserve logs"},
    "credential_abuse":     {"id": "T1078", "name": "Valid Accounts",
                              "tactic": "Defense Evasion",
                              "action": "Revoke token, enforce re-auth, audit access"},
    "lateral_movement":     {"id": "T1021", "name": "Remote Services",
                              "tactic": "Lateral Movement",
                              "action": "Isolate endpoint, map blast radius, alert IR team"},
    "privilege_escalation": {"id": "T1068", "name": "Privilege Escalation",
                              "tactic": "Privilege Escalation",
                              "action": "Revoke privileges, snapshot VM state, escalate to human"},
}

def classify_anomaly(row):
    """Rule engine: map feature signature → anomaly type"""
    if row["failed_logins"] >= 10:
        return "brute_force"
    if row["bytes_transferred_mb"] >= 200:
        return "data_exfiltration"
    if row["unique_ips"] >= 6:
        return "lateral_movement"
    if row["commands_executed"] >= 200 and row["sensitive_endpoints_hit"] >= 4:
        return "privilege_escalation"
    if row["login_hour"] <= 4 or row["login_hour"] >= 22:
        return "credential_abuse"
    return "unknown"

# ── Data prep ──────────────────────────────────────────────────────────────
def prepare_data(csv_path="data/network_logs.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[FEATURES])
    y = df["is_anomaly"].values
    anomaly_types = df["anomaly_type"].values

    # Build sequences
    Xs, ys, ats = [], [], []
    for i in range(len(X) - SEQ_LEN):
        Xs.append(X[i:i+SEQ_LEN])
        ys.append(y[i+SEQ_LEN-1])
        ats.append(anomaly_types[i+SEQ_LEN-1])

    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys)
    return Xs, ys, ats, scaler, df

# ── LSTM Autoencoder ───────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden, latent, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        # Encoder
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc2latent = nn.Linear(hidden, latent)
        # Decoder
        self.latent2dec = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.enc2latent(h[-1])
        h_dec = self.latent2dec(z).unsqueeze(0)
        # Repeat latent across sequence length
        dec_input = h_dec.permute(1,0,2).repeat(1, self.seq_len, 1)
        dec_out, _ = self.dec(dec_input, (h_dec, torch.zeros_like(h_dec)))
        return self.out(dec_out)

# ── Train ──────────────────────────────────────────────────────────────────
def train_model(Xs, ys):
    # Train ONLY on normal traffic
    normal_mask = ys == 0
    split = int(0.8 * normal_mask.sum())
    normal_X = Xs[normal_mask]
    train_X = torch.tensor(normal_X[:split])
    val_X   = torch.tensor(normal_X[split:])

    loader = DataLoader(TensorDataset(train_X), batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMAutoencoder(len(FEATURES), HIDDEN, LATENT, SEQ_LEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Training LSTM Autoencoder...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_recon = model(val_X)
                val_loss = criterion(val_recon, val_X).item()
            print(f"  Epoch {epoch+1:02d}/{EPOCHS} | train_loss={total_loss/len(loader):.5f} | val_loss={val_loss:.5f}")

    return model

# ── Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model, Xs, ys, anomaly_types):
    model.eval()
    X_tensor = torch.tensor(Xs)
    with torch.no_grad():
        recon = model(X_tensor)
    errors = ((recon - X_tensor)**2).mean(dim=(1,2)).numpy()

    # Threshold = 95th percentile of normal reconstruction errors
    normal_errors = errors[ys == 0]
    threshold = np.percentile(normal_errors, 95)

    preds = (errors > threshold).astype(int)

    print("\n── LSTM Autoencoder Results ──────────────────────────")
    print(f"Threshold (95th pct normal error): {threshold:.5f}")
    print(classification_report(ys, preds, target_names=["Normal","Anomaly"]))
    f1  = f1_score(ys, preds)
    auc = roc_auc_score(ys, errors)
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    return errors, threshold, preds, f1, auc

def baseline_isolation_forest(Xs, ys):
    X_flat = Xs.reshape(len(Xs), -1)
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X_flat[ys == 0])
    iso_preds = (iso.predict(X_flat) == -1).astype(int)

    print("\n── Isolation Forest Baseline ─────────────────────────")
    print(classification_report(ys, iso_preds, target_names=["Normal","Anomaly"]))
    f1_iso = f1_score(ys, iso_preds)
    print(f"F1 Score : {f1_iso:.4f}")
    return f1_iso

# ── Response Orchestrator ──────────────────────────────────────────────────
def orchestrate_response(row_dict, anomaly_type, reconstruction_error, threshold):
    mitre = MITRE_MAP.get(anomaly_type, {
        "id": "T????", "name": "Unknown",
        "tactic": "Unknown", "action": "Manual investigation required"
    })
    severity = "HIGH" if reconstruction_error > threshold * 2 else "MEDIUM"
    human_gate = severity == "HIGH" or anomaly_type in ["lateral_movement", "privilege_escalation"]

    return {
        "alert_id": f"ALERT-{np.random.randint(10000,99999)}",
        "severity": severity,
        "user_id": row_dict.get("user_id", "unknown"),
        "mitre_technique_id": mitre["id"],
        "mitre_technique_name": mitre["name"],
        "mitre_tactic": mitre["tactic"],
        "reconstruction_error": round(float(reconstruction_error), 5),
        "threshold": round(float(threshold), 5),
        "automated_action": mitre["action"],
        "human_escalation_required": human_gate,
        "reason": f"Reconstruction error {reconstruction_error:.4f} exceeds threshold {threshold:.4f}"
    }

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Xs, ys, anomaly_types, scaler, df = prepare_data()
    print(f"Dataset: {len(Xs)} sequences | Normal: {(ys==0).sum()} | Anomaly: {(ys==1).sum()}")

    model = train_model(Xs, ys)
    errors, threshold, preds, f1_lstm, auc_lstm = evaluate(model, Xs, ys, anomaly_types)
    f1_iso = baseline_isolation_forest(Xs, ys)

    print("\n── Model Comparison ──────────────────────────────────")
    print(f"LSTM Autoencoder F1 : {f1_lstm:.4f}  ✓")
    print(f"Isolation Forest F1 : {f1_iso:.4f}")
    improvement = ((f1_lstm - f1_iso) / f1_iso) * 100
    print(f"LSTM improvement    : +{improvement:.1f}%")

    # Demo orchestrator on detected anomalies
    print("\n── Sample Incident Responses ─────────────────────────")
    detected_idx = np.where((preds == 1) & (ys == 1))[0][:3]
    responses = []
    for idx in detected_idx:
        orig_idx = idx + SEQ_LEN - 1
        row = df.iloc[orig_idx].to_dict()
        atype = classify_anomaly(row)
        response = orchestrate_response(row, atype, errors[idx], threshold)
        responses.append(response)
        print(f"\n  User     : {response['user_id']}")
        print(f"  MITRE    : {response['mitre_technique_id']} — {response['mitre_technique_name']}")
        print(f"  Severity : {response['severity']}")
        print(f"  Action   : {response['automated_action']}")
        print(f"  Human gate required: {response['human_escalation_required']}")

    # Save model + metadata
    torch.save(model.state_dict(), "models/lstm_model.pt")
    meta = {
        "threshold": float(threshold),
        "f1_lstm": float(f1_lstm),
        "f1_iso": float(f1_iso),
        "roc_auc": float(auc_lstm),
        "features": FEATURES,
        "seq_len": SEQ_LEN
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n\nModel saved: lstm_model.pt")
    print("Metadata  : model_meta.json")
