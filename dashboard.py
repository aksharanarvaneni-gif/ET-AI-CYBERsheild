import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import json
import time
from datetime import datetime

st.set_page_config(
    page_title="CyberShield — AI Cyber Resilience",
    page_icon="🛡️",
    layout="wide"
)

FEATURES = [
    "login_hour", "session_duration_min", "bytes_transferred_mb",
    "failed_logins", "endpoints_accessed", "sensitive_endpoints_hit",
    "unique_ips", "commands_executed"
]
SEQ_LEN = 10

MITRE_MAP = {
    "brute_force":          {"id": "T1110", "name": "Brute Force",         "tactic": "Credential Access",     "color": "#e74c3c"},
    "data_exfiltration":    {"id": "T1041", "name": "Exfil Over C2",       "tactic": "Exfiltration",          "color": "#e67e22"},
    "credential_abuse":     {"id": "T1078", "name": "Valid Accounts",       "tactic": "Defense Evasion",       "color": "#9b59b6"},
    "lateral_movement":     {"id": "T1021", "name": "Remote Services",      "tactic": "Lateral Movement",      "color": "#e74c3c"},
    "privilege_escalation": {"id": "T1068", "name": "Privilege Escalation", "tactic": "Privilege Escalation",  "color": "#c0392b"},
    "unknown":              {"id": "T????", "name": "Unknown",               "tactic": "Unknown",               "color": "#7f8c8d"},
}

PLAYBOOKS = {
    "brute_force":          {"action": "Block source IP · Force MFA · Alert SOC",              "human_gate": False},
    "data_exfiltration":    {"action": "Throttle egress · Isolate session · Preserve logs",    "human_gate": False},
    "credential_abuse":     {"action": "Revoke token · Enforce re-auth · Audit access",        "human_gate": False},
    "lateral_movement":     {"action": "Isolate endpoint · Map blast radius · Alert IR team",  "human_gate": True},
    "privilege_escalation": {"action": "Revoke privileges · Snapshot VM · Escalate to human", "human_gate": True},
    "unknown":              {"action": "Manual investigation required",                         "human_gate": True},
}

# ── Model ──────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=8, hidden=64, latent=16, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc2latent = nn.Linear(hidden, latent)
        self.latent2dec = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.enc2latent(h[-1])
        h_dec = self.latent2dec(z).unsqueeze(0)
        dec_input = h_dec.permute(1,0,2).repeat(1, self.seq_len, 1)
        dec_out, _ = self.dec(dec_input, (h_dec, torch.zeros_like(h_dec)))
        return self.out(dec_out)

@st.cache_resource
def load_model_and_data():
    model = LSTMAutoencoder()
    model.load_state_dict(torch.load("models/lstm_model.pt", map_location="cpu"))
    model.eval()
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    df = pd.read_csv("data/network_logs.csv")
    df = df.sort_values("timestamp").reset_index(drop=True)
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES])
    return model, meta, df, scaler

def classify_anomaly(row):
    if row["failed_logins"] >= 10:           return "brute_force"
    if row["bytes_transferred_mb"] >= 200:   return "data_exfiltration"
    if row["unique_ips"] >= 6:               return "lateral_movement"
    if row["commands_executed"] >= 200 and row["sensitive_endpoints_hit"] >= 4:
        return "privilege_escalation"
    if row["login_hour"] <= 4 or row["login_hour"] >= 22:
        return "credential_abuse"
    return "unknown"

def run_inference(model, scaler, meta, rows_df):
    X = scaler.transform(rows_df[FEATURES])
    if len(X) < SEQ_LEN:
        X = np.pad(X, ((SEQ_LEN - len(X), 0), (0, 0)), mode='edge')
    seq = torch.tensor(X[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        recon = model(seq)
    error = ((recon - seq)**2).mean().item()
    is_anomaly = error > meta["threshold"]
    return error, is_anomaly

# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {background:#1e1e2e;border-radius:10px;padding:16px;border:1px solid #313155;}
.alert-high  {background:#2d0a0a;border-left:4px solid #e74c3c;padding:12px;border-radius:4px;margin:6px 0;}
.alert-med   {background:#2d1a0a;border-left:4px solid #e67e22;padding:12px;border-radius:4px;margin:6px 0;}
.mitre-badge {background:#1a1a3e;border:1px solid #4a4a8a;border-radius:4px;padding:2px 8px;font-size:12px;font-family:monospace;}
.human-gate  {background:#3d0a0a;border:1px solid #e74c3c;border-radius:4px;padding:4px 10px;font-size:12px;}
.auto-action {background:#0a2d1a;border:1px solid #27ae60;border-radius:4px;padding:4px 10px;font-size:12px;}
</style>
""", unsafe_allow_html=True)

model, meta, df, scaler = load_model_and_data()

# Header
st.markdown("# 🛡️ CyberShield — AI Cyber Resilience Platform")
st.markdown("**Behavioral Anomaly Detection · MITRE ATT&CK Mapping · Incident Response Orchestration**")
st.divider()

# Sidebar — controls
with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Mode", ["Live Demo", "Manual Input"])
    if mode == "Live Demo":
        n_events = st.slider("Events to simulate", 10, 100, 30)
        run_btn = st.button("▶ Run Simulation", type="primary")
    st.divider()
    st.markdown("**Model Performance**")
    st.metric("LSTM F1",        f"{meta['f1_lstm']:.3f}")
    st.metric("ROC-AUC",        f"{meta['roc_auc']:.3f}")
    st.metric("vs Isolation F.", f"+{((meta['f1_lstm']-meta['f1_iso'])/meta['f1_iso']*100):.1f}%")
    st.divider()
    st.markdown("**Threshold**")
    st.code(f"{meta['threshold']:.5f}")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📡 Live Monitor", "🗂️ MITRE Coverage", "🤖 Orchestrator", "📊 Analytics"])

# ── Tab 1: Live Monitor ────────────────────────────────────────────────────
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    if mode == "Live Demo" and 'run_btn' in dir() and run_btn:
        sample = df.sample(n_events).reset_index(drop=True)
        alerts = []
        normal_count = 0
        anomaly_count = 0

        progress = st.progress(0, text="Scanning events...")
        feed = st.empty()

        for i, row in sample.iterrows():
            window = df.iloc[max(0, df.index.get_loc(row.name)-SEQ_LEN):df.index.get_loc(row.name)+1]
            error, is_anomaly = run_inference(model, scaler, meta, window)

            if is_anomaly:
                anomaly_count += 1
                atype = classify_anomaly(row)
                mitre = MITRE_MAP[atype]
                play  = PLAYBOOKS[atype]
                severity = "HIGH" if error > meta["threshold"] * 2 else "MEDIUM"
                alerts.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "user": row["user_id"],
                    "dept": row["department"],
                    "mitre_id": mitre["id"],
                    "technique": mitre["name"],
                    "tactic": mitre["tactic"],
                    "severity": severity,
                    "error": round(error, 5),
                    "action": play["action"],
                    "human_gate": play["human_gate"],
                    "atype": atype
                })
            else:
                normal_count += 1

            progress.progress((i+1)/n_events, text=f"Scanned {i+1}/{n_events} events...")
            time.sleep(0.05)

        progress.empty()
        st.session_state["alerts"] = alerts
        st.session_state["normal"] = normal_count
        st.session_state["anomaly"] = anomaly_count

    alerts   = st.session_state.get("alerts", [])
    normal_c = st.session_state.get("normal", 0)
    anomaly_c= st.session_state.get("anomaly", 0)
    total    = normal_c + anomaly_c

    with col1: st.metric("Events Scanned",    total)
    with col2: st.metric("Anomalies Detected", anomaly_c, delta=f"{anomaly_c/total*100:.1f}%" if total else "0%")
    with col3: st.metric("Normal Traffic",     normal_c)
    with col4: st.metric("Alerts Generated",   len(alerts))

    st.subheader("Alert Feed")
    if alerts:
        for a in reversed(alerts[-15:]):
            css = "alert-high" if a["severity"] == "HIGH" else "alert-med"
            hg  = f'<span class="human-gate">⚠️ Human gate</span>' if a["human_gate"] else f'<span class="auto-action">✅ Auto-response</span>'
            st.markdown(f"""
<div class="{css}">
  <b>{a['time']}</b> &nbsp;|&nbsp; {a['user']} ({a['dept']}) &nbsp;|&nbsp; Severity: <b>{a['severity']}</b><br>
  <span class="mitre-badge">{a['mitre_id']}</span> &nbsp;<b>{a['technique']}</b> — {a['tactic']}<br>
  🔧 {a['action']} &nbsp;&nbsp; {hg}
</div>""", unsafe_allow_html=True)
    else:
        st.info("Run a simulation using the sidebar to see live alerts.")

# ── Tab 2: MITRE Coverage ──────────────────────────────────────────────────
with tab2:
    st.subheader("MITRE ATT&CK Technique Coverage")
    st.caption("Techniques this platform detects and responds to")

    tactics = {}
    for atype, m in MITRE_MAP.items():
        if atype == "unknown": continue
        tactic = m["tactic"]
        if tactic not in tactics:
            tactics[tactic] = []
        tactics[tactic].append((m["id"], m["name"], PLAYBOOKS[atype]["action"], PLAYBOOKS[atype]["human_gate"]))

    for tactic, techniques in tactics.items():
        with st.expander(f"🎯 {tactic}", expanded=True):
            for tid, tname, action, hg in techniques:
                gate = "⚠️ Human gate" if hg else "✅ Automated"
                st.markdown(f"""
<div style="background:#1a1a2e;border-radius:6px;padding:10px;margin:4px 0;border:1px solid #2a2a5e;">
  <span class="mitre-badge">{tid}</span> &nbsp; <b>{tname}</b><br>
  <span style="color:#aaa;font-size:13px">🔧 {action}</span><br>
  <span style="font-size:12px">{gate}</span>
</div>""", unsafe_allow_html=True)

# ── Tab 3: Orchestrator ────────────────────────────────────────────────────
with tab3:
    st.subheader("Incident Response Orchestrator")
    st.caption("Manual trigger — enter event values to get instant response")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        login_hour   = st.number_input("Login hour (0-23)", 0, 23, 14)
        session_dur  = st.number_input("Session duration (min)", 1, 1440, 60)
    with c2:
        bytes_mb     = st.number_input("Bytes transferred (MB)", 0.0, 10000.0, 10.0)
        failed_login = st.number_input("Failed logins", 0, 100, 0)
    with c3:
        endpoints    = st.number_input("Endpoints accessed", 1, 10, 3)
        sensitive    = st.number_input("Sensitive endpoints hit", 0, 10, 0)
    with c4:
        unique_ips   = st.number_input("Unique IPs", 1, 50, 1)
        commands     = st.number_input("Commands executed", 1, 1000, 30)

    if st.button("🔍 Analyze Event", type="primary"):
        row_dict = {
            "login_hour": login_hour, "session_duration_min": session_dur,
            "bytes_transferred_mb": bytes_mb, "failed_logins": failed_login,
            "endpoints_accessed": endpoints, "sensitive_endpoints_hit": sensitive,
            "unique_ips": unique_ips, "commands_executed": commands
        }
        row_df = pd.DataFrame([row_dict])
        error, is_anomaly = run_inference(model, scaler, meta, row_df)
        atype  = classify_anomaly(row_dict) if is_anomaly else "normal"
        mitre  = MITRE_MAP.get(atype, MITRE_MAP["unknown"])
        play   = PLAYBOOKS.get(atype, PLAYBOOKS["unknown"])
        severity = "HIGH" if error > meta["threshold"]*2 else "MEDIUM"

        st.divider()
        if is_anomaly:
            col_a, col_b = st.columns(2)
            with col_a:
                st.error(f"🚨 ANOMALY DETECTED — {severity}")
                st.metric("Reconstruction Error", f"{error:.5f}")
                st.metric("Threshold",             f"{meta['threshold']:.5f}")
            with col_b:
                st.markdown(f"**MITRE Technique**")
                st.markdown(f'<span class="mitre-badge">{mitre["id"]}</span> &nbsp; {mitre["name"]}', unsafe_allow_html=True)
                st.markdown(f"**Tactic:** {mitre['tactic']}")
                st.markdown(f"**Automated Action:** {play['action']}")
                if play["human_gate"]:
                    st.warning("⚠️ Human escalation required before automated action")
                else:
                    st.success("✅ Automated response triggered")
        else:
            st.success(f"✅ Normal behavior — Reconstruction error: {error:.5f} (below threshold {meta['threshold']:.5f})")

# ── Tab 4: Analytics ───────────────────────────────────────────────────────
with tab4:
    st.subheader("Dataset Analytics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Anomaly type distribution**")
        atype_counts = df[df.is_anomaly==1]["anomaly_type"].value_counts().reset_index()
        atype_counts.columns = ["type","count"]
        st.bar_chart(atype_counts.set_index("type"))

    with col2:
        st.markdown("**Bytes transferred: normal vs anomaly**")
        chart_df = pd.DataFrame({
            "Normal":  df[df.is_anomaly==0]["bytes_transferred_mb"].describe(),
            "Anomaly": df[df.is_anomaly==1]["bytes_transferred_mb"].describe()
        })
        st.dataframe(chart_df.round(2))

    st.markdown("**Model comparison**")
    perf = pd.DataFrame({
        "Model":    ["LSTM Autoencoder", "Isolation Forest"],
        "F1 Score": [round(meta["f1_lstm"],4), round(meta["f1_iso"],4)],
        "ROC-AUC":  [str(round(meta["roc_auc"],4)), "N/A"]
    })
    st.dataframe(perf, hide_index=True, use_container_width=True)

    st.markdown("**Raw log sample**")
    st.dataframe(df.sample(20)[["timestamp","user_id","department","login_hour",
                                 "bytes_transferred_mb","failed_logins",
                                 "unique_ips","is_anomaly","anomaly_type"]],
                 hide_index=True, use_container_width=True)
