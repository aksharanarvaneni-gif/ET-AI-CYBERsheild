import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

USERS = [f"user_{i:03d}" for i in range(1, 51)]
ENDPOINTS = [
    "/api/login", "/api/data/export", "/api/admin/users",
    "/api/files/download", "/api/reports", "/api/settings",
    "/api/dashboard", "/api/logs", "/api/backup", "/api/credentials"
]
SENSITIVE = {"/api/admin/users", "/api/credentials", "/api/backup", "/api/data/export"}
DEPARTMENTS = ["IT", "Finance", "HR", "Operations", "Engineering"]

USER_DEPT = {u: random.choice(DEPARTMENTS) for u in USERS}
USER_WORK_HOURS = {u: (random.randint(7, 10), random.randint(16, 19)) for u in USERS}

def normal_record(user, timestamp):
    start_h, end_h = USER_WORK_HOURS[user]
    hour = random.randint(start_h, end_h)
    minute = random.randint(0, 59)
    ts = timestamp.replace(hour=hour, minute=minute)
    n_endpoints = random.randint(1, 5)
    accessed = random.sample(ENDPOINTS[:7], n_endpoints)
    sensitive_hits = sum(1 for e in accessed if e in SENSITIVE)
    return {
        "timestamp": ts,
        "user_id": user,
        "department": USER_DEPT[user],
        "login_hour": hour,
        "session_duration_min": random.randint(15, 480),
        "bytes_transferred_mb": round(random.uniform(0.1, 50.0), 2),
        "failed_logins": random.choices([0, 1, 2], weights=[85, 12, 3])[0],
        "endpoints_accessed": n_endpoints,
        "sensitive_endpoints_hit": sensitive_hits,
        "unique_ips": 1,
        "commands_executed": random.randint(5, 80),
        "is_anomaly": 0,
        "anomaly_type": "normal"
    }

def anomaly_record(user, timestamp, anomaly_type):
    base = normal_record(user, timestamp)

    if anomaly_type == "brute_force":
        # T1110 - Brute Force
        base["failed_logins"] = random.randint(15, 50)
        base["login_hour"] = random.choice([1, 2, 3, 4, 22, 23])
        base["session_duration_min"] = random.randint(1, 10)
        base["bytes_transferred_mb"] = round(random.uniform(0.01, 2.0), 2)

    elif anomaly_type == "data_exfiltration":
        # T1041 - Exfiltration Over C2 Channel
        base["bytes_transferred_mb"] = round(random.uniform(500, 5000), 2)
        base["sensitive_endpoints_hit"] = random.randint(4, 8)
        base["endpoints_accessed"] = random.randint(6, 10)
        base["session_duration_min"] = random.randint(120, 600)

    elif anomaly_type == "credential_abuse":
        # T1078 - Valid Accounts (off-hours access)
        base["login_hour"] = random.choice([0, 1, 2, 3, 4, 23])
        base["unique_ips"] = random.randint(3, 10)
        base["sensitive_endpoints_hit"] = random.randint(3, 6)
        base["commands_executed"] = random.randint(150, 400)

    elif anomaly_type == "lateral_movement":
        # T1021 - Remote Services
        base["unique_ips"] = random.randint(8, 25)
        base["endpoints_accessed"] = random.randint(8, 10)
        base["commands_executed"] = random.randint(200, 500)
        base["session_duration_min"] = random.randint(300, 720)

    elif anomaly_type == "privilege_escalation":
        # T1068 - Exploitation for Privilege Escalation
        base["sensitive_endpoints_hit"] = random.randint(5, 8)
        base["commands_executed"] = random.randint(300, 600)
        base["bytes_transferred_mb"] = round(random.uniform(100, 800), 2)
        base["failed_logins"] = random.randint(3, 8)

    base["is_anomaly"] = 1
    base["anomaly_type"] = anomaly_type
    return base

def generate_dataset(n_normal=4000, n_anomaly=400):
    records = []
    start_date = datetime(2026, 1, 1)

    # Normal records
    for _ in range(n_normal):
        user = random.choice(USERS)
        day_offset = random.randint(0, 179)
        ts = start_date + timedelta(days=day_offset)
        records.append(normal_record(user, ts))

    # Anomaly records
    anomaly_types = [
        "brute_force",
        "data_exfiltration",
        "credential_abuse",
        "lateral_movement",
        "privilege_escalation"
    ]
    per_type = n_anomaly // len(anomaly_types)
    for atype in anomaly_types:
        for _ in range(per_type):
            user = random.choice(USERS)
            day_offset = random.randint(0, 179)
            ts = start_date + timedelta(days=day_offset)
            records.append(anomaly_record(user, ts, atype))

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Generating synthetic network logs...")
    df = generate_dataset(n_normal=4000, n_anomaly=400)

    df.to_csv("data/network_logs.csv", index=False)

    print(f"\nDataset generated: {len(df)} records")
    print(f"Normal records   : {(df.is_anomaly == 0).sum()}")
    print(f"Anomaly records  : {(df.is_anomaly == 1).sum()}")
    print(f"\nAnomaly breakdown:")
    print(df[df.is_anomaly == 1]["anomaly_type"].value_counts().to_string())
    print(f"\nFeature columns:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print(f"\nSample normal record:")
    print(df[df.is_anomaly == 0].iloc[0].to_dict())
    print(f"\nSample anomaly record (brute_force):")
    print(df[df.anomaly_type == "brute_force"].iloc[0].to_dict())
    print("\nSaved to network_logs.csv")
