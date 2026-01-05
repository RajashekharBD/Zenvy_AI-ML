"""
Anomaly Detection Engine for AI-Powered Payroll
Detects salary manipulation and fake overtime using unsupervised learning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import torch
import torch.nn as nn
from dataclasses import dataclass
from datetime import datetime
from collections import deque

# ============================================================
# DATA PREPROCESSING
# ============================================================

class PayrollPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()

        # Feature engineering
        df["salary_growth"] = df["salary"] / df["previous_salary"].replace(0, 1)
        df["overtime_ratio"] = df["overtime_hours"] / df["regular_hours"].replace(0, 1)
        df["is_round_overtime"] = (df["overtime_hours"] % 1 == 0).astype(int)

        features = [
            "salary",
            "salary_growth",
            "overtime_hours",
            "overtime_ratio",
            "is_round_overtime"
        ]

        X = df[features].fillna(0)

        if not self.fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, df

# ============================================================
# AUTOENCODER MODEL
# ============================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ============================================================
# CONCEPT DRIFT DETECTOR
# ============================================================

class ConceptDriftDetector:
    def __init__(self, window_size=200, threshold=0.1):
        self.ref_window = deque(maxlen=window_size)
        self.cur_window = deque(maxlen=window_size)
        self.threshold = threshold

    def initialize(self, X):
        for row in X:
            self.ref_window.append(row)

    def update(self, X):
        for row in X:
            self.cur_window.append(row)

        if len(self.cur_window) < len(self.ref_window) // 2:
            return False

        drift_score = 0
        for i in range(X.shape[1]):
            ks_stat, _ = stats.ks_2samp(
                np.array(self.ref_window)[:, i],
                np.array(self.cur_window)[:, i]
            )
            drift_score += ks_stat

        drift_score /= X.shape[1]
        return drift_score > self.threshold

# ============================================================
# ALERT STRUCTURE
# ============================================================

@dataclass
class AnomalyAlert:
    employee_id: str
    timestamp: datetime
    anomaly_type: str
    score: float
    explanation: str
    severity: str

# ============================================================
# MAIN ENGINE
# ============================================================

class PayrollAnomalyEngine:
    def __init__(self):
        self.preprocessor = PayrollPreprocessor()

        self.iforest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

        self.autoencoder = None
        self.threshold = None
        self.drift_detector = ConceptDriftDetector()
        self.trained = False

    def train(self, df: pd.DataFrame):
        X, _ = self.preprocessor.preprocess(df)

        # Isolation Forest
        self.iforest.fit(X)

        # Autoencoder
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.autoencoder = Autoencoder(X.shape[1])
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for _ in range(50):
            optimizer.zero_grad()
            recon = self.autoencoder(X_tensor)
            loss = loss_fn(recon, X_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            errors = torch.mean((recon - X_tensor) ** 2, dim=1)
            self.threshold = np.percentile(errors.numpy(), 95)

        self.drift_detector.initialize(X)
        self.trained = True

    def detect(self, df: pd.DataFrame):
        if not self.trained:
            raise RuntimeError("Model not trained")

        X, df_feat = self.preprocessor.preprocess(df)

        iso_pred = self.iforest.predict(X)
        iso_scores = self.iforest.decision_function(X)

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            recon = self.autoencoder(X_tensor)
            errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()

        anomalies = (iso_pred == -1) | (errors > self.threshold)
        drift = self.drift_detector.update(X)

        alerts = []
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                row = df_feat.iloc[i]
                anomaly_type = (
                    "salary_manipulation"
                    if row["salary_growth"] > 1.3
                    else "fake_overtime"
                )

                explanation = (
                    "Unusual salary increase"
                    if anomaly_type == "salary_manipulation"
                    else "Suspicious overtime pattern"
                )

                alerts.append(
                    AnomalyAlert(
                        employee_id=row["employee_id"],
                        timestamp=datetime.now(),
                        anomaly_type=anomaly_type,
                        score=float(abs(iso_scores[i])),
                        explanation=explanation,
                        severity="high"
                    )
                )

        return {
            "anomalies_detected": len(alerts),
            "concept_drift": drift,
            "alerts": alerts
        }

# ============================================================
# SAMPLE DATA + DEMO
# ============================================================

def generate_sample_data(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(n)],
        "salary": np.random.normal(50000, 12000, n),
        "previous_salary": np.random.normal(48000, 11000, n),
        "regular_hours": np.random.normal(40, 5, n),
        "overtime_hours": np.random.exponential(4, n)
    })

    # Inject anomalies
    df.loc[:20, "salary"] *= 1.6
    df.loc[21:40, "overtime_hours"] = 10

    return df

if __name__ == "__main__":
    data = generate_sample_data()

    engine = PayrollAnomalyEngine()
    engine.train(data)

    results = engine.detect(data)
    print("Anomalies detected:", results["anomalies_detected"])
    print("Concept drift detected:", results["concept_drift"])

    for alert in results["alerts"][:3]:
        print(alert)
