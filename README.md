🧠 Anomaly Detection Engine for AI-Powered Payroll

📌 Overview

This project implements an Anomaly Detection Engine for payroll systems using unsupervised machine learning.
It is designed to automatically identify suspicious payroll activities such as:

Salary manipulation
Fake or inflated overtime claims
Since real payroll fraud data is rarely labeled, the system relies entirely on unsupervised learning techniques and statistical analysis to detect abnormal patterns.

🎯 Key Features

Unsupervised anomaly detection (no fraud labels required)
Detects salary manipulation & fake overtime
Real-time friendly (Isolation Forest)
Batch analysis using Deep Learning (Autoencoder)
Concept drift detection for changing payroll behavior
Human-readable anomaly alerts
Modular and production-ready design

🏗️ System Architecture
Payroll Data
     ↓
Feature Engineering & Scaling
     ↓
Isolation Forest (Fast Detection)
     ↓
Autoencoder (Deep Pattern Learning)
     ↓
Ensemble Decision
     ↓
Concept Drift Detection
     ↓
Alert Generation & Explanation

📂 Project Structure

.
├── anomaly_detection.py   # Main implementation
├── README.md              # Project documentation

🧪 Technologies Used

Python

NumPy, Pandas

Scikit-learn

PyTorch

SciPy

📊 Feature Engineering

The system derives behavioral features from raw payroll data:

Feature	Description
salary_growth	Ratio of current salary to previous salary
overtime_ratio	Overtime hours relative to regular hours
is_round_overtime	Detects suspicious rounded overtime values
salary	Absolute salary value
overtime_hours	Total overtime claimed

These features help the model learn normal payroll behavior and detect deviations.

🧠 Models Used

🔹 Isolation Forest
Primary anomaly detection model
Efficient for real-time payroll validation
Flags records that deviate strongly from normal patterns
🔹 Autoencoder (Neural Network)
Learns compressed representation of normal payroll data
High reconstruction error indicates anomalies
Used for deeper batch analysis
🔹 Ensemble Strategy
An employee record is marked anomalous if any model detects abnormal behavior, reducing false negatives.

🔄 Concept Drift Handling

Payroll patterns evolve due to:

Policy changes

Promotions

Seasonal overtime

Organizational growth

To handle this, the system:

Maintains sliding windows of historical and recent data

Uses Kolmogorov–Smirnov statistical tests

Flags distribution shifts automatically

This prevents the model from becoming outdated over time.

🚨 Alert Generation
Each anomaly generates a structured alert containing:
Employee ID
Timestamp
Anomaly type (salary_manipulation / fake_overtime)
Anomaly score
Severity level
Human-readable explanation

Example Alert
Employee: EMP0032
Type: fake_overtime
Severity: high
Explanation: Suspicious overtime pattern

▶️ How to Run
1️⃣ Install Dependencies
pip install numpy pandas scikit-learn torch scipy

2️⃣ Run the Program
python anomaly_detection.py


The script will:
Generate synthetic payroll data
Train anomaly detection models
Detect anomalies
Display sample alerts

📈 Sample Output

Anomalies detected: 40
Concept drift detected: False
AnomalyAlert(employee_id='EMP0003', anomaly_type='salary_manipulation', severity='high')

📌 Why Unsupervised Learning?

Fraud labels are rare or unavailable
Fraud patterns constantly change
Manual labeling is expensive and unreliable
Unsupervised learning allows the system to learn normal behavior first and flag deviations automatically.
