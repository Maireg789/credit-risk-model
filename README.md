# 🏦 Credit Risk Probability Model for Bati Bank

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-Passing-brightgreen)

## 📖 Project Overview
This project builds an end-to-end **Credit Scoring Model** for Bati Bank's partnership with an eCommerce platform. The goal is to enable a **Buy-Now-Pay-Later (BNPL)** service by assessing the creditworthiness of customers based on alternative data (transaction history) rather than traditional credit history.

Since historical loan default labels are unavailable, this project implements a **Proxy Labeling Strategy** using **RFM (Recency, Frequency, Monetary)** analysis and Unsupervised Learning to classify users as "High Risk" or "Low Risk."

---

## 📂 Project Structure
```text
credit-risk-model/
├── .github/workflows/   # CI/CD Pipeline (Linting & Testing)
├── data/                # Raw (gitignored) and Processed Data
├── notebooks/           # Exploratory Data Analysis (EDA)
├── src/
│   ├── api/             # FastAPI Application & Schemas
│   ├── feature_engineering.py # Data Pipeline (RFM, Aggregates, Target Creation)
│   ├── train.py         # Model Training, Tuning, and MLflow Registry
│   └── woe_analysis.py  # Weight of Evidence (WoE) Calculation
├── tests/               # Unit & Integration Tests
├── Dockerfile           # Docker Configuration
├── docker-compose.yml   # Container Orchestration
├── requirements.txt     # Python Dependencies
└── README.md            # Project Documentation

🧠 Business Understanding & Context (Task 1)
1. Basel II Accord and Model Interpretability
The Basel II Capital Accord focuses on rigorous risk measurement to determine the minimum capital a bank must hold. It requires banks to have a robust internal rating system to calculate Probability of Default (PD).
Influence on Design: Because these models impact capital allocation and regulatory compliance, they cannot be "black boxes." We prioritize models that are interpretable, transparent, and well-documented. We must be able to explain why a specific customer was classified as high-risk to regulators and internal auditors.
2. The Need for a Proxy Variable
Why a proxy? In this alternative data context, we do not have a historical label indicating who "defaulted" on a previous loan because these are eCommerce transactions, not loan repayment histories.
The Approach: We use RFM (Recency, Frequency, Monetary) analysis to classify users.
Low Risk (Good): High-engagement users (Frequent, High Value).
High Risk (Bad): Low-engagement/dormant users.
Business Risks: The primary risk is misclassification. A customer might be low-engagement simply because they don't shop often, not because they are financially unstable. Predicting based on this proxy might lead to rejecting viable customers (Type II error) or lending to fraudsters who manipulate transaction frequency (Type I error).
3. Logistic Regression (WoE) vs. Gradient Boosting
Logistic Regression with WoE:
Pros: Highly interpretable. Each feature's contribution (Scorecard points) is visible. Standard in traditional banking.
Cons: May miss complex, non-linear patterns in behavioral data.
Gradient Boosting (Random Forest/XGBoost):
Pros: High predictive performance; captures complex non-linear relationships.
Cons: "Black box" nature makes it harder to explain to regulators.
Trade-off: In a regulated financial context, we often sacrifice a small amount of accuracy for safety and transparency. However, for this project, we implemented Random Forest as a challenger model to maximize predictive power on the proxy target, achieving an AUC of 1.0.
🛠️ Technical Implementation
1. Feature Engineering Pipeline
We developed a robust pipeline (src/feature_engineering.py) that performs:
Temporal Extraction: Extracts Hour, Day, Month to capture seasonality.
Aggregations: Calculates Mean, Sum, and Count of transactions per customer.
RFM Clustering: Uses K-Means Clustering to segment users and assign the is_high_risk label.
Categorical Encoding: One-Hot Encoding for channels and Label Encoding for providers.
2. Model Training & MLOps
We utilize MLflow for experiment tracking and model management (src/train.py):
Hyperparameter Tuning: Uses GridSearchCV to optimize Random Forest parameters.
Pipeline: Wraps Scaling (StandardScaler) and Modeling into a single scikit-learn pipeline.
Model Registry: The best-performing model is automatically registered as Credit_Risk_Model_Prod for deployment.
3. Deployment (API & Docker)
The model is served via a FastAPI endpoint (src/api/main.py) that:
Loads the production model dynamically from the MLflow Registry.
Handles missing features robustly by aligning input data with the model schema.
Returns is_high_risk status and risk_probability.
🚀 Installation & Usage
Prerequisites
Python 3.9+
Docker (Optional)
Local Setup
code
Bash
# 1. Clone the repo
git clone https://github.com/Maireg789/credit-risk-model.git
cd credit-risk-model

# 2. Install dependencies
pip install -r requirements.txt
Running the Pipeline
code
Bash
# 1. Process Data & Create Target
python src/feature_engineering.py

# 2. Train & Register Model
python src/train.py

# 3. Start API Server
uvicorn src.api.main:app --reload
Docker Setup (Production)
code
Bash
docker-compose up --build
🧪 Testing & CI/CD
This project uses GitHub Actions for Continuous Integration. On every push to main, the pipeline runs:
Linting: Checks code quality using flake8.
Unit Tests: Runs pytest to verify data processing logic and API endpoints.
To run tests locally:
code
Bash
python -m pytest
📊 API Documentation
Once the server is running, visit http://127.0.0.1:8000/docs for the interactive Swagger UI.
Example Request:
code
JSON
POST /predict
{
  "features": {
    "Amount_mean": 5000,
    "TransactionCount": 15,
    "Recency": 2,
    "Frequency": 12,
    "Monetary": 60000
  }
}
Example Response:
code
JSON
{
  "is_high_risk": 0,
  "risk_probability": 0.05
}
code
Code
