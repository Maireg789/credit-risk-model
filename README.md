# Credit Risk Probability Model for Alternative Data

## Project Overview
This project focuses on building a Credit Scoring Model for Bati Bank's partnership with an eCommerce platform. The goal is to enable a buy-now-pay-later service by assessing the creditworthiness of customers based on their transaction history.

## Credit Scoring Business Understanding (Task 1)

### 1. Basel II Accord and Model Interpretability
The Basel II Capital Accord focuses on rigorous risk measurement to determine the minimum capital a bank must hold. It requires banks to have a robust internal rating system to calculate Probability of Default (PD). 
*   **Influence on our model:** Because these models impact capital allocation and regulatory compliance, they cannot be "black boxes." We prioritize models that are interpretable, transparent, and well-documented. We must be able to explain *why* a specific customer was classified as high-risk to regulators and internal auditors.

### 2. The Need for a Proxy Variable
*   **Why a proxy?** In this alternative data context, we do not have a historical label indicating who "defaulted" on a previous loan because these are eCommerce transactions, not loan repayment histories.
*   **The Approach:** We use RFM (Recency, Frequency, Monetary) analysis to classify users. High-engagement users are proxies for "Good" credit risk, while low-engagement/dormant users are proxies for "High" risk.
*   **Business Risks:** The primary risk is **misclassification**. A customer might be low-engagement simply because they don't shop often, not because they are financially unstable. Predicting based on this proxy might lead to rejecting viable customers (Type II error) or lending to fraudsters who manipulate transaction frequency (Type I error).

### 3. Logistic Regression (WoE) vs. Gradient Boosting
*   **Logistic Regression with WoE:** 
    *   *Pros:* Highly interpretable. Each feature's contribution (Scorecard points) is visible. Standard in traditional banking.
    *   *Cons:* May miss complex, non-linear patterns in behavioral data.
*   **Gradient Boosting (e.g., XGBoost):**
    *   *Pros:* High predictive performance; captures complex non-linear relationships.
    *   *Cons:* "Black box" nature makes it harder to explain to regulators.
*   **Trade-off:** In a regulated financial context, we often sacrifice a small amount of accuracy (Boosting) for the safety and transparency of interpretability (Logistic Regression), though techniques like SHAP values are closing this gap.