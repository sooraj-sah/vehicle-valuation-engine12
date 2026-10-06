# 🚗 Vehicle Valuation & Resale Price Engine with Explainable AI (XAI)

An end-to-end Machine Learning regression system that predicts vehicle resale market values using XGBoost, features zero-leakage Scikit-Learn pipelines, provides model interpretability via SHAP, and serves low-latency predictions through FastAPI.

---

## 📌 Project Overview
Estimating vehicle depreciation accurately is challenging due to non-linear interactions between brand equity, mileage patterns, and mechanical specifications. 

This project solves this problem by:
- Normalizing heavily skewed pricing distributions using `Log1p` transformations.
- Preventing data leakage via Scikit-Learn `ColumnTransformer`.
- Explaining the "Why" behind predictions using SHAP (Shapley Additive exPlanations).
- Delivering sub-50ms inference times via FastAPI REST endpoints.

---

## 🛠️ Tech Stack & Tools
- **Language & IDE:** Python 3.10+, JetBrains PyCharm
- **Machine Learning:** Scikit-Learn, XGBoost
- **Explainable AI:** SHAP (TreeExplainer)
- **API & Serving:** FastAPI, Uvicorn, Pydantic
- **Model Serialization:** Joblib

---

## 📊 Model Evaluation & Benchmarks

| Metric | Baseline (Ridge Regression) | XGBoost (Final Tuned) |
| :--- | :--- | :--- |
| **$R^2$ Score** | 0.7410 | **0.9124** |
| **RMSE** | ₹1,42,300 | **₹84,500** |
| **MAE** | ₹98,400 | **₹56,200** |

---

## ⚙️ Project Setup & Installation (PyCharm)

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/vehicle-valuation-engine.git](https://github.com/your-username/vehicle-valuation-engine.git)
cd vehicle-valuation-engine
```

### 2. Set Up Virtual Environment & Dependencies
PyCharm Terminal open karein aur execute karein:
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Model & Generate Artifacts
```bash
python train.py
```
*Output: Saves `preprocessor.joblib` and `xgb_model.joblib` to the root directory.*

### 4. Start the FastAPI Server
```bash
uvicorn app:app --reload
```
API runs locally at: `http://127.0.0.1:8000`  
Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

---

## 🔌 API Endpoint Usage

### Sample Request (`POST /predict`)
```json
{
  "brand": "Toyota",
  "year": 2021,
  "mileage": 38000,
  "engine_cc": 1500,
  "transmission": "Automatic",
  "fuel": "Petrol"
}
```

### Sample Response with SHAP Explanations
```json
{
  "predicted_resale_price_inr": 845200.0,
  "top_market_drivers": [
    {
      "factor": "vehicle_age",
      "impact_score": -0.412
    },
    {
      "factor": "transmission_Automatic",
      "impact_score": 0.285
    },
    {
      "factor": "mileage_per_year",
      "impact_score": -0.194
    }
  ]
}
```
