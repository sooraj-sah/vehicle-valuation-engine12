import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Vehicle Resale Value Prediction Engine",
    description="Real-time ML valuation with SHAP feature attribution"
)

# Model aur Preprocessor load karein
preprocessor = joblib.load('preprocessor.joblib')
model = joblib.load('xgb_model.joblib')
explainer = shap.TreeExplainer(model)

class CarSpecs(BaseModel):
    brand: str = Field(..., example = "Toyota")
    year: int = Field( ge=2000, le=2026, example=2021)
    mileage: float = Field(..., ge=0, example=38000)
    engine_cc: int = Field(..., example=1500)
    transmission: str = Field(..., example="Automatic")
    fuel: str = Field(..., example="Petrol")

@app.post("/predict")
def predict_valuation(specs: CarSpecs):
    vehicle_age = 2026 - specs.year
    mileage_per_year = specs.mileage / (vehicle_age + 1)

    df_input = pd.DataFrame([{
        'brand': specs.brand,
        'fuel': specs.fuel,
        'transmission': specs.transmission,
        'engine_cc': specs.engine_cc,
        'vehicle_age': vehicle_age,
        'mileage_per_year': mileage_per_year
    }])

    # Preprocessing aur Prediction
    processed_x = preprocessor.transform(df_input)
    log_price = model.predict(processed_x)[0]
    estimated_price = float(np.expm1(log_price))

    # SHAP local feature attribution
    shap_vals = explainer.shap_values(processed_x)[0]
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['brand', 'fuel', 'transmission'])
    feature_names = ['engine_cc', 'vehicle_age', 'mileage_per_year'] + list(cat_names)

    top_drivers = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "predicted_resale_price_inr": round(estimated_price, 2),
        "top_market_drivers": [
            {"factor": factor, "impact_score": round(float(impact), 4)} for factor, impact in top_drivers
        ]
    }
@app.get("/")
def home():
    return {"message": "Vehicle Valuation Engine is running"}