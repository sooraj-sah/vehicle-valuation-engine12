import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def train_model():
    print("[1/4] Preparing dataset...")
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        'brand': np.random.choice(['Maruti', 'Hyundai', 'Toyota', 'Honda', 'BMW'], n),
        'year': np.random.randint(2012, 2025, n),
        'mileage': np.random.uniform(10000, 150000, n),
        'engine_cc': np.random.choice([1000, 1200, 1500, 2000, 2500], n),
        'transmission': np.random.choice(['Manual', 'Automatic'], n),
        'fuel': np.random.choice(['Petrol', 'Diesel', 'CNG'], n),
        'price': np.random.uniform(250000, 3000000, n)
    })

    # Domain Feature Engineering
    current_year = 2026
    df['vehicle_age'] = current_year - df['year']
    df['mileage_per_year'] = df['mileage'] / (df['vehicle_age'] + 1)

    X = df[['brand', 'fuel', 'transmission', 'engine_cc', 'vehicle_age', 'mileage_per_year']]
    # Target distribution skewness handle karne ke liye Log1p
    y = np.log1p(df['price'])

    cat_cols = ['brand', 'fuel', 'transmission']
    num_cols = ['engine_cc', 'vehicle_age', 'mileage_per_year']

    # Preprocessor Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )

    print("[2/4] Splitting and fitting pipeline...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print("[3/4] Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_proc, y_train)

    # Performance Evaluation
    preds_log = model.predict(X_test_proc)
    preds = np.expm1(preds_log)
    y_test_orig = np.expm1(y_test)

    print("\n--- Model Benchmark ---")
    print(f"R² Score : {r2_score(y_test_orig, preds):.4f}")
    print(f"RMSE     : ₹{np.sqrt(mean_squared_error(y_test_orig, preds)):,.2f}")
    print(f"MAE      : ₹{mean_absolute_error(y_test_orig, preds):,.2f}")

    print("\n[4/4] Saving artifacts...")
    joblib.dump(preprocessor, 'preprocessor.joblib')
    joblib.dump(model, 'xgb_model.joblib')
    print("Files 'preprocessor.joblib' & 'xgb_model.joblib' created in project root.")

if __name__ == '__main__':
    train_model()