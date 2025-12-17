import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Paths
DATA_PATH = "data/processed/fe_dataset_with_allotment.csv"
MODEL_PATH = "models/allotment_model.joblib"

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# ---------------- TARGET ----------------
target = "Allotment_Probability"

# ---------------- FEATURES ----------------
features = [
    "Retail Subscription",
    "Subscription QIB",
    "Subscription NII",
    "log_Issue_Size",
    "Price_Band_Width",
    "Sub_Strength"
]

# Keep only available columns
features = [f for f in features if f in df.columns]
print("Using features:", features)

X = df[features].fillna(0)
y = df[target]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ---------------- EVALUATION ----------------
preds = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nAllotment Probability Model Evaluation")
print("MAE :", round(mae, 4))
print("R²  :", round(r2, 4))

# ---------------- SAVE ----------------
joblib.dump(
    {"model": model, "scaler": scaler, "features": features},
    MODEL_PATH
)

print("\nSaved Allotment Probability model to:", MODEL_PATH)
