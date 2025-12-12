# scripts/train_xgb_improved.py
"""
Improved XGBoost training pipeline (3-class) with:
 - safe feature selection + derived features
 - SMOTE+Tomek resampling (only on training data)
 - stratified CV (f1_macro) evaluation
 - sensible/hard hyperparameters (from prior tuning)
 - final model save
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib
import math

# ---------- config ----------
proj = Path(__file__).resolve().parent.parent
DATA = proj / "data" / "processed" / "fe_dataset_labeled1.csv"   # change if needed
OUT_MODEL = proj / "models" / "xgb_improved.joblib"
OUT_LE = proj / "models" / "label_encoder_improved.joblib"
RND = 42

# ---------- load ----------
print("Loading:", DATA)
df = pd.read_csv(DATA, low_memory=False)
print("Rows:", len(df))

# ---------- detect target ----------
target = None
for cand in ["Recommendation", "RecommendationLabel", "Label"]:
    if cand in df.columns:
        target = cand
        break
if target is None:
    raise SystemExit("No target column found (Recommendation/Label).")

print("Using target column:", target)
print(df[target].value_counts())

# ---------- select safe base features (adjust to your columns) ----------
# Prefer columns you created during FE; update this list if your column names differ
preferred = [
    "Issue Price (Rs)", "IPO_Duration", "Price_Band_Width",
    "log_Issue_Size", "Sub_Strength", "upper_price", "lower_price",
    "no_of_shares_cr"
]

features = [c for c in preferred if c in df.columns]
print("Found base features:", features)

# ---------- derived features (create if source columns available) ----------
# price band ratio
if ("upper_price" in df.columns) and ("lower_price" in df.columns):
    df["Price_Band_Ratio"] = (pd.to_numeric(df["upper_price"], errors="coerce").fillna(0) -
                               pd.to_numeric(df["lower_price"], errors="coerce").fillna(0)) / \
                              (pd.to_numeric(df["lower_price"], errors="coerce").replace(0, np.nan)).fillna(1)
    features.append("Price_Band_Ratio")

# normalized GMP if any gmp columns exist
gmp_candidates = ["GMP", "GMP_day_before", "gmp"]
gmp_col = None
for g in gmp_candidates:
    if g in df.columns:
        gmp_col = g
        break
if gmp_col:
    # normalized by price to remove scale effects (safe)
    if "upper_price" in df.columns:
        df["GMP_Score"] = pd.to_numeric(df[gmp_col], errors="coerce").fillna(0) / \
                          (pd.to_numeric(df["upper_price"], errors="coerce").replace(0, np.nan).fillna(1))
    else:
        df["GMP_Score"] = pd.to_numeric(df[gmp_col], errors="coerce").fillna(0)
    features.append("GMP_Score")

# conservative issue size log already present (log_Issue_Size) else create from issue size
if "log_Issue_Size" not in features:
    if "Issue Size (Rs Cr)" in df.columns or "Total Issue Size (Rs Cr)" in df.columns:
        col_name = "Issue Size (Rs Cr)" if "Issue Size (Rs Cr)" in df.columns else "Total Issue Size (Rs Cr)"
        df["log_Issue_Size"] = np.log1p(pd.to_numeric(df[col_name], errors="coerce").fillna(0))
        features.append("log_Issue_Size")

# ensure all feature columns exist
for f in features:
    if f not in df.columns:
        df[f] = 0

# remove any obvious post-listing leakage columns if accidentally included
leakage_keywords = ["Listing", "Basis", "Allotment", "Listing Gain", "Listing_Price", "basis_of_allotment"]
features = [f for f in features if not any(k.lower() in f.lower() for k in leakage_keywords)]

print("Final feature list:", features)

# ---------- prepare X,y ----------
X = df[features].astype(float).fillna(0)
y_raw = df[target].astype(str).fillna("Neutral")
le = LabelEncoder(); y = le.fit_transform(y_raw)
print("Classes:", le.classes_, "Counts:", dict(zip(le.classes_, np.bincount(y))))

# ---------- train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RND, stratify=y)
print("Train/test sizes:", X_train.shape, X_test.shape)

# ---------- compute class weights (on original training) ----------
cls = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=cls, y=y_train)
class_weight_map = {c: w for c, w in zip(cls, cw)}
print("Class weights:", class_weight_map)

# ---------- resampling: SMOTE + Tomek (only on training set) ----------
smote_tomek = SMOTETomek(random_state=RND)
X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
print("After resampling train counts:", dict(zip(*np.unique(y_res, return_counts=True))))

# ---------- pipeline: scaler + xgb ----------
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
X_test_scaled = scaler.transform(X_test)

# sensible XGBoost hyperparameters (from tuning)
xgb = XGBClassifier(
    n_estimators=330,
    max_depth=6,
    learning_rate=0.107,
    subsample=0.9627,
    colsample_bytree=0.70825,
    min_child_weight=9,
    gamma=2.65,
    objective="multi:softprob",
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=RND,
    verbosity=0
)

# ---------- optionally apply sample weights (based on original class weights) ----------
# create a sample_weight vector for the resampled training set using original class weights
sample_weight = np.array([class_weight_map.get(label, 1.0) for label in y_res])

# ---------- cross-validate with stratified folds (evaluate) ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
scores = []
for train_idx, val_idx in cv.split(X_res_scaled, y_res):
    Xtr, Xval = X_res_scaled[train_idx], X_res_scaled[val_idx]
    ytr, yval = y_res[train_idx], y_res[val_idx]
    # fit on fold (we pass sample_weight for this fold)
    sw = sample_weight[train_idx]
    xgb.fit(Xtr, ytr, sample_weight=sw)
    preds = xgb.predict(Xval)
    scores.append(f1_score(yval, preds, average="macro"))
print("CV f1_macro on resampled data (in-fold): mean=%.4f std=%.4f" % (np.mean(scores), np.std(scores)))

# ---------- final fit on full resampled training set ----------
xgb.fit(X_res_scaled, y_res, sample_weight=sample_weight)

# ---------- evaluate on real untouched test set ----------
pred_test = xgb.predict(X_test_scaled)
print("\nClassification report (real hold-out test):")
print(classification_report(y_test, pred_test, target_names=le.classes_))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_test))
acc = (pred_test == y_test).mean()
print("Test accuracy: %.4f" % acc)

# ---------- save model, scaler and encoder ----------
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": xgb, "scaler": scaler, "features": features}, OUT_MODEL)
joblib.dump(le, OUT_LE)
print("Saved model to:", OUT_MODEL)
print("Saved label encoder to:", OUT_LE)
