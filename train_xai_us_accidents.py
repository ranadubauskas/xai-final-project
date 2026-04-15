from __future__ import annotations

import os
import json
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ----------------------------
# Config
# ----------------------------

DATA_PATH = "data/US_Accidents_March23.csv"
ARTIFACT_DIR = "artifacts"
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000   # lower for debugging / laptop memory
TEST_SIZE = 0.2

TARGET_COL = "high_severity"   # derived from Severity >= 3


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

    df["year"] = df["Start_Time"].dt.year
    df["month"] = df["Start_Time"].dt.month
    df["day"] = df["Start_Time"].dt.day
    df["hour"] = df["Start_Time"].dt.hour
    df["dayofweek"] = df["Start_Time"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # cyclical encoding for hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = (df["Severity"] >= 3).astype(int)
    return df


def select_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Split features into:
    - numeric features
    - text categorical features
    - boolean indicator features

    This matches the actual US Accidents CSV structure better.
    """
    numeric_features = [
        "Start_Lat",
        "Start_Lng",
        "Distance(mi)",
        "Temperature(F)",
        "Wind_Chill(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Precipitation(in)",
        "hour_sin",
        "hour_cos",
        "month",
        "dayofweek",
        "is_weekend",
    ]

    categorical_features = [
        "State",
        "Timezone",
        "Weather_Condition",
        "Wind_Direction",
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
    ]

    boolean_features = [
        "Junction",
        "Traffic_Signal",
        "Crossing",
        "Stop",
        "Give_Way",
        "Railway",
        "Roundabout",
        "Amenity",
        "Bump",
        "Station",
    ]

    available_numeric = [c for c in numeric_features if c in df.columns]
    available_categorical = [c for c in categorical_features if c in df.columns]
    available_boolean = [c for c in boolean_features if c in df.columns]

    keep_cols = (
        available_numeric
        + available_categorical
        + available_boolean
        + [TARGET_COL]
    )
    df = df[keep_cols].copy()

    return df, available_numeric, available_categorical, available_boolean


def subsample_df(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=RANDOM_STATE)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # keep bin ids in [0, n_bins-1]
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) == 0:
            continue
        avg_conf = np.mean(y_prob[mask])
        avg_acc = np.mean(y_true[mask])
        ece += (np.sum(mask) / n) * abs(avg_acc - avg_conf)

    return float(ece)


def plot_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: str
) -> None:
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ensure_dir(ARTIFACT_DIR)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print(f"Original shape: {df.shape}")

    # Basic filtering
    df = df.dropna(subset=["Severity", "Start_Time"])
    df = add_time_features(df)
    df = build_target(df)

    # Keep only rows with valid target
    df = df[df[TARGET_COL].isin([0, 1])]

    # Optional: narrow to junction-related incidents
    # if "Junction" in df.columns:
    #     df = df[df["Junction"] == True]

    df, numeric_features, categorical_features, boolean_features = select_columns(df)

    feature_cols = numeric_features + categorical_features + boolean_features

    # Drop rows that are entirely missing for selected predictors
    df = df.dropna(subset=feature_cols, how="all")

    # Subsample for faster iteration
    df = subsample_df(df, SAMPLE_SIZE)
    print(f"Working shape after sampling: {df.shape}")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Force consistent dtypes
    for col in categorical_features:
        X[col] = X[col].fillna("missing").astype(str)

    for col in boolean_features:
        # sample file shows these as True/False flags
        X[col] = X[col].fillna(False).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Positive rate train: {y_train.mean():.4f}")
    print(f"Positive rate test:  {y_test.mean():.4f}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    boolean_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bool", boolean_transformer, boolean_features),
        ]
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training model...")
    clf.fit(X_train, y_train)

    print("Predicting...")
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "ece": float(compute_ece(y_test.to_numpy(), y_prob)),
    }

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    save_json(metrics, os.path.join(ARTIFACT_DIR, "metrics.json"))
    plot_calibration_curve(
        y_test.to_numpy(),
        y_prob,
        os.path.join(ARTIFACT_DIR, "reliability_diagram.png"),
    )

    # Save predictions for report tables / error analysis
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(os.path.join(ARTIFACT_DIR, "test_predictions.csv"), index=False)

    # Save model
    joblib.dump(clf, os.path.join(ARTIFACT_DIR, "xgb_pipeline.joblib"))

    # ----------------------------
    # SHAP
    # ----------------------------
    print("Preparing SHAP explanations...")

    shap_sample = min(5000, len(X_test))
    X_shap = X_test.sample(n=shap_sample, random_state=RANDOM_STATE)

    preprocessor_fitted = clf.named_steps["preprocessor"]
    model_fitted = clf.named_steps["model"]

    X_shap_transformed = preprocessor_fitted.transform(X_shap)

    feature_names = preprocessor_fitted.get_feature_names_out()

    explainer = shap.TreeExplainer(model_fitted)
    shap_values = explainer.shap_values(X_shap_transformed)

    # Handle possible list output from older/newer SHAP behavior
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(ARTIFACT_DIR, "shap_summary.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(
        os.path.join(ARTIFACT_DIR, "shap_feature_importance.csv"),
        index=False,
    )

    top_cases_idx = np.argsort(-y_prob)[:3]
    local_cases = pred_df.iloc[top_cases_idx].copy()
    local_cases.to_csv(os.path.join(ARTIFACT_DIR, "top_risk_cases.csv"), index=False)

    print("\nDone. Artifacts saved in:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()