"""
train_model.py
--------------
Handles: data loading, preprocessing, training,
evaluation, and saving the logistic regression model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "dataset", "water_potability.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "water_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
PLOT_DIR   = os.path.join(BASE_DIR, "model", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ─── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.isnull().sum())
    return df


# ─── 2. Preprocess ────────────────────────────────────────────────────────────
def preprocess(df):
    # Fill missing values with column mean
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    return X, y


# ─── 3. Scale + Split ─────────────────────────────────────────────────────────
def scale_and_split(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler


# ─── 4. Train ─────────────────────────────────────────────────────────────────
def train(X_train, y_train):
    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")
    return model


# ─── 5. Evaluate ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, feature_names):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    auc  = roc_auc_score(y_test, probs)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # ── Confusion Matrix Plot ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Potable", "Potable"],
                yticklabels=["Not Potable", "Potable"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close()

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="steelblue")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"))
    plt.close()

    # ── Feature Importance ────────────────────────────────────────────────────
    coefs = model.coef_[0]
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs
    }).sort_values("Coefficient", key=abs, ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_df, x="Coefficient", y="Feature",
                palette="coolwarm")
    plt.title("Logistic Regression — Feature Coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"))
    plt.close()

    print("[INFO] Evaluation plots saved to model/plots/")
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc}


# ─── 6. Save ──────────────────────────────────────────────────────────────────
def save(model, scaler):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Model saved  → {MODEL_PATH}")
    print(f"[INFO] Scaler saved → {SCALER_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df                              = load_data()
    X, y                            = preprocess(df)
    X_train, X_test, y_train, y_test, scaler = scale_and_split(X, y)
    model                           = train(X_train, y_train)
    metrics                         = evaluate(model, X_test, y_test, list(X.columns))
    save(model, scaler)
    print("\n[DONE] Training pipeline complete.")
