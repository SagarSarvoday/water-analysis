"""
train_model.py — AI Water Quality Prediction System
=====================================================
Pipeline:
  1. Load CSV & fill nulls with median
  2. WHO-based label engineering (weighted violation ratio)
  3. IQR outlier clipping
  4. RobustScaler + stratified 80/20 split
  5. SMOTE on training data
  6. Train 4 models:
       - Logistic Regression          (9 features, GridSearchCV tuned)
       - Random Forest                (9 features)
       - Gradient Boosting            (9 features)
       - Improved Logistic Regression (6 features via RFE, heavily tuned)
  7. Save all models + scaler + rfe_selector
  8. Save plots + metrics.json
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing     import RobustScaler
from sklearn.model_selection   import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics           import (accuracy_score, precision_score, recall_score,
                                        f1_score, roc_auc_score, confusion_matrix, roc_curve)
from imblearn.over_sampling    import SMOTE

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "water_potability.csv")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── WHO Standards & Weights ────────────────────────────────────────────────────
WHO_STANDARDS = {
    "ph":              (6.5,  8.5),
    "Hardness":        (0,    500.0),
    "Solids":          (0,    600.0),
    "Chloramines":     (0,    4.0),
    "Sulfate":         (0,    500.0),
    "Conductivity":    (0,    400.0),
    "Organic_carbon":  (0,    4.0),
    "Turbidity":       (0,    4.0),
    "Trihalomethanes": (0,    80.0),
}
WHO_WEIGHTS = {
    "ph": 2, "Hardness": 1, "Solids": 1, "Chloramines": 2,
    "Sulfate": 1, "Conductivity": 1, "Organic_carbon": 2,
    "Turbidity": 2, "Trihalomethanes": 2,
}

FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Turbidity", "Trihalomethanes"]

# ── WHO Label Engineering ──────────────────────────────────────────────────────
def who_label(row):
    weighted_violations = 0
    total_weight        = 0
    for col, (lo, hi) in WHO_STANDARDS.items():
        if col in row.index and pd.notna(row[col]):
            w = WHO_WEIGHTS.get(col, 1)
            total_weight += w
            if not (lo <= row[col] <= hi):
                weighted_violations += w
    violation_ratio = weighted_violations / total_weight if total_weight > 0 else 0
    return 1 if violation_ratio < 0.50 else 0

# ── Metrics helper ─────────────────────────────────────────────────────────────
def get_metrics(clf, X, y):
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    return {
        "accuracy":  round(float(accuracy_score(y, y_pred)),               4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, y_pred, zero_division=0)),    4),
        "f1":        round(float(f1_score(y, y_pred, zero_division=0)),        4),
        "auc":       round(float(roc_auc_score(y, y_prob)),                4),
    }

# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  AI WATER QUALITY — MODEL TRAINING")
    print("="*60)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print("\n[1/9] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"     Shape: {df.shape}")

    # ── 2. Fill nulls ──────────────────────────────────────────────────────────
    print("[2/9] Filling nulls with median...")
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ── 3. WHO label engineering ───────────────────────────────────────────────
    print("[3/9] Regenerating labels with WHO weighted standards...")
    df["Potability"] = df.apply(who_label, axis=1)
    vc = df["Potability"].value_counts()
    print(f"     Safe(1): {vc.get(1,0)}  Unsafe(0): {vc.get(0,0)}")

    # ── 4. Clip outliers (IQR) ─────────────────────────────────────────────────
    print("[4/9] Clipping outliers (3xIQR)...")
    for col in FEATURES:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR     = Q3 - Q1
        df[col] = df[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)

    # ── 5. Scale + split ───────────────────────────────────────────────────────
    print("[5/9] Scaling and splitting...")
    X = df[FEATURES]
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler         = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURES)
    X_test_df  = pd.DataFrame(X_test_scaled,  columns=FEATURES)

    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    print(f"     Train: {X_train_df.shape}  Test: {X_test_df.shape}")

    # ── 6. SMOTE ───────────────────────────────────────────────────────────────
    print("[6/9] Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_train_df, y_train)
    print(f"     After SMOTE — Safe: {sum(y_bal==1)}  Unsafe: {sum(y_bal==0)}")

    all_metrics = {}
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 7a. Logistic Regression — GridSearchCV tuned (9 features) ──────────────
    print("\n[7/9] Training models...")
    print("  >> Logistic Regression (GridSearchCV, 9 features)...")
    lr_params = {
        "C":            [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "penalty":      ["l1", "l2", "elasticnet"],
        "solver":       ["saga"],
        "l1_ratio":     [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_iter":     [2000],
        "class_weight": ["balanced"],
    }
    lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params,
                           cv=cv5, scoring="f1", n_jobs=-1, refit=True)
    lr_grid.fit(X_bal, y_bal)
    lr = lr_grid.best_estimator_
    print(f"     Best: {lr_grid.best_params_}")
    cv_lr = cross_val_score(lr, X_bal, y_bal, cv=cv5, scoring="f1").mean()
    all_metrics["Logistic Regression"] = get_metrics(lr, X_test_df, y_test)
    all_metrics["Logistic Regression"]["cv_f1"] = round(float(cv_lr), 4)
    joblib.dump(lr, os.path.join(BASE_DIR, "logistic_regression.pkl"))
    print(f"     Acc={all_metrics['Logistic Regression']['accuracy']:.3f}  "
          f"F1={all_metrics['Logistic Regression']['f1']:.3f}  CV-F1={cv_lr:.3f}")

    # ── 7b. Random Forest (9 features) ────────────────────────────────────────
    print("  >> Random Forest (9 features)...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_bal, y_bal)
    cv_rf = cross_val_score(rf, X_bal, y_bal, cv=cv5, scoring="f1").mean()
    all_metrics["Random Forest"] = get_metrics(rf, X_test_df, y_test)
    all_metrics["Random Forest"]["cv_f1"] = round(float(cv_rf), 4)
    joblib.dump(rf, os.path.join(BASE_DIR, "random_forest.pkl"))
    joblib.dump(rf, os.path.join(BASE_DIR, "water_model.pkl"))
    print(f"     Acc={all_metrics['Random Forest']['accuracy']:.3f}  "
          f"F1={all_metrics['Random Forest']['f1']:.3f}  CV-F1={cv_rf:.3f}")

    # ── 7c. Gradient Boosting (9 features) ────────────────────────────────────
    print("  >> Gradient Boosting (9 features)...")
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=4, min_samples_leaf=2,
        subsample=0.8, random_state=42)
    gb.fit(X_bal, y_bal)
    cv_gb = cross_val_score(gb, X_bal, y_bal, cv=cv5, scoring="f1").mean()
    all_metrics["Gradient Boosting"] = get_metrics(gb, X_test_df, y_test)
    all_metrics["Gradient Boosting"]["cv_f1"] = round(float(cv_gb), 4)
    joblib.dump(gb, os.path.join(BASE_DIR, "gradient_boosting.pkl"))
    print(f"     Acc={all_metrics['Gradient Boosting']['accuracy']:.3f}  "
          f"F1={all_metrics['Gradient Boosting']['f1']:.3f}  CV-F1={cv_gb:.3f}")

    # ── 7d. RFE (select 6 best features) ──────────────────────────────────────
    print("  >> RFE feature selection...")
    rfe_est = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe     = RFE(estimator=rfe_est, n_features_to_select=6, step=1)
    rfe.fit(X_bal, y_bal)
    joblib.dump(rfe, os.path.join(BASE_DIR, "rfe_selector.pkl"))
    selected = [FEATURES[i] for i in range(len(FEATURES)) if rfe.support_[i]]
    print(f"     Selected: {selected}")

    X_bal_sel  = rfe.transform(X_bal)
    X_test_sel = rfe.transform(X_test_df)

    # ── 7e. Improved LR — heavily tuned on 6 RFE features ─────────────────────
    print("  >> Improved Logistic Regression (GridSearchCV, 6 RFE features)...")
    ilr_params = {
        "C":            [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        "penalty":      ["l1", "l2", "elasticnet"],
        "solver":       ["saga"],
        "l1_ratio":     [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_iter":     [3000],
        "class_weight": ["balanced"],
    }
    ilr_grid = GridSearchCV(LogisticRegression(random_state=42), ilr_params,
                            cv=cv5, scoring="f1", n_jobs=-1, refit=True)
    ilr_grid.fit(X_bal_sel, y_bal)
    best_lr = ilr_grid.best_estimator_
    print(f"     Best: {ilr_grid.best_params_}")
    cv_blr = cross_val_score(best_lr, X_bal_sel, y_bal, cv=cv5, scoring="f1").mean()
    all_metrics["Improved Logistic Regression"] = get_metrics(best_lr, X_test_sel, y_test)
    all_metrics["Improved Logistic Regression"]["cv_f1"] = round(float(cv_blr), 4)
    joblib.dump(best_lr, os.path.join(BASE_DIR, "improved_logistic_regression.pkl"))
    print(f"     Acc={all_metrics['Improved Logistic Regression']['accuracy']:.3f}  "
          f"F1={all_metrics['Improved Logistic Regression']['f1']:.3f}  CV-F1={cv_blr:.3f}")

    # ── 8. Save metrics ────────────────────────────────────────────────────────
    print("\n[8/9] Saving metrics.json...")
    with open(os.path.join(PLOT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n  ── FINAL SUMMARY ────────────────────────────")
    for name, m in all_metrics.items():
        print(f"  {name:<35} Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  AUC={m['auc']:.3f}")

    # ── 9. Plots ───────────────────────────────────────────────────────────────
    print("\n[9/9] Saving plots...")

    # Confusion matrix
    cm = confusion_matrix(y_test, rf.predict(X_test_df))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Unsafe","Safe"]); ax.set_yticklabels(["Unsafe","Safe"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test_df)[:,1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1a6fa8", lw=2,
            label=f"RF AUC={all_metrics['Random Forest']['auc']:.3f}")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve — Random Forest"); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=150); plt.close()

    # Feature importance
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(FEATURES)), importances[idx],
           color=["#1a6fa8" if rfe.support_[i] else "#aaaaaa" for i in idx])
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in idx], rotation=35, ha="right")
    ax.set_title("Feature Importance — RF (Blue=RFE Selected)")
    ax.set_ylabel("Importance")
    ax.legend(handles=[mpatches.Patch(color="#1a6fa8", label="RFE Selected"),
                        mpatches.Patch(color="#aaaaaa", label="Not Selected")])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance_selector.png"), dpi=150); plt.close()

    # RFE ranking
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(FEATURES, rfe.ranking_,
           color=["#2e7d32" if rfe.support_[i] else "#c62828" for i in range(len(FEATURES))])
    ax.set_xticklabels(FEATURES, rotation=35, ha="right")
    ax.set_title("RFE Feature Ranking (Green=Selected, Rank 1=Best)")
    ax.set_ylabel("Rank")
    ax.legend(handles=[mpatches.Patch(color="#2e7d32", label="Selected"),
                        mpatches.Patch(color="#c62828", label="Eliminated")])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rfe_feature_ranking.png"), dpi=150); plt.close()

    # Model comparison
    model_names  = list(all_metrics.keys())
    metrics_keys = ["accuracy","precision","recall","f1","auc"]
    x     = np.arange(len(metrics_keys))
    width = 0.18
    clrs  = ["#1565c0","#2e7d32","#e65100","#6a1b9a"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        ax.bar(x + i*width, [all_metrics[name][k] for k in metrics_keys],
               width, label=name, color=clrs[i])
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(["Accuracy","Precision","Recall","F1","AUC"])
    ax.set_ylim(0, 1.2); ax.set_title("All 4 Models Comparison"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"), dpi=150); plt.close()

    # LR improvement
    base = all_metrics["Logistic Regression"]
    imp  = all_metrics["Improved Logistic Regression"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(metrics_keys))
    ax.bar(x - 0.2, [base[k] for k in metrics_keys], 0.38, label="Baseline LR", color="#1565c0")
    ax.bar(x + 0.2, [imp[k]  for k in metrics_keys], 0.38, label="Improved LR", color="#6a1b9a")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy","Precision","Recall","F1","AUC"])
    ax.set_ylim(0, 1.2); ax.set_title("Baseline LR vs Improved LR"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lr_improvement.png"), dpi=150); plt.close()

    print("\n✅ All done!")
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()