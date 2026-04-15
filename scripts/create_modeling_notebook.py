import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Modeling\n",
            "Train Logistic Regression, Random Forest, and XGBoost.\n",
            "Tune decision threshold to meet business constraint:\n",
            "reduce defaults by 30%, reject at most 10% more good borrowers."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from xgboost import XGBClassifier\n",
            "from sklearn.metrics import (\n",
            "    classification_report, roc_auc_score, roc_curve,\n",
            "    confusion_matrix, precision_recall_curve\n",
            ")\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Load preprocessed data\n",
            "X_train = pd.read_parquet('../data/X_train.parquet')\n",
            "X_test  = pd.read_parquet('../data/X_test.parquet')\n",
            "y_train = pd.read_parquet('../data/y_train.parquet').squeeze()\n",
            "y_test  = pd.read_parquet('../data/y_test.parquet').squeeze()\n",
            "\n",
            "# Drop redundant features identified in significance testing\n",
            "DROP_COLS = ['int_rate', 'installment', 'total_acc']\n",
            "X_train = X_train.drop(columns=DROP_COLS)\n",
            "X_test  = X_test.drop(columns=DROP_COLS)\n",
            "\n",
            "print('X_train shape:', X_train.shape)\n",
            "print('X_test shape: ', X_test.shape)\n",
            "print('Features used:', list(X_train.columns))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Helper: Evaluation Function\n",
            "Reusable function to evaluate any model consistently."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def evaluate_model(name, model, X_test, y_test, threshold=0.5):\n",
            "    proba = model.predict_proba(X_test)[:, 1]\n",
            "    preds = (proba >= threshold).astype(int)\n",
            "\n",
            "    auc   = roc_auc_score(y_test, proba)\n",
            "    cm    = confusion_matrix(y_test, preds)\n",
            "    tn, fp, fn, tp = cm.ravel()\n",
            "\n",
            "    default_catch_rate = tp / (tp + fn)   # recall for defaults\n",
            "    good_rejection_rate = fp / (tn + fp)  # % good borrowers incorrectly rejected\n",
            "\n",
            "    print(f'\\n=== {name} (threshold={threshold}) ===')\n",
            "    print(f'ROC-AUC:              {auc:.4f}')\n",
            "    print(f'Default Catch Rate:   {default_catch_rate:.2%}  (recall for class 1)')\n",
            "    print(f'Good Rejection Rate:  {good_rejection_rate:.2%}  (false positive rate)')\n",
            "    print(f'Confusion Matrix:')\n",
            "    print(f'  TN={tn:,}  FP={fp:,}')\n",
            "    print(f'  FN={fn:,}  TP={tp:,}')\n",
            "    print()\n",
            "    print(classification_report(y_test, preds, target_names=['Fully Paid', 'Default']))\n",
            "    return proba, auc"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Model 1: Logistic Regression (Baseline)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)\n",
            "lr.fit(X_train, y_train)\n",
            "lr_proba, lr_auc = evaluate_model('Logistic Regression', lr, X_test, y_test)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Model 2: Random Forest"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "rf = RandomForestClassifier(\n",
            "    n_estimators=200,\n",
            "    max_depth=12,\n",
            "    min_samples_leaf=50,\n",
            "    class_weight='balanced',\n",
            "    random_state=42,\n",
            "    n_jobs=-1\n",
            ")\n",
            "rf.fit(X_train, y_train)\n",
            "rf_proba, rf_auc = evaluate_model('Random Forest', rf, X_test, y_test)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Model 3: XGBoost"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()\n",
            "\n",
            "xgb = XGBClassifier(\n",
            "    n_estimators=300,\n",
            "    max_depth=6,\n",
            "    learning_rate=0.1,\n",
            "    subsample=0.8,\n",
            "    colsample_bytree=0.8,\n",
            "    scale_pos_weight=scale_pos_weight,\n",
            "    random_state=42,\n",
            "    n_jobs=-1,\n",
            "    eval_metric='auc',\n",
            "    verbosity=0\n",
            ")\n",
            "xgb.fit(X_train, y_train)\n",
            "xgb_proba, xgb_auc = evaluate_model('XGBoost', xgb, X_test, y_test)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## ROC Curve Comparison"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(8, 6))\n",
            "for name, proba, color in [\n",
            "    ('Logistic Regression', lr_proba, 'steelblue'),\n",
            "    ('Random Forest',       rf_proba, 'green'),\n",
            "    ('XGBoost',             xgb_proba,'tomato')\n",
            "]:\n",
            "    fpr, tpr, _ = roc_curve(y_test, proba)\n",
            "    auc = roc_auc_score(y_test, proba)\n",
            "    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', color=color)\n",
            "\n",
            "plt.plot([0,1],[0,1],'k--', label='Random')\n",
            "plt.xlabel('False Positive Rate (Good Borrowers Rejected)')\n",
            "plt.ylabel('True Positive Rate (Defaults Caught)')\n",
            "plt.title('ROC Curve - All Models')\n",
            "plt.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/roc_curves.png', dpi=100)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Threshold Tuning for Business Constraint\n",
            "**Goal:** Catch at least 30% more defaults, reject at most 10% more good borrowers.\n",
            "We tune the decision threshold on the best model (XGBoost)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Baseline at default threshold 0.5\n",
            "base_preds = (xgb_proba >= 0.5).astype(int)\n",
            "cm_base = confusion_matrix(y_test, base_preds)\n",
            "tn_b, fp_b, fn_b, tp_b = cm_base.ravel()\n",
            "base_catch = tp_b / (tp_b + fn_b)\n",
            "base_good_reject = fp_b / (tn_b + fp_b)\n",
            "print(f'Baseline (t=0.5): catch={base_catch:.2%}, good_rejected={base_good_reject:.2%}')\n",
            "\n",
            "# Target: catch >= base + 30%, good_rejected <= base_good_reject + 10%\n",
            "target_catch = base_catch * 1.30\n",
            "max_good_reject = base_good_reject + 0.10\n",
            "print(f'Target catch rate:      >= {target_catch:.2%}')\n",
            "print(f'Max good rejection:     <= {max_good_reject:.2%}')\n",
            "\n",
            "# Sweep thresholds\n",
            "results = []\n",
            "for t in np.arange(0.1, 0.9, 0.01):\n",
            "    preds = (xgb_proba >= t).astype(int)\n",
            "    cm = confusion_matrix(y_test, preds)\n",
            "    tn, fp, fn, tp = cm.ravel()\n",
            "    catch = tp / (tp + fn)\n",
            "    good_rej = fp / (tn + fp)\n",
            "    results.append({'threshold': round(t, 2), 'catch_rate': catch, 'good_rejection': good_rej})\n",
            "\n",
            "thresh_df = pd.DataFrame(results)\n",
            "\n",
            "# Find thresholds meeting the business constraint\n",
            "valid = thresh_df[\n",
            "    (thresh_df['catch_rate'] >= target_catch) &\n",
            "    (thresh_df['good_rejection'] <= max_good_reject)\n",
            "].copy()\n",
            "\n",
            "print(f'\\nThresholds meeting business constraint:')\n",
            "if len(valid) > 0:\n",
            "    print(valid.to_string(index=False))\n",
            "    best_t = valid.sort_values('good_rejection').iloc[0]['threshold']\n",
            "    print(f'\\nBest threshold: {best_t}')\n",
            "else:\n",
            "    print('No threshold meets both constraints exactly.')\n",
            "    print('\\nClosest options:')\n",
            "    thresh_df['score'] = thresh_df['catch_rate'] - thresh_df['good_rejection']\n",
            "    print(thresh_df.sort_values('score', ascending=False).head(5).to_string(index=False))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Final Model Evaluation at Optimal Threshold"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Use best threshold found above (or manually set if none found)\n",
            "try:\n",
            "    final_threshold = best_t\n",
            "except NameError:\n",
            "    final_threshold = 0.4  # fallback\n",
            "\n",
            "print(f'Evaluating XGBoost at threshold = {final_threshold}')\n",
            "_, _ = evaluate_model('XGBoost (tuned threshold)', xgb, X_test, y_test, threshold=final_threshold)\n",
            "\n",
            "# Threshold vs metrics plot\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "axes[0].plot(thresh_df['threshold'], thresh_df['catch_rate'], color='tomato', label='Default Catch Rate')\n",
            "axes[0].plot(thresh_df['threshold'], thresh_df['good_rejection'], color='steelblue', label='Good Rejection Rate')\n",
            "axes[0].axvline(final_threshold, color='black', linestyle='--', label=f'Chosen threshold={final_threshold}')\n",
            "axes[0].axhline(target_catch, color='tomato', linestyle=':', alpha=0.5, label=f'Target catch={target_catch:.0%}')\n",
            "axes[0].axhline(max_good_reject, color='steelblue', linestyle=':', alpha=0.5, label=f'Max good reject={max_good_reject:.0%}')\n",
            "axes[0].set_xlabel('Threshold')\n",
            "axes[0].set_ylabel('Rate')\n",
            "axes[0].set_title('Threshold vs Rates')\n",
            "axes[0].legend(fontsize=8)\n",
            "\n",
            "# Feature importance\n",
            "feat_imp = pd.Series(xgb.feature_importances_, index=X_train.columns)\n",
            "feat_imp.sort_values(ascending=True).tail(15).plot(kind='barh', ax=axes[1], color='steelblue')\n",
            "axes[1].set_title('XGBoost Feature Importance (Top 15)')\n",
            "axes[1].set_xlabel('Importance Score')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/threshold_tuning.png', dpi=100)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Save Final Model"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pickle\n",
            "\n",
            "with open('../data/xgb_model.pkl', 'wb') as f:\n",
            "    pickle.dump(xgb, f)\n",
            "\n",
            "with open('../data/threshold.txt', 'w') as f:\n",
            "    f.write(str(final_threshold))\n",
            "\n",
            "print('Saved: xgb_model.pkl')\n",
            "print('Saved: threshold.txt ->', final_threshold)\n",
            "print('\\nDropped features (redundant):', DROP_COLS)\n",
            "print('Final feature count:', X_train.shape[1])"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('e:/Projects/LendingClub_Loan/notebooks/04_modeling.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Created 04_modeling.ipynb successfully')
