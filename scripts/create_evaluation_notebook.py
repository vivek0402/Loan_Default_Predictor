import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Final Evaluation\n", "Comprehensive evaluation of the XGBoost model with business impact analysis."]
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
            "import matplotlib.gridspec as gridspec\n",
            "import seaborn as sns\n",
            "import pickle\n",
            "from xgboost import XGBClassifier\n",
            "from sklearn.metrics import (\n",
            "    roc_auc_score, roc_curve, confusion_matrix,\n",
            "    classification_report, precision_recall_curve, fbeta_score\n",
            ")\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "\n",
            "# Load data and model\n",
            "X_train = pd.read_parquet('../data/X_train.parquet').drop(columns=['installment','total_acc'])\n",
            "X_test  = pd.read_parquet('../data/X_test.parquet').drop(columns=['installment','total_acc'])\n",
            "y_train = pd.read_parquet('../data/y_train.parquet').squeeze()\n",
            "y_test  = pd.read_parquet('../data/y_test.parquet').squeeze()\n",
            "\n",
            "with open('../data/xgb_model.pkl', 'rb') as f:\n",
            "    model = pickle.load(f)\n",
            "\n",
            "with open('../data/threshold.txt') as f:\n",
            "    THRESHOLD = float(f.read())\n",
            "\n",
            "proba = model.predict_proba(X_test)[:, 1]\n",
            "preds = (proba >= THRESHOLD).astype(int)\n",
            "cm = confusion_matrix(y_test, preds)\n",
            "tn, fp, fn, tp = cm.ravel()\n",
            "\n",
            "print('Model loaded. Threshold:', THRESHOLD)\n",
            "print('Test set size:', len(y_test))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Confusion Matrix"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(6, 5))\n",
            "cm_df = pd.DataFrame(\n",
            "    cm,\n",
            "    index=['Actual: Paid', 'Actual: Default'],\n",
            "    columns=['Predicted: Paid', 'Predicted: Default']\n",
            ")\n",
            "sns.heatmap(cm_df, annot=True, fmt=',', cmap='Blues', linewidths=0.5, ax=ax)\n",
            "ax.set_title(f'Confusion Matrix (threshold={THRESHOLD})')\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/eval_confusion_matrix.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print(classification_report(y_test, preds, target_names=['Fully Paid', 'Default']))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. ROC & Precision-Recall Curves"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# ROC Curve\n",
            "fpr, tpr, _ = roc_curve(y_test, proba)\n",
            "auc = roc_auc_score(y_test, proba)\n",
            "axes[0].plot(fpr, tpr, color='tomato', lw=2, label=f'XGBoost (AUC={auc:.4f})')\n",
            "axes[0].plot([0,1],[0,1],'k--', label='Random classifier')\n",
            "axes[0].scatter(\n",
            "    fp/(tn+fp), tp/(tp+fn),\n",
            "    color='black', s=100, zorder=5,\n",
            "    label=f'Operating point (t={THRESHOLD})'\n",
            ")\n",
            "axes[0].set_xlabel('False Positive Rate')\n",
            "axes[0].set_ylabel('True Positive Rate')\n",
            "axes[0].set_title('ROC Curve')\n",
            "axes[0].legend()\n",
            "\n",
            "# Precision-Recall Curve\n",
            "prec, rec, thresh_pr = precision_recall_curve(y_test, proba)\n",
            "axes[1].plot(rec, prec, color='steelblue', lw=2)\n",
            "baseline_pr = y_test.mean()\n",
            "axes[1].axhline(baseline_pr, color='k', linestyle='--', label=f'Baseline precision={baseline_pr:.2f}')\n",
            "axes[1].set_xlabel('Recall (Default Catch Rate)')\n",
            "axes[1].set_ylabel('Precision')\n",
            "axes[1].set_title('Precision-Recall Curve')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/eval_roc_pr.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print(f'ROC-AUC: {auc:.4f}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Feature Importance"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)\n",
            "feat_imp_sorted = feat_imp.sort_values(ascending=True)\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 8))\n",
            "colors = ['tomato' if v > feat_imp.median() else 'steelblue' for v in feat_imp_sorted]\n",
            "feat_imp_sorted.plot(kind='barh', ax=ax, color=colors)\n",
            "ax.set_title('XGBoost Feature Importance')\n",
            "ax.set_xlabel('Importance Score')\n",
            "ax.axvline(feat_imp.median(), color='black', linestyle='--', alpha=0.5, label='Median')\n",
            "ax.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/eval_feature_importance.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print('Top 10 features:')\n",
            "print(feat_imp.sort_values(ascending=False).head(10).round(4))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Business Impact Summary"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "total = len(y_test)\n",
            "approved = tn + fn\n",
            "rejected = tp + fp\n",
            "default_rate_no_model = (tp + fn) / total\n",
            "default_rate_approved  = fn / approved if approved > 0 else 0\n",
            "default_reduction = (default_rate_no_model - default_rate_approved) / default_rate_no_model\n",
            "catch_rate = tp / (tp + fn)\n",
            "good_rej_rate = fp / (tn + fp)\n",
            "\n",
            "print('=' * 50)\n",
            "print('       BUSINESS IMPACT SUMMARY')\n",
            "print('=' * 50)\n",
            "print(f'Total applications evaluated:  {total:>10,}')\n",
            "print(f'Loans approved by model:       {approved:>10,}  ({approved/total:.1%})')\n",
            "print(f'Loans rejected by model:       {rejected:>10,}  ({rejected/total:.1%})')\n",
            "print()\n",
            "print(f'--- Default Performance ---')\n",
            "print(f'Defaults in test set:          {tp+fn:>10,}')\n",
            "print(f'Defaults caught & rejected:    {tp:>10,}  ({catch_rate:.1%} catch rate)')\n",
            "print(f'Defaults missed (approved):    {fn:>10,}')\n",
            "print()\n",
            "print(f'--- Default Rate Comparison ---')\n",
            "print(f'Without model (all approved):  {default_rate_no_model:>10.2%}')\n",
            "print(f'With model (approved only):    {default_rate_approved:>10.2%}')\n",
            "print(f'Default rate reduction:        {default_reduction:>10.1%}')\n",
            "print()\n",
            "print(f'--- Good Borrower Impact ---')\n",
            "print(f'Good borrowers in test set:    {tn+fp:>10,}')\n",
            "print(f'Good borrowers approved:       {tn:>10,}  ({tn/(tn+fp):.1%})')\n",
            "print(f'Good borrowers rejected (FP):  {fp:>10,}  ({good_rej_rate:.1%})')\n",
            "print()\n",
            "print(f'ROC-AUC:  {auc:.4f}')\n",
            "print(f'F2 Score: {fbeta_score(y_test, preds, beta=2):.4f}')\n",
            "print(f'Threshold used: {THRESHOLD}')\n",
            "print('=' * 50)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Hypothesis Validation Summary"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "hypotheses = [\n",
            "    ['H1', 'Higher DTI -> Higher Default',        'dti',                  19.75, 17.10, 'CONFIRMED',       'dti in top 5 features'],\n",
            "    ['H2', 'Shorter Credit History -> Default',   'credit_history_years', 17.85, 18.68, 'WEAK',            'Only 0.83yr difference'],\n",
            "    ['H3', 'Loan Purpose -> Higher Risk',         'purpose',              '-',   '-',   'CONFIRMED',       'Chi2=4182, p=0.0'],\n",
            "    ['H4', 'Lower Income -> Higher Default',      'annual_inc',           60000, 65000, 'CONFIRMED',       '$5k median gap'],\n",
            "    ['H5', 'More Derogatory Marks -> Default',    'has_derogatory',       0.246, 0.207, 'WEAK',            'Converted to binary flag'],\n",
            "]\n",
            "\n",
            "hyp_df = pd.DataFrame(\n",
            "    hypotheses,\n",
            "    columns=['ID', 'Hypothesis', 'Feature', 'Default Val', 'Paid Val', 'Verdict', 'Notes']\n",
            ")\n",
            "print(hyp_df.to_string(index=False))\n",
            "\n",
            "# Feature importance for hypothesis features\n",
            "hyp_features = ['dti', 'credit_history_years', 'annual_inc', 'has_derogatory']\n",
            "print('\\nFeature importance for hypothesis features:')\n",
            "for f in hyp_features:\n",
            "    if f in feat_imp.index:\n",
            "        print(f'  {f:25s}: {feat_imp[f]:.4f}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Score Distribution Plot"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            "\n",
            "ax.hist(proba[y_test==0], bins=50, alpha=0.6, color='steelblue', label='Fully Paid', density=True)\n",
            "ax.hist(proba[y_test==1], bins=50, alpha=0.6, color='tomato',    label='Default',    density=True)\n",
            "ax.axvline(THRESHOLD, color='black', linestyle='--', lw=2, label=f'Threshold={THRESHOLD}')\n",
            "ax.set_xlabel('Predicted Default Probability')\n",
            "ax.set_ylabel('Density')\n",
            "ax.set_title('Score Distribution: Paid vs Default')\n",
            "ax.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/eval_score_dist.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print('Median predicted probability:')\n",
            "print(f'  Fully Paid: {proba[y_test==0].mean():.3f}')\n",
            "print(f'  Default:    {proba[y_test==1].mean():.3f}')"
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

with open('e:/Projects/LendingClub_Loan/notebooks/05_evaluation.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Created 05_evaluation.ipynb successfully')
