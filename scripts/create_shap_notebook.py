import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# SHAP Explainability\n",
            "SHAP (SHapley Additive exPlanations) explains *why* the model made each individual prediction.\n",
            "Rather than just knowing which features matter overall, SHAP tells us exactly how much\n",
            "each feature pushed a specific loan toward or away from default.\n\n",
            "**Sections:**\n",
            "1. Setup & compute SHAP values\n",
            "2. Global summary plot (beeswarm)\n",
            "3. Mean absolute SHAP bar plot\n",
            "4. Single loan waterfall — explain one prediction\n",
            "5. DTI dependence plot\n",
            "6. High-risk vs low-risk loan comparison\n",
            "7. Key takeaways"
        ]
    },
    # ── 1. Setup ────────────────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Setup & Compute SHAP Values"]
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
            "import matplotlib.ticker as mticker\n",
            "import pickle\n",
            "import shap\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Load model and test data\n",
            "X_test = pd.read_parquet('../data/X_test.parquet').drop(columns=['installment', 'total_acc'])\n",
            "y_test = pd.read_parquet('../data/y_test.parquet').squeeze()\n",
            "\n",
            "with open('../data/xgb_model.pkl', 'rb') as f:\n",
            "    model = pickle.load(f)\n",
            "\n",
            "with open('../data/threshold.txt') as f:\n",
            "    THRESHOLD = float(f.read())\n",
            "\n",
            "print('Model loaded.')\n",
            "print('Test set shape:', X_test.shape)\n",
            "print('Threshold:', THRESHOLD)\n",
            "\n",
            "# Predict probabilities\n",
            "proba = model.predict_proba(X_test)[:, 1]\n",
            "preds = (proba >= THRESHOLD).astype(int)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute SHAP values using a sample (full test set is 270K rows — too slow)\n",
            "# 5,000 rows gives stable, representative SHAP values in ~1-2 minutes\n",
            "print('Sampling 5,000 rows for SHAP computation...')\n",
            "np.random.seed(42)\n",
            "sample_idx = np.random.choice(len(X_test), size=5000, replace=False)\n",
            "X_sample   = X_test.iloc[sample_idx].reset_index(drop=True)\n",
            "y_sample   = y_test.iloc[sample_idx].reset_index(drop=True)\n",
            "p_sample   = proba[sample_idx]\n",
            "\n",
            "print('Computing SHAP values (TreeExplainer — fast for XGBoost)...')\n",
            "explainer   = shap.TreeExplainer(model)\n",
            "shap_values = explainer.shap_values(X_sample)\n",
            "\n",
            "print('Done. SHAP values shape:', shap_values.shape)\n",
            "print('Expected value (base rate):', explainer.expected_value.round(4))"
        ]
    },
    # ── 2. Beeswarm Summary ─────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Global Summary Plot (Beeswarm)\n",
            "Each dot is one loan. Position on X-axis = how much that feature pushed the prediction\n",
            "toward default (positive) or away from default (negative).\n",
            "Colour = the actual feature value (red = high, blue = low)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 8))\n",
            "shap.summary_plot(\n",
            "    shap_values, X_sample,\n",
            "    max_display=20,\n",
            "    show=False,\n",
            "    plot_size=(10, 8)\n",
            ")\n",
            "plt.title('SHAP Summary Plot — Top 20 Features\\n'\n",
            "          'Each dot = one loan | X-axis = impact on default probability | '\n",
            "          'Colour = feature value (red=high, blue=low)',\n",
            "          fontsize=10, pad=15)\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/shap_summary.png', dpi=120, bbox_inches='tight')\n",
            "plt.show()\n",
            "print('Saved: shap_summary.png')"
        ]
    },
    # ── 3. Bar Plot ─────────────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Mean Absolute SHAP — Feature Importance Bar Chart\n",
            "The average magnitude of each feature's SHAP value across all loans.\n",
            "This is a more rigorous version of XGBoost's built-in feature importance."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "mean_abs_shap = pd.Series(\n",
            "    np.abs(shap_values).mean(axis=0),\n",
            "    index=X_sample.columns\n",
            ").sort_values(ascending=True)\n",
            "\n",
            "top20 = mean_abs_shap.tail(20)\n",
            "colors = ['#C0392B' if v > mean_abs_shap.median() else '#2980B9' for v in top20]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 7))\n",
            "bars = ax.barh(top20.index, top20.values, color=colors, edgecolor='white')\n",
            "for bar, val in zip(bars, top20.values):\n",
            "    ax.text(val + 0.0003, bar.get_y() + bar.get_height()/2,\n",
            "            f'{val:.4f}', va='center', fontsize=8)\n",
            "ax.set_xlabel('Mean |SHAP value| (average impact on model output)', fontsize=11)\n",
            "ax.set_title('Feature Importance via SHAP\\nTop 20 features ranked by average impact on default probability',\n",
            "             fontsize=11)\n",
            "ax.axvline(mean_abs_shap.median(), color='gray', linestyle='--', alpha=0.6, label='Median')\n",
            "ax.legend(fontsize=9)\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/shap_bar.png', dpi=120, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print('Top 10 features by mean |SHAP|:')\n",
            "print(mean_abs_shap.sort_values(ascending=False).head(10).round(5).to_string())"
        ]
    },
    # ── 4. Waterfall — Single Loan ───────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Waterfall Plot — Explaining a Single Loan\n",
            "Pick one high-risk loan and one low-risk loan, and show exactly which features\n",
            "pushed the model's prediction up or down from the base rate.\n\n",
            "**How to read it:**\n",
            "- Start from E[f(x)] — the average model output across all loans (base rate)\n",
            "- Red bars push the prediction HIGHER (toward default)\n",
            "- Blue bars push the prediction LOWER (away from default)\n",
            "- The final value f(x) is the model's predicted default probability for that loan"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Pick a high-risk loan (high predicted probability, actually defaulted)\n",
            "high_risk_mask = (y_sample == 1) & (p_sample >= 0.7)\n",
            "if high_risk_mask.sum() == 0:\n",
            "    high_risk_mask = (p_sample >= p_sample.quantile(0.95))\n",
            "high_risk_idx = np.where(high_risk_mask)[0][0]\n",
            "\n",
            "# Pick a low-risk loan (low predicted probability, actually paid)\n",
            "low_risk_mask = (y_sample == 0) & (p_sample <= 0.15)\n",
            "if low_risk_mask.sum() == 0:\n",
            "    low_risk_mask = (p_sample <= p_sample.quantile(0.05))\n",
            "low_risk_idx = np.where(low_risk_mask)[0][0]\n",
            "\n",
            "print(f'High-risk loan index: {high_risk_idx}')\n",
            "print(f'  Predicted probability: {p_sample[high_risk_idx]:.3f}')\n",
            "print(f'  Actual outcome: {\"DEFAULT\" if y_sample.iloc[high_risk_idx]==1 else \"PAID\"}')\n",
            "print()\n",
            "print(f'Low-risk loan index: {low_risk_idx}')\n",
            "print(f'  Predicted probability: {p_sample[low_risk_idx]:.3f}')\n",
            "print(f'  Actual outcome: {\"DEFAULT\" if y_sample.iloc[low_risk_idx]==1 else \"PAID\"}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── High-risk waterfall ──────────────────────────────────────────────────────\n",
            "explanation_high = shap.Explanation(\n",
            "    values        = shap_values[high_risk_idx],\n",
            "    base_values   = explainer.expected_value,\n",
            "    data          = X_sample.iloc[high_risk_idx].values,\n",
            "    feature_names = X_sample.columns.tolist()\n",
            ")\n",
            "\n",
            "plt.figure(figsize=(10, 7))\n",
            "shap.plots.waterfall(explanation_high, max_display=15, show=False)\n",
            "plt.title(f'HIGH-RISK Loan — Predicted Probability: {p_sample[high_risk_idx]:.3f}  |  '\n",
            "          f'Actual: {\"DEFAULT\" if y_sample.iloc[high_risk_idx]==1 else \"PAID\"}',\n",
            "          fontsize=10, color='#C0392B', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/shap_waterfall_high.png', dpi=120, bbox_inches='tight')\n",
            "plt.show()\n",
            "print('Saved: shap_waterfall_high.png')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Low-risk waterfall ───────────────────────────────────────────────────────\n",
            "explanation_low = shap.Explanation(\n",
            "    values        = shap_values[low_risk_idx],\n",
            "    base_values   = explainer.expected_value,\n",
            "    data          = X_sample.iloc[low_risk_idx].values,\n",
            "    feature_names = X_sample.columns.tolist()\n",
            ")\n",
            "\n",
            "plt.figure(figsize=(10, 7))\n",
            "shap.plots.waterfall(explanation_low, max_display=15, show=False)\n",
            "plt.title(f'LOW-RISK Loan — Predicted Probability: {p_sample[low_risk_idx]:.3f}  |  '\n",
            "          f'Actual: {\"DEFAULT\" if y_sample.iloc[low_risk_idx]==1 else \"PAID\"}',\n",
            "          fontsize=10, color='#27AE60', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/shap_waterfall_low.png', dpi=120, bbox_inches='tight')\n",
            "plt.show()\n",
            "print('Saved: shap_waterfall_low.png')"
        ]
    },
    # ── 5. Dependence Plot ──────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Dependence Plots — How Feature Values Drive Predictions\n",
            "Shows the relationship between a feature's value (X-axis) and its SHAP value (Y-axis).\n",
            "The colour shows the value of the most interacting feature.\n",
            "This reveals non-linear relationships and interaction effects the model learned."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Find top feature by mean |SHAP|\n",
            "top_features = mean_abs_shap.sort_values(ascending=False).index.tolist()\n",
            "feat1 = top_features[0]\n",
            "feat2 = top_features[1]\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Feature 1\n",
            "shap.dependence_plot(\n",
            "    feat1, shap_values, X_sample,\n",
            "    ax=axes[0], show=False, alpha=0.4\n",
            ")\n",
            "axes[0].set_title(f'Dependence Plot: {feat1}\\n'\n",
            "                  f'X = feature value | Y = SHAP impact on default probability',\n",
            "                  fontsize=10)\n",
            "\n",
            "# Feature 2\n",
            "shap.dependence_plot(\n",
            "    feat2, shap_values, X_sample,\n",
            "    ax=axes[1], show=False, alpha=0.4\n",
            ")\n",
            "axes[1].set_title(f'Dependence Plot: {feat2}\\n'\n",
            "                  f'X = feature value | Y = SHAP impact on default probability',\n",
            "                  fontsize=10)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/shap_dependence.png', dpi=120, bbox_inches='tight')\n",
            "plt.show()\n",
            "print(f'Saved: shap_dependence.png  (features: {feat1}, {feat2})')"
        ]
    },
    # ── 6. High vs Low Risk Comparison ─────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. High-Risk vs Low-Risk Loan Profile Comparison\n",
            "Side-by-side comparison of the actual feature values for the two loans we explained above.\n",
            "This makes the waterfall plots concrete and easy to interpret."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "high_loan = X_sample.iloc[high_risk_idx]\n",
            "low_loan  = X_sample.iloc[low_risk_idx]\n",
            "\n",
            "# Show top 15 features by mean |SHAP|\n",
            "top15_feats = mean_abs_shap.sort_values(ascending=False).head(15).index.tolist()\n",
            "\n",
            "comparison = pd.DataFrame({\n",
            "    'Feature':          top15_feats,\n",
            "    'High-Risk Value':  [round(high_loan[f], 3) for f in top15_feats],\n",
            "    'Low-Risk Value':   [round(low_loan[f],  3) for f in top15_feats],\n",
            "    'SHAP (High-Risk)': [round(shap_values[high_risk_idx][X_sample.columns.get_loc(f)], 4) for f in top15_feats],\n",
            "    'SHAP (Low-Risk)':  [round(shap_values[low_risk_idx][X_sample.columns.get_loc(f)],  4) for f in top15_feats],\n",
            "})\n",
            "\n",
            "print(f'HIGH-RISK loan  — predicted prob: {p_sample[high_risk_idx]:.3f}  |  actual: {\"DEFAULT\" if y_sample.iloc[high_risk_idx]==1 else \"PAID\"}')\n",
            "print(f'LOW-RISK  loan  — predicted prob: {p_sample[low_risk_idx]:.3f}  |  actual: {\"DEFAULT\" if y_sample.iloc[low_risk_idx]==1 else \"PAID\"}')\n",
            "print()\n",
            "print(comparison.to_string(index=False))"
        ]
    },
    # ── 7. Key Takeaways ────────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. Key Takeaways"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "top5 = mean_abs_shap.sort_values(ascending=False).head(5)\n",
            "\n",
            "print('=' * 60)\n",
            "print('         SHAP EXPLAINABILITY — KEY TAKEAWAYS')\n",
            "print('=' * 60)\n",
            "print()\n",
            "print('TOP 5 FEATURES BY MEAN |SHAP|:')\n",
            "for rank, (feat, val) in enumerate(top5.items(), 1):\n",
            "    print(f'  {rank}. {feat:30s}  mean |SHAP| = {val:.5f}')\n",
            "print()\n",
            "print('WHAT SHAP ADDS OVER STANDARD FEATURE IMPORTANCE:')\n",
            "print('  - Direction: does a HIGH value of this feature increase or decrease default risk?')\n",
            "print('  - Magnitude: by exactly how much does each feature shift the probability?')\n",
            "print('  - Individual: explains every single loan prediction, not just the model overall')\n",
            "print('  - Interactions: dependence plots reveal how two features interact')\n",
            "print()\n",
            "print('BUSINESS USE:')\n",
            "print('  - Loan officer can see EXACTLY why a loan was flagged')\n",
            "print('  - Applicant can be told which factors hurt their application')\n",
            "print('  - Regulatory compliance: model decisions are fully explainable')\n",
            "print('=' * 60)"
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

with open('e:/Projects/LendingClub_Loan/notebooks/07_shap_explainability.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Created 07_shap_explainability.ipynb successfully')
