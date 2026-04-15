import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# EDA - Hypothesis Validation\n", "Validating H1-H5 against the filtered LendingClub dataset."]
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
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "\n",
            "df = pd.read_parquet('../data/loans_filtered.parquet')\n",
            "print('Shape:', df.shape)\n",
            "print('Target counts:')\n",
            "print(df['target'].value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## H1: Higher DTI -> Higher Default"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "df.groupby('target')['dti'].median().plot(kind='bar', ax=axes[0], color=['steelblue','tomato'])\n",
            "axes[0].set_title('Median DTI by Target')\n",
            "axes[0].set_xlabel('Target (0=Paid, 1=Default)')\n",
            "axes[0].set_ylabel('Median DTI')\n",
            "axes[0].tick_params(axis='x', rotation=0)\n",
            "\n",
            "for t, label, color in [(0, 'Fully Paid', 'steelblue'), (1, 'Default', 'tomato')]:\n",
            "    df[df['target']==t]['dti'].dropna().clip(0, 60).plot.kde(ax=axes[1], label=label, color=color)\n",
            "axes[1].set_title('DTI Distribution by Target')\n",
            "axes[1].set_xlabel('DTI (clipped at 60)')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/h1_dti.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print('Median DTI:')\n",
            "print(df.groupby('target')['dti'].median())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## H2: Shorter Credit History -> Higher Default"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')\n",
            "df['credit_history_years'] = (pd.Timestamp('2018-12-31') - df['earliest_cr_line']).dt.days / 365\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "df.groupby('target')['credit_history_years'].median().plot(kind='bar', ax=axes[0], color=['steelblue','tomato'])\n",
            "axes[0].set_title('Median Credit History (Years) by Target')\n",
            "axes[0].set_xlabel('Target (0=Paid, 1=Default)')\n",
            "axes[0].set_ylabel('Years')\n",
            "axes[0].tick_params(axis='x', rotation=0)\n",
            "\n",
            "for t, label, color in [(0, 'Fully Paid', 'steelblue'), (1, 'Default', 'tomato')]:\n",
            "    df[df['target']==t]['credit_history_years'].dropna().plot.kde(ax=axes[1], label=label, color=color)\n",
            "axes[1].set_title('Credit History Distribution by Target')\n",
            "axes[1].set_xlabel('Credit History (Years)')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/h2_credit_history.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print('Median Credit History (years):')\n",
            "print(df.groupby('target')['credit_history_years'].median())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## H3: Certain Loan Purposes -> Higher Risk"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "purpose_risk = df.groupby('purpose')['target'].mean().sort_values(ascending=False)\n",
            "purpose_counts = df.groupby('purpose')['target'].count()\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "\n",
            "purpose_risk.plot(kind='bar', ax=axes[0], color='tomato')\n",
            "axes[0].set_title('Default Rate by Loan Purpose')\n",
            "axes[0].set_ylabel('Default Rate')\n",
            "axes[0].tick_params(axis='x', rotation=45)\n",
            "\n",
            "purpose_counts.sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='steelblue')\n",
            "axes[1].set_title('Loan Count by Purpose')\n",
            "axes[1].set_ylabel('Count')\n",
            "axes[1].tick_params(axis='x', rotation=45)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/h3_purpose.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print('Default rate by purpose:')\n",
            "print(purpose_risk.round(3))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## H4: Lower Income -> Higher Default"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "income_cap = df['annual_inc'].quantile(0.99)\n",
            "df['annual_inc_capped'] = df['annual_inc'].clip(0, income_cap)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "df.groupby('target')['annual_inc_capped'].median().plot(kind='bar', ax=axes[0], color=['steelblue','tomato'])\n",
            "axes[0].set_title('Median Annual Income by Target')\n",
            "axes[0].set_xlabel('Target (0=Paid, 1=Default)')\n",
            "axes[0].set_ylabel('Median Income ($)')\n",
            "axes[0].tick_params(axis='x', rotation=0)\n",
            "\n",
            "for t, label, color in [(0, 'Fully Paid', 'steelblue'), (1, 'Default', 'tomato')]:\n",
            "    df[df['target']==t]['annual_inc_capped'].dropna().plot.kde(ax=axes[1], label=label, color=color)\n",
            "axes[1].set_title('Income Distribution by Target')\n",
            "axes[1].set_xlabel('Annual Income (capped at 99th pct)')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/h4_income.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print('Median Annual Income:')\n",
            "print(df.groupby('target')['annual_inc_capped'].median())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## H5: More Derogatory Marks -> Higher Default"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "pub_rec_rate = df.groupby('pub_rec')['target'].mean().head(8)\n",
            "pub_rec_rate.plot(kind='bar', ax=axes[0], color='tomato')\n",
            "axes[0].set_title('Default Rate by # Public Records')\n",
            "axes[0].set_xlabel('Number of Public Records')\n",
            "axes[0].set_ylabel('Default Rate')\n",
            "axes[0].tick_params(axis='x', rotation=0)\n",
            "\n",
            "df.groupby('target')['pub_rec'].mean().plot(kind='bar', ax=axes[1], color=['steelblue','tomato'])\n",
            "axes[1].set_title('Mean Public Records by Target')\n",
            "axes[1].set_xlabel('Target (0=Paid, 1=Default)')\n",
            "axes[1].set_ylabel('Mean # Public Records')\n",
            "axes[1].tick_params(axis='x', rotation=0)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../data/h5_derogatory.png', dpi=100)\n",
            "plt.show()\n",
            "\n",
            "print('Mean pub_rec by target:')\n",
            "print(df.groupby('target')['pub_rec'].mean().round(3))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Hypothesis Summary"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "summary = {\n",
            "    'Hypothesis': ['H1: Higher DTI', 'H2: Shorter Credit History', 'H3: Loan Purpose', 'H4: Lower Income', 'H5: More Derogatory Marks'],\n",
            "    'Feature': ['dti', 'credit_history_years', 'purpose', 'annual_inc', 'pub_rec'],\n",
            "    'Default_val': [\n",
            "        round(df[df['target']==1]['dti'].median(), 2),\n",
            "        round(df[df['target']==1]['credit_history_years'].median(), 2),\n",
            "        'See chart',\n",
            "        round(df[df['target']==1]['annual_inc_capped'].median(), 2),\n",
            "        round(df[df['target']==1]['pub_rec'].mean(), 3)\n",
            "    ],\n",
            "    'Paid_val': [\n",
            "        round(df[df['target']==0]['dti'].median(), 2),\n",
            "        round(df[df['target']==0]['credit_history_years'].median(), 2),\n",
            "        'See chart',\n",
            "        round(df[df['target']==0]['annual_inc_capped'].median(), 2),\n",
            "        round(df[df['target']==0]['pub_rec'].mean(), 3)\n",
            "    ]\n",
            "}\n",
            "print(pd.DataFrame(summary).to_string(index=False))"
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

with open('e:/Projects/LendingClub_Loan/notebooks/02_eda.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Created 02_eda.ipynb successfully')
