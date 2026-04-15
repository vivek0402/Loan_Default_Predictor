import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Preprocessing\n", "Feature selection, cleaning, encoding, scaling, train/test split."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "df = pd.read_parquet('../data/loans_filtered.parquet')\n",
            "df = df.copy()  # defragment\n",
            "print('Loaded shape:', df.shape)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 1: Feature Selection\n",
                   "Keep only features relevant to our hypotheses + known credit risk predictors."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "FEATURES = [\n",
            "    # Hypothesis features\n",
            "    'dti',                  # H1\n",
            "    'earliest_cr_line',     # H2 - will engineer\n",
            "    'purpose',              # H3\n",
            "    'annual_inc',           # H4\n",
            "    'pub_rec',              # H5\n",
            "\n",
            "    # Strong credit risk predictors\n",
            "    'loan_amnt',\n",
            "    'int_rate',\n",
            "    'installment',\n",
            "    'grade',\n",
            "    'sub_grade',\n",
            "    'emp_length',\n",
            "    'home_ownership',\n",
            "    'verification_status',\n",
            "    'open_acc',\n",
            "    'revol_bal',\n",
            "    'revol_util',\n",
            "    'total_acc',\n",
            "    'mort_acc',\n",
            "    'delinq_2yrs',\n",
            "    'inq_last_6mths',\n",
            "\n",
            "    # Target\n",
            "    'target'\n",
            "]\n",
            "\n",
            "df = df[FEATURES].copy()\n",
            "print('Shape after feature selection:', df.shape)\n",
            "print('\\nMissing values:')\n",
            "print(df.isnull().sum()[df.isnull().sum() > 0])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 2: Feature Engineering"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# H2: Convert earliest_cr_line to credit_history_years\n",
            "df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')\n",
            "df['credit_history_years'] = (pd.Timestamp('2018-12-31') - df['earliest_cr_line']).dt.days / 365\n",
            "df = df.drop(columns=['earliest_cr_line'])\n",
            "\n",
            "# H5: Convert pub_rec to binary flag\n",
            "df['has_derogatory'] = (df['pub_rec'] > 0).astype(int)\n",
            "df = df.drop(columns=['pub_rec'])\n",
            "\n",
            "# Cap annual_inc at 99th percentile\n",
            "income_cap = df['annual_inc'].quantile(0.99)\n",
            "df['annual_inc'] = df['annual_inc'].clip(0, income_cap)\n",
            "\n",
            "# emp_length: convert to numeric\n",
            "emp_map = {\n",
            "    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,\n",
            "    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,\n",
            "    '8 years': 8, '9 years': 9, '10+ years': 10\n",
            "}\n",
            "df['emp_length'] = df['emp_length'].map(emp_map)\n",
            "\n",
            "# grade: convert to numeric (A=1, B=2, ..., G=7)\n",
            "grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}\n",
            "df['grade'] = df['grade'].map(grade_map)\n",
            "\n",
            "print('Shape after engineering:', df.shape)\n",
            "print('\\nNew features added: credit_history_years, has_derogatory')\n",
            "print('Engineered: emp_length, grade')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 3: Handle Missing Values"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Missing before imputation:')\n",
            "missing = df.isnull().sum()\n",
            "print(missing[missing > 0])\n",
            "\n",
            "# Numeric: fill with median\n",
            "num_cols = df.select_dtypes(include='number').columns.tolist()\n",
            "num_cols = [c for c in num_cols if c != 'target']\n",
            "df[num_cols] = df[num_cols].fillna(df[num_cols].median())\n",
            "\n",
            "# Categorical: fill with mode\n",
            "cat_cols = df.select_dtypes(include='object').columns.tolist()\n",
            "for col in cat_cols:\n",
            "    df[col] = df[col].fillna(df[col].mode()[0])\n",
            "\n",
            "print('\\nMissing after imputation:')\n",
            "print(df.isnull().sum().sum(), 'nulls remaining')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 4: Encode Categorical Features"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Drop sub_grade (redundant with grade)\n",
            "df = df.drop(columns=['sub_grade'])\n",
            "\n",
            "# One-hot encode low-cardinality categoricals\n",
            "cat_cols = df.select_dtypes(include='object').columns.tolist()\n",
            "print('Categorical columns to encode:', cat_cols)\n",
            "\n",
            "df = pd.get_dummies(df, columns=cat_cols, drop_first=True)\n",
            "\n",
            "print('Shape after encoding:', df.shape)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 5: Train/Test Split"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X = df.drop(columns=['target'])\n",
            "y = df['target']\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y,\n",
            "    test_size=0.2,\n",
            "    random_state=42,\n",
            "    stratify=y  # preserve class ratio\n",
            ")\n",
            "\n",
            "print('Train shape:', X_train.shape)\n",
            "print('Test shape: ', X_test.shape)\n",
            "print('\\nTrain target split:')\n",
            "print(y_train.value_counts(normalize=True).mul(100).round(2))\n",
            "print('\\nTest target split:')\n",
            "print(y_test.value_counts(normalize=True).mul(100).round(2))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 6: Scale Numeric Features"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "scaler = StandardScaler()\n",
            "\n",
            "# Only scale numeric columns (not binary/dummy columns)\n",
            "scale_cols = [c for c in X_train.columns if X_train[c].nunique() > 2]\n",
            "\n",
            "X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])\n",
            "X_test[scale_cols] = scaler.transform(X_test[scale_cols])  # use train stats only\n",
            "\n",
            "print('Scaled', len(scale_cols), 'numeric columns')\n",
            "print('Sample scaled values (X_train first row):')\n",
            "print(X_train[scale_cols[:5]].iloc[0].round(3))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 7: Save Processed Data"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pickle\n",
            "\n",
            "X_train.to_parquet('../data/X_train.parquet', index=False)\n",
            "X_test.to_parquet('../data/X_test.parquet', index=False)\n",
            "y_train.to_frame().to_parquet('../data/y_train.parquet', index=False)\n",
            "y_test.to_frame().to_parquet('../data/y_test.parquet', index=False)\n",
            "\n",
            "# Save scaler for use in production\n",
            "with open('../data/scaler.pkl', 'wb') as f:\n",
            "    pickle.dump(scaler, f)\n",
            "\n",
            "print('Saved:')\n",
            "print('  X_train.parquet:', X_train.shape)\n",
            "print('  X_test.parquet: ', X_test.shape)\n",
            "print('  y_train.parquet:', y_train.shape)\n",
            "print('  y_test.parquet: ', y_test.shape)\n",
            "print('  scaler.pkl')\n",
            "print('\\nFeature count:', X_train.shape[1])\n",
            "print('\\nAll features:')\n",
            "print(list(X_train.columns))"
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

with open('e:/Projects/LendingClub_Loan/notebooks/03_preprocessing.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Created 03_preprocessing.ipynb successfully')
