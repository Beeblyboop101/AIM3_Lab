import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

# 1. DATA PREPARATION
print("Loading full dataset")
df = pd.read_csv('cancer_dataset.csv')

df = df[df['Cause of Death'].isin(['Alive or Not Cancer-related', 'Cancer'])].copy()
df['Survived_Cancer'] = (df['Cause of Death'] == 'Alive or Not Cancer-related').astype(int)

# 9 Feature Variables representing clinical and disease characteristics
features = [
    'Age', 'Tumor Location', 'Cancer Cell Type', 'Tumor Extent at Diagnosis',
    'Regional Lymph Node Involvement', 'Metastatic Spread', 'Tumor Laterality',
    'Extent of Regional Lymph Node Surgery', 'Treatment Plan'
]

# Modelling missing data explicitly as 'Unknown'
df[features] = df[features].fillna('Unknown/Missing')

# Encoding categorical variables for the Discrete BBN
label_encoders = {}
encoded_data = pd.DataFrame()
for col in features + ['Survived_Cancer']:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 2. FULL DATA SPLIT
train, test = train_test_split(encoded_data, test_size=0.2, random_state=42, stratify=encoded_data['Survived_Cancer'])

# 3. BBN STRUCTURE DEFINITION
# Fixed Naive Bayes Architecture: Root (Survival) -> Features
edges = [('Survived_Cancer', f) for f in features]
model_mle = DiscreteBayesianNetwork(edges)
model_bayesian = DiscreteBayesianNetwork(edges)

# 4. FULL PARAMETER LEARNING
print("\n--- Training Models on Full Dataset ---")

print("Learning parameters via MLE (Frequency-based)...")
model_mle.fit(train, estimator=MaximumLikelihoodEstimator)

print("Learning parameters via Bayesian Estimation (Dirichlet Priors, pseudo_counts=1)...")
model_bayesian.fit(train, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)

# Evaluation
def evaluate_full(model, test_df, name):
    print(f"\nEvaluating {name} on full test set ({len(test_df)} rows)...")
    X_test = test_df[features]
    y_true = test_df['Survived_Cancer'].values
    
    # Vectorized Prediction (DiscreteBayesianNetwork)
    y_pred_df = model.predict(X_test)
    y_pred = y_pred_df['Survived_Cancer'].values
    
    # Vectorized Probability (for AUC calculation)
    y_prob_df = model.predict_probability(X_test)
    
    # Identify the correct column for Class 1 (Survived)
    prob_col = [col for col in y_prob_df.columns if col.endswith('_1')][0]
    y_prob = y_prob_df[prob_col].values
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"Results for {name}: Accuracy = {acc:.4f}, AUC = {auc:.4f}")
    return acc, auc

# Run evaluations on all test data
mle_acc, mle_auc = evaluate_full(model_mle, test, "MLE Method")
bay_acc, bay_auc = evaluate_full(model_bayesian, test, "Bayesian Method")

# 6. SUMMARY 
summary = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC-ROC'],
    'MLE Estimation': [mle_acc, mle_auc],
    'Bayesian (Dirichlet)': [bay_acc, bay_auc]
})

print("\n--- FINAL RESEARCH METRICS ---")
print(summary.to_string(index=False))
