import networkx as nx            # Used to manipulate the Bayesian Network(BN) as a directed graph (DAG)
import matplotlib.pyplot as plt  # Used to visualize the BN
import pandas as pd
from collections import Counter
from google.colab import drive   # Used to mount Google Drive
from typing import Any           # type annotations

# pgmpy libraries for BBN construction
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, MaximumLikelihoodEstimator, K2,GES,MmhcEstimator
from pgmpy.inference import VariableElimination

import pandas as pd
from google.colab import drive

# Connect to Google Drive
drive.mount('/content/drive')

# Load the original CSV file
file_path = '/content/drive/MyDrive/Lung_Cancer_Data/Cleaned_Lung_Cancer_dataset.csv'
df = pd.read_csv(file_path)

# 1. Merge Cancer Cell Type categories
# Reason: There are many subtypes of NSCLC (non-small cell lung carcinoma) and rare types
# in the original dataset. Merging subtypes into broader categories reduces sparsity,
# ensures each category has enough samples for statistical modeling,
# and simplifies the Bayesian network structure.
cell_type_map = {
    'NSCLC-Adenocarcinoma, NOS': 'NSCLC-Adenocarcinoma',
    'NSCLC-lepidic predominant adenocarcinoma': 'NSCLC-Adenocarcinoma',
    'NSCLC-Solid Predominant Adenocarcinoma': 'NSCLC-Adenocarcinoma',
    'NSCLC-Papillary Predominant Adenocarcinoma': 'NSCLC-Adenocarcinoma',
    'NSCLC-Adenocarcinoma w/ Mixed Subtypes': 'NSCLC-Adenocarcinoma',
    'NSCLC-Squamous, NOS': 'NSCLC-Squamous',
    'NSCLC-Adenosquamous Carcinoma': 'Other NSCLC',
    'NSCLC-Large Cell Neuroendocrine Carcinoma': 'Other NSCLC',
    'NSCLC-NOS': 'Other NSCLC',
    'Typical Carcinoid': 'Neuroendocrine / Carcinoid',
    'Neuroendocrine Carcinoma': 'Neuroendocrine / Carcinoid',
    'Malignant Tumor Cells, NOS': 'Malignant / Carcinoma NOS',
    'Carcinoma, NOS': 'Malignant / Carcinoma NOS',
    'Other / Rare': 'Other / Rare',
    'Not Lung Cancer': 'Not Lung Cancer'
}
df['Cancer Cell Type'] = df['Cancer Cell Type'].map(cell_type_map)

# 2. Merge Regional Lymph Node Involvement categories
# Reason: Original dataset has multiple detailed lymph node statuses.
# For modeling purposes, simplifying to 'Nodal Involvement', 'No Nodal Involvement',
# and 'Unknown / NA' reduces sparsity, avoids overfitting, and makes causal relationships
# more interpretable in Bayesian networks.
node_map = {
    'No Nodal Involvement': 'No Nodal Involvement',
    'Mediastinal Ipsilateral': 'Nodal Involvement',
    'Bilateral / Contralateral': 'Nodal Involvement',
    'Ipsilateral Nodes Only': 'Nodal Involvement',
    'Ipsilateral / Contralateral': 'Nodal Involvement',
    'Node Involvement, NOS': 'Nodal Involvement',
    'Regional Node Involvement, NOS': 'Nodal Involvement',
    'Unknown Node Status': 'Unknown / NA',
    'Not Applicable': 'Unknown / NA'
}
df['Regional Lymph Node Involvement'] = df['Regional Lymph Node Involvement'].map(node_map)

# 3. Bin Days from Diagnosis to Treatment
# Reason: Original 'Days from Diagnosis' is continuous with high variability.
# Binning into categories ('<=3 months', '<=1 year', '>1 year') simplifies the data,
# reduces noise, and allows the Bayesian network to capture temporal patterns without
# being affected by outliers or extreme values.
def days_bin(x):
    if x <= 90:   # 3 months or less
        return '<=3 months'
    elif x <= 365:
        return '<=1 year'
    else:
        return '>1 year'
df['Days from Diagnosis to Treatment'] = df['Days from Diagnosis to Treatment'].apply(days_bin)

# 4. Merge Metastatic Spread categories
# Reason: Original dataset lists multiple specific metastatic sites.
# Combining into 'Distant Metastasis', 'No Distant Metastasis', and 'Unknown / NA'
# reduces sparsity and allows clearer modeling of the causal impact of metastasis on
# treatment outcomes and survival.
metastatic_map = {
    'No Distant Metastasis': 'No Distant Metastasis',
    'Lung': 'Distant Metastasis',
    'Distant Lymph Node': 'Distant Metastasis',
    'Other Specified Distant Metastasis': 'Distant Metastasis',
    'Distant Metastasis, NOS': 'Distant Metastasis',
    'Unknown': 'Unknown / NA',
    'Entry Error': 'Unknown / NA'
}
df['Metastatic Spread'] = df['Metastatic Spread'].map(metastatic_map)

# 5. Merge Tumor Laterality categories
# Reason: Original laterality has multiple levels (Right, Left, Bilateral, etc.).
# Combining Right and Left simplifies the analysis while still retaining the important distinction
# of Bilateral vs. Single side. 'Unknown / NA' handles missing or inapplicable cases.
laterality_map = {
    'Right': 'Right / Left',
    'Left': 'Right / Left',
    'Bilateral': 'Bilateral',
    'Unspecified': 'Unknown / NA',
    'Only One Side, NOS': 'Unknown / NA',
    'Inapplicable': 'Unknown / NA'
}
df['Tumor Laterality'] = df['Tumor Laterality'].map(laterality_map)

# 6. Merge Extent of Regional Lymph Node Surgery categories
# Reason: Multiple surgery types exist, some rare. Grouping into 'Nodes Removed / Dissection',
# 'Biopsy Only', and 'Unknown / NA' reduces sparsity and simplifies modeling the effect of surgery.
surgery_map = {
    '4+ Nodes Removed': 'Nodes Removed / Dissection',
    '1-3 Nodes Removed': 'Nodes Removed / Dissection',
    'Sentinel Biopsy + Dissection': 'Nodes Removed / Dissection',
    'Sentinel Biopsy only': 'Nodes Removed / Dissection',
    'Node(s) Removed, NOS': 'Nodes Removed / Dissection',
    'Regional Biopsy/aspiration only': 'Biopsy Only',
    'Unknown/Inapplicable': 'Unknown / NA'
}
df['Extent of Regional Lymph Node Surgery'] = df['Extent of Regional Lymph Node Surgery'].map(surgery_map)


# 7. Merge Treatment Plan categories
# Reason: Original treatment plan is detailed with combinations of surgery, chemo, radiotherapy.
# Merging into broader categories reduces sparsity, prevents overfitting, and allows the Bayesian network
# to capture meaningful causal relationships without too many rare combinations.
treatment_map = {
    'Surgery': 'Surgery',
    'Radiotherapy': 'Radiotherapy',
    'Radiotherapy after Surgery': 'Radiotherapy',
    'Chemotherapy': 'Chemotherapy / Combined',
    'Chemotherapy after Surgery': 'Chemotherapy / Combined',
    'Chemotherapy and Radiotherapy': 'Chemotherapy / Combined',
    'Chemotherapy and Radiotherapy after Surgery': 'Chemotherapy / Combined',
    'Radiotherapy and Chemotherapy before Surgery': 'Chemotherapy / Combined',
    'Chemotherapy before Surgery': 'Chemotherapy / Combined',
    'Refused Treatment': 'Refused',
    'Other': 'Other',
    'Unknown Treatment': 'Unknown / NA'
}
df['Treatment Plan'] = df['Treatment Plan'].map(treatment_map)

# 8. Save the cleaned and recategorized CSV
output_path = '/content/drive/MyDrive/Lung_Cancer_Data/Recategorized_Cleaned_Lung_Cancer_dataset.csv'
df.to_csv(output_path, index=False)

print("Recategorized CSV file successfully created:", output_path)
print(df.describe())

#1. Mount Google Drive and Load Dataset
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/Lung_Cancer_Data/Recategorized_Cleaned_Lung_Cancer_dataset.csv'
df = pd.read_csv(file_path)
print(df[:10])


# 2. Preprocessing: remove spaces in column names and convert to strings
df.columns = [col.replace(' ', '_') for col in df.columns]
for col in df.columns:
    df[col] = df[col].astype(str)

# 3 Perform Train-Validation Split
from sklearn.model_selection import train_test_split
print("Confirming DataFrame `df` is ready for train-validation split:")
df.info()
print("\nDataFrame `df` is confirmed to be ready. Column names are cleaned, and all columns are of string type as per previous preprocessing steps.")

train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
print(len(df))


# 4. Structure Learning
# Uses Hill Climbing algorithm to find the optimal Directed Acyclic Graph (DAG) based on BIC score.
print("Learning structure... (this may take a moment)")
hc = HillClimbSearch(train_data)
best_model = hc.estimate(scoring_method=BIC(train_data))
consistent_edges = list(best_model.edges())

#5. Construct Bayesian Belief Network (BBN)
# Initialize the Bayesian Network with the learned structure.
bbn_model = DiscreteBayesianNetwork(consistent_edges)

# Add isolated nodes (nodes with no edges) to the model if any
bbn_model.add_nodes_from(train_data.columns)

print("\nLearned Edges:")
for edge in consistent_edges:
    print(edge)


# 5. Parameter Learning
# Calculate Conditional Probability Tables (CPTs) using Maximum Likelihood Estimation.
bbn_model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# Validate the model (Should return True)
print(f"\nBBN Model Validation: {bbn_model.check_model()}")

# 6. Visualization
plt.figure(figsize=(15, 10))
G = nx.DiGraph()
G.add_edges_from(consistent_edges)
G.add_nodes_from(train_data.columns)

pos = circular_layout(G)

nx.draw(
    G, pos, with_labels=True, node_size=2500, node_color='skyblue',
    arrowsize=20, font_size=9, font_weight='bold', edge_color='gray'
)
plt.title("Learned Bayesian Belief Network Structure", fontsize=15)
plt.show()


#7. Probability Inference
# BBN allows us to calculate the probability of an outcome given specific evidence.
print("\n--- Inference Results ---")
infer = VariableElimination(bbn_model)


def query_to_df(q):
    """
    Convert pgmpy query result (DiscreteFactor) into a pandas DataFrame.
    Output columns: target variable states and their corresponding probabilities.
    """
    var = q.variables[0]
    states = q.state_names[var]
    probs = q.values

    out = pd.DataFrame({var: states, "prob": probs})
    return out

def query_to_df(q):
    """
    Convert pgmpy query result (DiscreteFactor) into a pandas DataFrame similar to bnlearn style.
    Output column names: variable states + 'prob'

    :type q: pgmpy.factors.discrete.DiscreteFactor
    :rtype: pandas.DataFrame
    """
    var = q.variables[0]
    states = q.state_names[var]
    probs = q.values

    out = pd.DataFrame({var: states, "prob": probs})
    return out

target = "Cause_of_Death"

q1 = infer.query(variables=[target], evidence={"Metastatic_Spread": "Distant Metastasis"})
q1.df = query_to_df(q1)
print("q1 = P(Treatment_Plan | Metastatic_Spread='Distant Metastasis')")
print(q1.df)

q2 = infer.query(
    variables=[target],
    evidence={
        "Tumor_Laterality": "Bilateral",
        "Regional_Lymph_Node_Involvement": "Nodal Involvement"
    }
)
q2.df = query_to_df(q2)
print("\nq2 = P(Treatment_Plan | Tumor_Laterality='Bilateral', Regional_Lymph_Node_Involvement='Nodal Involvement')")
print(q2.df)

q3 = infer.query(
    variables=[target],
    evidence={"Days_from_Diagnosis_to_Treatment": "<=3 months"}
)
q3.df = query_to_df(q3)
print("\nq3 = P(Treatment_Plan | Days_from_Diagnosis_to_Treatment='<=3 months')")
print(q3.df)

q4 = infer.query(
    variables=[target],
    evidence={
        "Age": "50-79 years",
        "Cancer_Cell_Type": "NSCLC-Adenocarcinoma",
        "Metastatic_Spread": "No Distant Metastasis",
        "Regional_Lymph_Node_Involvement": "Nodal Involvement",
        "Tumor_Laterality": "Right / Left",
        "Treatment_Plan": "Chemotherapy / Combined",
        "Days_from_Diagnosis_to_Treatment": "<=3 months"
    }
)

q4.df = query_to_df(q4)

print(
    "\nq4 = P(Cause_of_Death | "
    "Age='50-79 years', "
    "Cancer_Cell_Type='NSCLC-Adenocarcinoma', "
    "Metastatic_Spread='No Distant Metastasis', "
    "Regional_Lymph_Node_Involvement='Nodal Involvement', "
    "Tumor_Laterality='Right / Left', "
    "Treatment_Plan='Chemotherapy / Combined', "
    "Days_from_Diagnosis_to_Treatment='<=3 months'"
)
print(q4.df)

import timeit
import numpy as np
from sklearn.metrics import accuracy_score
from pgmpy.inference import VariableElimination
from functools import lru_cache



target_col = 'Cause_of_Death'
X_test = test_data.drop(columns=[target_col])
y_true = test_data[target_col]

infer = VariableElimination(bbn_model)

# ------------------  Basic query loop ------------------
def basic_query():
    y_pred_basic = []

    for _, row in X_test.iterrows():
        q = infer.query(variables = [target_col], evidence=row.to_dict(), show_progress=True)
        pred = q.state_names[target_col][np.argmax(q.values)]
        y_pred_basic.append(pred)

    return y_pred_basic

@lru_cache(maxsize = 20_000)
def cached_predict(evidence_key):
    y_pred_lru = []

    q = infer.query(variables=[target_col], evidence=dict(evidence_key), show_progress=False)
    pred = q.state_names[target_col][np.argmax(q.values)]
    y_pred_lru.append(pred)
    return y_pred_lru

def evidence_key(row):
    return tuple(sorted(row.items())) # dict -> tuple(immutable) = hashable

def predict_row(row):
    return cached_predict(evidence_key(row.to_dict()))

def lru_cache_query():
    return X_test.apply(predict_row, axis=1)

#Time the execution of 'basic_query' 1 times
y_pred_basic = basic_query()
accuracy = accuracy_score(y_true, y_pred_basic)
print(f"Accuracy for Basic query: {accuracy:.4f}")
basic_execution_time = timeit.timeit(stmt="basic_query()", setup="from __main__ import basic_query", number=1)
print(f"Average time per run for basic_query: {basic_execution_time/1 :.6f} seconds")



# Time the execution of 'lru_cache_query' 1 times
y_pred_lru = basic_query()
print(cached_predict.cache_info())
accuracy = accuracy_score(y_true, y_pred_lru)
print(f"Accuracy for LRU cache: {accuracy:.4f}")
lru_execution_time = timeit.timeit(stmt="lru_cache_query()", setup="from __main__ import lru_cache_query", number = 1)
print(f"Averatge time per run for lru_cache_query: {lru_execution_time/1 :.6f} seconds")