import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder


# Connect to the SQLite database
conn = sqlite3.connect('db\MasterDatabase.db')
cursor = conn.cursor()

# Query to get all values from the solubility table
query = """SELECT 
compound_id, 
solvent_1,
solvent_2,
solvent_1_weight_fraction,
temperature,
solubility_g_g
FROM solubility"""
df_db = pd.read_sql_query(query, conn)

# Load the Excel file
excel_file = 'src\\2-featureEngineering\\descriptorsAPIs.xlsx'
df_excel = pd.read_excel(excel_file)

# Merge the two dataframes based on PUBCHEM_COMPOUND_CID column in the excel
merged_df = pd.merge(df_db, df_excel, left_on='compound_id', right_on='PUBCHEM_COMPOUND_CID')
# One hot encode the solvent_1 and solvent_2 columns as a combination
merged_df['solvent_combination'] = merged_df['solvent_1'].astype(str) + '_' + merged_df['solvent_2'].astype(str)
encoder = TargetEncoder()
encoded_solvents = encoder.fit_transform(merged_df[['solvent_combination']])
# Drop the columns solvent_1, solvent_2 and compound_id
one_hot_encoded_df = encoded_solvents.drop(columns=['solvent_1', 'solvent_2', 'compound_id','mol'])

columns_to_drop = [col for col in one_hot_encoded_df.columns if col.startswith('PUBCHEM_')]
one_hot_encoded_df = one_hot_encoded_df.drop(columns=columns_to_drop)
one_hot_encoded_df = one_hot_encoded_df.dropna(axis=1, how='any')


X = one_hot_encoded_df.drop(columns=['solubility_g_g'])
y = one_hot_encoded_df['solubility_g_g']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Perform PCA
pca = PCA(n_components=0.8)  # Keep 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Display the explained variance ratio of each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio of each principal component:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Principal Component {i+1}: {ratio:.4f}")

# Display the most relevant principal components (those with variance ratio above a threshold)
threshold = 0.01  # Example threshold for relevance
print("\nMost relevant principal components (variance ratio > 0.01):")
for i, (ratio, component) in enumerate(zip(explained_variance_ratio, pca.components_)):
    if ratio > threshold:
        print(f"Principal Component {i+1} (Variance Ratio: {ratio:.4f}):")
        relevant_columns = [col for col, val in zip(one_hot_encoded_df.columns, component) if abs(val) > 0.1]  # Example threshold for relevance in components
        print("Relevant columns:", relevant_columns)

# Close the database connection
conn.close()

# Add this to your code to analyze components vs variance
def analyze_pca_components(X_scaled):
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = range(1, len(cumulative_variance) + 1)
    
    print("\nCumulative Variance Explained:")
    for n, var in zip(n_components, cumulative_variance):
        print(f"Components: {n}, Variance Explained: {var:.4f}")
        if var > 0.8 and var < 0.95:  # Key thresholds
            print(f"*** Potential sweet spot at {n} components ***")
            
analyze_pca_components(X_scaled)