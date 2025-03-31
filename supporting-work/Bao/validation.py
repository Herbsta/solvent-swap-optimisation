import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Define colors
black = '#515265'
red = '#DD706E'
yellow = '#FAAF3A'
blue = '#3A93C2'

# Set data and model parameters
data_for = '_ExpMP'  # Options: '_ExpMP', '_PredMP', '_NoMP'
model_for = 'XGB'

# Load dataset
file_path = "refined_dataset" + data_for + ".csv"
data = pd.read_csv(file_path)

# Split data into train, test, and lab sets
train = data[data['Type'] == 'Train']
test = data[data['Type'] == 'Test']
lab = data[data['Type'] == 'Lab']

# Check for overlapping drug-solvent systems between train and test
overlapping_values = set(train['Drug-solvent system']).intersection(test['Drug-solvent system'])
print("Overlapping drug-solvent systems:", overlapping_values)

# Define data processing function
def data_processing(dataset):
    X = dataset.drop(['Type', 'Drug', 'Solvent_1', 'Solvent_2', 'Drug-solvent system', 'LogS', 'Class', 'Solubility (g/100g)'], axis=1)
    Y = dataset['LogS']
    G = dataset['Drug-solvent system']
    return X, Y, G

# Process datasets
X_train, Y_train, G_train = data_processing(train)
X_test, Y_test, G_test = data_processing(test)
X_lab, Y_lab, G_lab = data_processing(lab)

# Load hyperparameter screening results
pickle_file_path = 'BSCV_results' + data_for + '.pkl'
with open(pickle_file_path, 'rb') as handle:
    results = pickle.load(handle)

# Get best model from results
best_model = results[model_for]['best_estimator']

# Combine train and test data for final model training
literature_X_scaled = pd.concat([X_train, X_test], ignore_index=True)
literature_Y = pd.concat([Y_train, Y_test], ignore_index=True)

# Train model on combined literature data
best_model.fit(literature_X_scaled, literature_Y)

# Save the trained model
with open(model_for + '.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Load the model (demonstration of loading)
with open(model_for + '.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Predict on lab validation set
Y_pred = best_model.predict(X_lab)

# Create summary dataframe
summary = pd.DataFrame()
summary['Y_pred'] = Y_pred
summary['Y_exp'] = Y_lab.values
summary['Drug'] = lab['Drug'].values
summary['Temperature'] = lab['Temperature (K)'].values
summary['Solvent_mass_fraction'] = lab['Solvent_mass_fraction'].values

# Sort by drug, temperature, and solvent mass fraction
drug_order = ['Celecoxib', 'Acetaminophen', '3-Indolepropionic acid', 'Aspirin']
summary['Drug'] = pd.Categorical(summary['Drug'], categories=drug_order, ordered=True)
summary_sorted = summary.sort_values(by=['Drug', 'Temperature', 'Solvent_mass_fraction'])
summary_sorted = summary_sorted.reset_index(drop=True)

# Load original data for additional analysis
original_file_path = "Raw_dataset_dataset_20240705.csv"
original_data = pd.read_csv(original_file_path)
lab_X = original_data[original_data['Type'] == 'Lab']
literature_X = original_data[original_data['Type'] != 'Lab']

# Add experimental standard deviations
experiment_STD = [0.026547199, 0.007765518, 0.00609362, 0.011489668, 0.009077589, 0.004943748,
                 0.009819746, 0.007260406, 0.020924193, 0.008093572, 0.005818861, 0.009665102,
                 0.016553364, 0.008737235, 0.005338153, 0.021981287, 0.023570222, 0.012345814,
                 0.004954186, 0.007637141, 0.003202658, 0.012285287, 0.0108218, 0.005967770]

# Create plot dataframe
plot = lab_X[['Drug', 'Solvent_mass_fraction', 'Temperature (K)', 'LogS']]
plot['LogS_SD'] = experiment_STD
plot['Y_pred'] = summary_sorted['Y_pred'].values
plot['AE'] = abs(plot['LogS'] - plot['Y_pred'])

# Split by drug
ACM = plot[plot['Drug'] == 'Acetaminophen']
CXB = plot[plot['Drug'] == 'Celecoxib']
IPA = plot[plot['Drug'] == '3-Indolepropionic acid']
ASA = plot[plot['Drug'] == 'Aspirin']

# Define function to plot experimental data
def exp(df, ax, color, drug, ann):
    unique_temperatures = df['Temperature (K)'].unique()
    fontsize = 12
    width = 0.35
    x = np.arange(len(df['Solvent_mass_fraction'].unique()))
    
    for i, temp in enumerate(unique_temperatures):
        subset = df[df['Temperature (K)'] == temp]
        bar_color = color
        ax.bar(x + i * width, subset['LogS'], width=width, label=f'Temperature {temp} K', 
               color=bar_color, edgecolor='black', linewidth=0.5)
        
        if temp == 313.15:
            ax.bar(x + i * width, subset['LogS'], width=width, hatch='///', 
                   edgecolor='black', fill=False, linewidth=0)
    
    for i, temp in enumerate(unique_temperatures):
        subset = df[df['Temperature (K)'] == temp]
        ax.errorbar(x + i * width, subset['LogS'], yerr=subset['LogS_SD'], 
                   fmt='none', color='k', capsize=5)

    ax.set_xlabel('Ethanol Fraction')
    ax.set_ylabel('Experimental LogS')
    ax.set_title(drug)
    ax.set_xticks(x + width * (len(unique_temperatures) - 1) / 2)
    ax.set_xticklabels(df['Solvent_mass_fraction'].unique())
    ax.legend().set_visible(False)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.annotate(ann, xy=(0, 1.08), xycoords="axes fraction", va="top", ha="left", fontsize=fontsize)

# Define function to plot metrics
def metrics(df, ax, color, ann):
    fontsize = 12
    mae = mean_absolute_error(df['LogS'], df['Y_pred'])
    medae = median_absolute_error(df['LogS'], df['Y_pred'])
    mse = mean_squared_error(df['LogS'], df['Y_pred'])
    rmse = np.sqrt(mse)
    pearson_r = pearsonr(df['LogS'], df['Y_pred'])[0]
    spearman_r = spearmanr(df['LogS'], df['Y_pred'])[0]
    
    metrics = ['MAE', 'MedAE', 'RMSE', 'MSE', 'PCC', 'SCC']
    values = [mae, medae, rmse, mse, pearson_r, spearman_r]

    bars = ax.bar(metrics, values, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('Values')
    ax.set_ylim(0, 3.3)
    ax.set_title('')
    
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height, 
                    f'{height:.2f}', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0, 
                    f'{height:.2f}', ha='center', va='bottom')

    ax.annotate(ann, xy=(0, 1.08), xycoords="axes fraction", va="top", ha="left", fontsize=fontsize)

# Define function to create scatter plots
def scatter(df, ax, color, ann):
    fontsize = 12
    ax.scatter(df['LogS'], df['Y_pred'], color=color)
    max_limit = max(df['LogS'].max(), df['Y_pred'].max())
    min_limit = min(df['LogS'].min(), df['Y_pred'].min())
    ax.set_xlim(min_limit-0.2, max_limit+0.2)
    ax.set_ylim(min_limit-0.2, max_limit+0.2)
    ax.set_xlabel('Actual LogS')
    ax.set_ylabel('Predicted LogS')
    ax.set_title(df['Drug'].iloc[0])
    ax.annotate(ann, xy=(0, 1.08), xycoords="axes fraction", va="top", ha="left", fontsize=fontsize)

# Define colors for each drug
acm_color = 'dimgray'
ipa_color = red
cxb_color = blue
asa_color = yellow

# Create comparison figure
fig = plt.figure(figsize=(20, 8))
grid = plt.GridSpec(2, 4, wspace=0.3, hspace=0.3)

# First row: Experimental data for each drug
ax1 = fig.add_subplot(grid[0, 0])
exp(df=ASA, ax=ax1, color=asa_color, ann='a)', drug='Aspirin')

ax2 = fig.add_subplot(grid[0, 1])
exp(df=ACM, ax=ax2, color=acm_color, ann='b)', drug='Acetaminophen')

ax3 = fig.add_subplot(grid[0, 2])
exp(df=IPA, ax=ax3, color=ipa_color, ann='c)', drug='3-Indolepropionic acid')

ax4 = fig.add_subplot(grid[0, 3])
exp(df=CXB, ax=ax4, color=cxb_color, ann='d)', drug='Celecoxib')

# Second row: Metrics for each drug
ax5 = fig.add_subplot(grid[1, 0])
metrics(df=ASA, ax=ax5, color=asa_color, ann='e)')

ax6 = fig.add_subplot(grid[1, 1])
metrics(df=ACM, ax=ax6, color=acm_color, ann='f)')

ax7 = fig.add_subplot(grid[1, 2])
metrics(df=IPA, ax=ax7, color=ipa_color, ann='g)')

ax8 = fig.add_subplot(grid[1, 3])
metrics(df=CXB, ax=ax8, color=cxb_color, ann='h)')

fig.patch.set(facecolor='none')
plt.tight_layout()
plt.savefig('Figure_5_' + model_for + '_Validation' + data_for + '.png', dpi=600)
plt.show()

# Function to plot feature importance for XGBoost model
def plot_feature_importance_xgb(booster, ax, selected, ann, importance_type='weight', top_n=15, filter_drug_features=True):
    fontsize = 12
    importances = booster.get_score(importance_type=importance_type)
    feature_names = list(importances.keys())
    importance_values = list(importances.values())
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    if filter_drug_features:
        importance_df = importance_df[importance_df['Feature'].str.startswith('Drug_')]
    
    importance_df_top_n = importance_df.head(top_n)
    importance_df_top_n['Feature'] = importance_df_top_n['Feature'].replace('Drug_Collected_Melting_temp (K)', 'Drug_Melting_Temp')
    
    palette_choice = ['lightcoral' if feature in selected else blue for feature in importance_df_top_n['Feature']]
    sns.barplot(x='Importance', y='Feature', data=importance_df_top_n, ax=ax, palette=palette_choice, edgecolor='black')
    
    ax.annotate(ann, xy=(0, 1.025), xycoords="axes fraction", va="top", ha="left", fontsize=fontsize)
    ax.set_title("", fontsize=fontsize)
    ax.set_xlabel("Feature Importance", fontsize=fontsize)
    ax.set_ylabel("", fontsize=fontsize)
    
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.tick_params(axis='x', which='minor', labelsize=fontsize)
    ax.tick_params(axis='y', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', which='minor', labelsize=fontsize)
    
    importance_df_top_n['Feature'] = importance_df_top_n['Feature'].replace('Drug_Melting_Temp', 'Drug_Collected_Melting_temp (K)')
    
    return importance_df_top_n

# Add Type column for visualization
lab_X['Type'] = lab_X['Drug']

# Function to plot distribution of features
def plot_distribution_with_labels_on_trend(ann, prop, ax, show_legend=False, legend_pos=(-0.08, 1.28), ncol=2):
    fontsize = 12
    
    sns.boxplot(y=literature_X[prop], color='lightcoral', fliersize=1, ax=ax)
    
    unique_drugs = ['Aspirin', 'Acetaminophen', '3-Indolepropionic acid', 'Celecoxib']
    colors = [yellow, black, red, blue]
    
    drug_color_map = dict(zip(unique_drugs, colors))
    
    x_positions = np.linspace(-0.0, 0.0, len(unique_drugs))
    
    legend_handles = []
    
    for idx, drug in enumerate(unique_drugs):
        drug_data = lab_X[lab_X["Drug"] == drug]
        scatter = ax.scatter([x_positions[idx]] * len(drug_data), drug_data[prop], 
                           color=drug_color_map[drug], label=drug, 
                           alpha=1.0, edgecolor='black', linewidth=0.8, s=180,
                           zorder=10)
        legend_handles.append(scatter)
    
    ax.set_ylabel('', fontsize=fontsize)
    ax.set_title(prop, fontsize=fontsize)
    
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    
    if show_legend:
        ax.legend(handles=legend_handles, title='', loc='upper left', 
                 bbox_to_anchor=legend_pos, frameon=False, ncol=ncol, fontsize=fontsize+1)
        
    ax.annotate(ann, xy=(0, 1.05), xycoords="axes fraction", va="top", ha="left", fontsize=fontsize)

# Create feature importance figure
fig = plt.figure(figsize=(18, 12))
grid = gridspec.GridSpec(2, 4, fig, wspace=0.2, hspace=0.1)
selected=['Drug_MinPartialCharge', 'Drug_MaxPartialCharge', 'Drug_FpDensityMorgan3', 'Drug_MinEStateIndex']

ax1 = fig.add_subplot(grid[0:2, 0:2])
important_features = plot_feature_importance_xgb(booster=best_model.get_booster(), ax=ax1, 
                                                selected=selected, ann='a)')

ax2 = fig.add_subplot(grid[0, 2])
plot_distribution_with_labels_on_trend(ann='b)', prop='Drug_MinPartialCharge', ax=ax2, 
                                     legend_pos=(0.25, 1.25), ncol=2, show_legend=True)

ax3 = fig.add_subplot(grid[0, 3])
plot_distribution_with_labels_on_trend(ann='c)', prop='Drug_MinEStateIndex', ax=ax3, show_legend=False)

ax4 = fig.add_subplot(grid[1, 2])
plot_distribution_with_labels_on_trend(ann='d)', prop='Drug_FpDensityMorgan3', ax=ax4, show_legend=False)

ax5 = fig.add_subplot(grid[1, 3])
plot_distribution_with_labels_on_trend(ann='e)', prop='Drug_MaxPartialCharge', ax=ax5, show_legend=False)

fig.patch.set(facecolor='none')
plt.savefig('Figure_6_' + model_for + data_for + '_Interpretation.png', dpi=600)
plt.show()

# Create top 20 features figure (without drug filtering)
fig, ax1 = plt.subplots(figsize=(18, 12))
fontsize = 24

all_important_features = plot_feature_importance_xgb(booster=best_model.get_booster(), ax=ax1, 
                                                   selected='', ann='', top_n=20, 
                                                   filter_drug_features=False)

ax1.set_xlabel('Feature Importance', fontsize=fontsize)
ax1.set_ylabel('', fontsize=fontsize)
ax1.tick_params(axis='x', labelsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize)
plt.tight_layout()

fig.patch.set(facecolor='none')
plt.savefig('Figure_SI' + data_for + '_' + model_for + '_top 20 features.png', dpi=300)
plt.show()

# Print top features
for i, feature in enumerate(important_features['Feature']):
    print(f"{i+1}. {feature}")

# Create figure for all feature distributions
fig = plt.figure(figsize=(25, 15))
gs = gridspec.GridSpec(3, 5, fig)

for i, feature in enumerate(important_features['Feature']):
    row = i // 5
    col = i % 5
    ax = fig.add_subplot(gs[row, col])

    plot_distribution_with_labels_on_trend(
        ann=f'({chr(97+i)})',
        prop=feature,
        ax=ax,
        show_legend=(i==14), legend_pos=(-3, 3.65), ncol=4
    )

plt.tight_layout()
fig.patch.set(facecolor='none')
plt.savefig('Figure_SI' + data_for + '_' + model_for + '_All_feture_distribution.png', dpi=600)
plt.show()

# Function to analyze deviation of lab drug features from literature distribution
def print_summary_with_deviation_count():
    quantiles = literature_X.quantile([0.25, 0.75])
    unique_drugs = ['Acetaminophen', '3-Indolepropionic acid', 'Celecoxib', 'Aspirin']
    summary = []
    
    for feature in important_features['Feature']:
        Q1, Q3 = quantiles[feature].values
        for drug in unique_drugs:
            drug_data = lab_X[lab_X["Drug"] == drug]
            drug_feature_values = drug_data[feature]
            
            below_Q1 = drug_feature_values < Q1
            above_Q3 = drug_feature_values > Q3
            
            deviation = "deviates" if any(below_Q1) or any(above_Q3) else "does not deviate"
            
            summary.append([drug, feature, deviation])
    
    summary_df = pd.DataFrame(summary, columns=['Drug', 'Feature', 'Deviation'])
    summary_pivot = summary_df.pivot(index='Drug', columns='Feature', values='Deviation')
    
    deviation_counts = summary_df.groupby('Drug')['Deviation'].apply(lambda x: (x == "deviates").sum()).reset_index(name='Deviation Count')
    
    summary_with_counts = summary_pivot.merge(deviation_counts, left_on='Drug', right_on='Drug')
    summary_with_counts.set_index('Drug', inplace=True)
    
    columns_order = ['Deviation Count'] + [col for col in summary_with_counts if col != 'Deviation Count']
    summary_with_counts = summary_with_counts[columns_order]
    
    return summary_with_counts

summary_with_counts = print_summary_with_deviation_count()
print("\nDeviation Summary:")
print(summary_with_counts)

# Function to filter drug columns for chemical space analysis
def filter_drug_columns(df):
    drug_columns = [col for col in df.columns if col.startswith('Drug')]
    new_df = df[drug_columns]
    if 'Drug-solvent system' in new_df.columns:
        new_df = new_df.drop(['Drug-solvent system'], axis=1)
    new_df = new_df.drop_duplicates()
    return new_df

# Filter datasets for chemical space analysis
drug_lab = filter_drug_columns(lab_X)
drug_literature = filter_drug_columns(literature_X)

# Drop columns based on data type
if data_for == '_ExpMP':
    to_drop = ['Drug_SMILES', 'Drug_Predicted_Melting_temp (K)']
elif data_for == '_PredMP':
    to_drop = ['Drug_SMILES', 'Drug_Collected_Melting_temp (K)']
elif data_for == '_NoMP':
    to_drop = ['Drug_SMILES', 'Drug_Predicted_Melting_temp (K)', 'Drug_Collected_Melting_temp (K)']

drug_literature = drug_literature.drop(to_drop, axis=1)
drug_lab = drug_lab.drop(to_drop, axis=1)

# Prepare data for clustering
X_literature = drug_literature.iloc[:, 1:].values
X_lab = drug_lab.iloc[:, 1:].values
drug_names = drug_lab.iloc[:, 0].values

# Scale data
scaler = StandardScaler()
X_literature_scaled = scaler.fit_transform(X_literature)
X_lab_scaled = scaler.transform(X_lab)

# Perform clustering
kmeans = KMeans(random_state=0)
clusters = kmeans.fit_predict(X_literature_scaled)

# Calculate Mahalanobis distances
cluster_sizes = pd.Series(clusters).value_counts().sort_index()
unique_clusters = np.unique(clusters)
distance_matrix = pd.DataFrame(index=unique_clusters, columns=drug_names)

for cluster_id in unique_clusters:
    cluster_points = X_literature_scaled[clusters == cluster_id]

    if cluster_points.shape[0] <= 1:
        print(f"Cluster {cluster_id} has too few points for covariance calculation.")
        continue

    mean = np.mean(cluster_points, axis=0)
    cov_matrix = np.cov(cluster_points, rowvar=False)
    
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    for i, point in enumerate(X_lab_scaled):
        distance = mahalanobis(point, mean, inv_cov_matrix)
        distance_matrix.at[cluster_id, drug_names[i]] = distance

distance_matrix.fillna('NaN', inplace=True)
distance_matrix['Cluster Size'] = cluster_sizes
distance_matrix = distance_matrix.astype(float, errors='ignore')

# Save Mahalanobis distances
distance_matrix.to_excel('Table' + data_for + '_M_distances.xlsx')
print("\nMahalanobis Distances:")
print(distance_matrix)

# Calculate structural similarity between lab drugs
sim_data = pd.read_csv("Raw_dataset_dataset_20240705.csv")
sim_data = sim_data[['Drug', 'Drug_SMILES', 'Type']]
sim_lab = sim_data[sim_data['Type'] == 'Lab']
sim_lab = sim_lab.drop(['Type'], axis=1)
sim_lab = sim_lab.drop_duplicates()

lab_drugs = sim_lab['Drug'].values
lab_smiles = sim_lab['Drug_SMILES'].values

# Calculate Tanimoto similarity
similarity_df_lab = pd.DataFrame(index=lab_drugs, columns=lab_drugs)

for i, lab_smile_1 in enumerate(lab_smiles):
    mol_lab_1 = Chem.MolFromSmiles(lab_smile_1)
    fp_lab_1 = AllChem.GetMorganFingerprintAsBitVect(mol_lab_1, 2, nBits=2048)
    for j, lab_smile_2 in enumerate(lab_smiles):
        mol_lab_2 = Chem.MolFromSmiles(lab_smile_2)
        fp_lab_2 = AllChem.GetMorganFingerprintAsBitVect(mol_lab_2, 2, nBits=2048)
        similarity = DataStructs.TanimotoSimilarity(fp_lab_1, fp_lab_2)
        similarity_df_lab.iloc[i, j] = float(similarity)

similarity_df_lab = round(similarity_df_lab, 3)
similarity_df_lab.to_excel('Lab_drug_similarity.xlsx')

# Calculate mean similarity
mask = np.tril(np.ones(similarity_df_lab.shape), k=-1).astype(bool)
lower_triangle_values = similarity_df_lab.where(mask).stack()
mean_value = lower_triangle_values.mean()
std_value = lower_triangle_values.std()

print("\nDrug Similarity Statistics:")
print(f"Mean Tanimoto Similarity: {round(mean_value, 3)}")
print(f"Std Dev Tanimoto Similarity: {round(std_value, 3)}")