import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
import os
import joblib

# Create output directory for plots and models
os.makedirs('output/feature_engineering', exist_ok=True)

def load_data():
    """Load and merge solubility data with compound descriptors"""
    # Connect to SQLite database
    conn = sqlite3.connect('db/MasterDatabase.db')
    
    # Query to get all values from the solubility table
    query = """SELECT 
    compound_id, 
    solvent_1,
    solvent_2,
    solvent_1_weight_fraction,
    solvent_1_mol_fraction,
    temperature,
    solubility_g_g,
    solubility_mol_mol
    FROM solubility"""
    
    df_sol = pd.read_sql_query(query, conn)
    print(f"Loaded {len(df_sol)} solubility records")
    
    # Load the compound descriptors
    excel_file = 'src/2-featureEngineering/descriptorsAPIs.xlsx'
    df_desc = pd.read_excel(excel_file)
    print(f"Loaded {len(df_desc)} compound descriptors")
    
    # Merge the two dataframes
    merged_df = pd.merge(df_sol, df_desc, left_on='compound_id', right_on='PUBCHEM_COMPOUND_CID')
    print(f"Final merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    
    conn.close()
    return merged_df

def domain_split(data, random_state=42):
    """Create domain-specific splits for better evaluation"""
    # Regular random split
    generic_train, generic_test = train_test_split(
        data, test_size=0.10, random_state=random_state
    )
    
    # Split based on compounds (APIs)
    unique_apis = data['compound_id'].unique()
    api_train, api_test = train_test_split(
        unique_apis, test_size=0.10, random_state=random_state
    )
    
    # Create masks
    api_test_mask = data['compound_id'].isin(api_test)
    test_new_apis = data[api_test_mask]
    
    # Split based on solvent combinations
    unique_solv_combos = data.groupby(['solvent_1', 'solvent_2']).size().reset_index()
    train_combos, test_combos = train_test_split(
        unique_solv_combos, test_size=0.10, random_state=random_state
    )
    
    # Create solvent combination mask
    test_combo_mask = data.apply(
        lambda x: any((
            (test_combos['solvent_1'] == x['solvent_1']) & 
            (test_combos['solvent_2'] == x['solvent_2'])
        ).values),
        axis=1
    )
    
    test_new_combinations = data[test_combo_mask & ~api_test_mask]
    train_final = generic_train[~test_combo_mask & ~api_test_mask]
    
    print(f"Train set: {len(train_final)} samples")
    print(f"Test (generic): {len(generic_test)} samples")
    print(f"Test (new APIs): {len(test_new_apis)} samples")
    print(f"Test (new combinations): {len(test_new_combinations)} samples")
    
    return {
        'train': train_final,
        'test_generic': generic_test,
        'test_new_apis': test_new_apis,
        'test_new_combinations': test_new_combinations
    }

def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Create solvent combination feature
    df['solvent_combo'] = df['solvent_1'].astype(str) + '_' + df['solvent_2'].astype(str).fillna('')
    
    # Log-transform target for better distribution
    df['log_solubility'] = np.log1p(df['solubility_g_g'])
    
    # Calculate solvent ratio features
    df['solvent_ratio'] = df['solvent_1_weight_fraction'] / (1 - df['solvent_1_weight_fraction'])
    df['solvent_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['solvent_ratio'].fillna(1.0, inplace=True)  # Replace inf and NaN with 1.0 (pure solvent)
    
    # Add temperature interaction features
    df['temp_weight_interaction'] = df['temperature'] * df['solvent_1_weight_fraction']
    
    return df

def prepare_features(data, target='log_solubility'):
    """Prepare feature matrices X and target y"""
    # Drop unnecessary columns
    drop_cols = [
        'compound_id', 'solvent_1', 'solvent_2', 'PUBCHEM_COMPOUND_CID',
        'solubility_g_g', 'solubility_mol_mol', 'log_solubility',
        'mol'  # Remove mol column if present
    ]
    
    # Add other PUBCHEM columns to drop
    pubchem_cols = [col for col in data.columns if col.startswith('PUBCHEM_')]
    drop_cols.extend(pubchem_cols)
    
    # Create X and y
    X = data.drop(columns=drop_cols, errors='ignore')
    y = data[target]
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    return X, y, cat_cols

def encode_categorical(X_train, X_test, categorical_cols, y_train):
    """Encode categorical features using target encoding"""
    encoder = TargetEncoder()
    
    # Only encode if categorical columns exist
    if categorical_cols:
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols], y_train)
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        
        # Replace original columns with encoded versions
        X_train = X_train.drop(columns=categorical_cols)
        X_test = X_test.drop(columns=categorical_cols)
        
        X_train = pd.concat([X_train, X_train_encoded], axis=1)
        X_test = pd.concat([X_test, X_test_encoded], axis=1)
    
    return X_train, X_test, encoder

def remove_correlated_features(X_train, X_test, threshold=0.9):
    """Remove highly correlated features"""
    # Calculate correlation matrix
    corr_matrix = X_train.corr().abs()
    
    # Plot correlation matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='viridis', vmax=1, annot=False)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('output/feature_engineering/correlation_matrix.png', dpi=300)
    
    # Find highly correlated features
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} highly correlated features")
    
    # Drop correlated features
    X_train_filtered = X_train.drop(columns=to_drop)
    X_test_filtered = X_test.drop(columns=to_drop)
    
    return X_train_filtered, X_test_filtered, to_drop

def pca_analysis(X_train_scaled, feature_names):
    """Perform PCA analysis and return important components"""
    # Full PCA to analyze variance explained
    pca = PCA()
    pca.fit(X_train_scaled)
    
    # Plot variance explained
    plt.figure(figsize=(12, 6))
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7,
            label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
             label='Cumulative explained variance', color='red')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA: Explained Variance by Components')
    plt.axhline(y=0.9, color='k', linestyle='--', label='90% Variance Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/feature_engineering/pca_variance.png', dpi=300)
    
    # Find number of components for 90% variance
    n_components = np.argmax(cumulative_variance >= 0.9) + 1
    print(f"Number of components needed for 90% variance: {n_components}")
    
    # Create PCA for selected number of components
    pca_selected = PCA(n_components=n_components)
    pca_selected.fit(X_train_scaled)
    
    # Analyze top features in principal components
    feature_importance = []
    
    for i, component in enumerate(pca_selected.components_):
        # Get absolute component values
        abs_components = np.abs(component)
        # Create feature-component mapping
        component_features = pd.DataFrame({
            'Feature': feature_names,
            'Importance': abs_components,
            'Component': i+1,
            'Variance': pca_selected.explained_variance_ratio_[i]
        }).sort_values('Importance', ascending=False)
        
        feature_importance.append(component_features)
        
        # Print top features for important components
        if i < 5:  # Show top 5 components
            print(f"\nTop features for PC{i+1} (Variance: {pca_selected.explained_variance_ratio_[i]:.4f}):")
            print(component_features.head(10)[['Feature', 'Importance']])
    
    # Combine all component data
    all_component_data = pd.concat(feature_importance)
    all_component_data.to_csv('output/feature_engineering/pca_feature_importance.csv', index=False)
    
    return pca_selected, n_components, all_component_data

# Define custom selector class outside any function
class NoiseBasedSelector(SelectFromModel):
    """Custom feature selector that uses random noise variables as a threshold."""
    
    def __init__(self, estimator, noise_threshold=None, noise_feature_prefix='random_'):
        self.noise_threshold = noise_threshold
        self.noise_feature_prefix = noise_feature_prefix
        super().__init__(estimator, threshold=None)
        
    def _get_support_mask(self):
        # Override the method to use our custom noise-based threshold
        if hasattr(self.estimator, 'feature_importances_'):
            importances = self.estimator.feature_importances_
        else:
            importances = self.estimator.coef_[0] if len(self.estimator.coef_.shape) > 1 else self.estimator.coef_
            
        # Get feature names from estimator if available
        if hasattr(self.estimator, 'feature_names_in_'):
            feature_names = self.estimator.feature_names_in_
            noise_indices = [i for i, name in enumerate(feature_names) 
                           if self.noise_feature_prefix in str(name)]
        else:
            # Use indices directly if feature names not available
            # This is a fallback and might not correctly identify noise features
            feature_count = len(importances)
            noise_indices = []
        
        if self.noise_threshold == 'max' and noise_indices:
            # Use max noise importance as threshold
            noise_importances = [importances[i] for i in noise_indices]
            threshold = max(noise_importances) if noise_importances else 0
        elif self.noise_threshold == 'mean' and noise_indices:
            # Use mean noise importance as threshold
            noise_importances = [importances[i] for i in noise_indices]
            threshold = np.mean(noise_importances) if noise_importances else 0
        else:
            # Default to mean feature importance
            threshold = np.mean(importances)
        
        return importances > threshold

def random_forest_importance(X_train, y_train, X_test, y_test):
    """Analyze feature importance using Random Forest with random noise variables"""
    # Create a copy of the training data to add noise
    X_train_with_noise = X_train.copy()
    X_test_with_noise = X_test.copy()
    
    # Add random noise variables with different distributions
    np.random.seed(42)  # For reproducibility
    
    # Uniform random noise [0, 1]
    X_train_with_noise['random_uniform'] = np.random.uniform(0, 1, size=X_train.shape[0])
    X_test_with_noise['random_uniform'] = np.random.uniform(0, 1, size=X_test.shape[0])
    
    # Gaussian noise (normal distribution)
    X_train_with_noise['random_normal'] = np.random.normal(0, 1, size=X_train.shape[0])
    X_test_with_noise['random_normal'] = np.random.normal(0, 1, size=X_test.shape[0])
    
    # Exponential noise
    X_train_with_noise['random_exponential'] = np.random.exponential(1, size=X_train.shape[0])
    X_test_with_noise['random_exponential'] = np.random.exponential(1, size=X_test.shape[0])
    
    # Random permutation of target (this creates a feature that has the same distribution as the target but is randomly shuffled)
    X_train_with_noise['random_target'] = np.random.permutation(y_train.values)
    X_test_with_noise['random_target'] = np.random.permutation(y_test.values)
    
    print(f"Added 4 noise variables to feature set (total: {X_train_with_noise.shape[1]} features)")
    
    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_with_noise, y_train)
    
    # Store feature names in the model for the NoiseBasedSelector
    rf.feature_names_in_ = X_train_with_noise.columns
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_train_with_noise.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importances with noise variables highlighted
    plt.figure(figsize=(14, 12))
    
    # Create a color map where noise variables are highlighted in red
    colors = ['red' if 'random_' in feat else 'steelblue' for feat in feature_importances.head(30)['Feature']]
    
    ax = sns.barplot(x='Importance', y='Feature', data=feature_importances.head(30), palette=colors)
    plt.title('Random Forest Feature Importance (with Noise Variables)', fontsize=14)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Actual Features'),
        Patch(facecolor='red', label='Random Noise')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('output/feature_engineering/rf_importance_with_noise.png', dpi=300)
    
    # Find the highest ranked noise variable
    noise_features = [f for f in feature_importances['Feature'] if 'random_' in f]
    highest_noise_idx = feature_importances['Feature'].tolist().index(noise_features[0])
    highest_noise_importance = feature_importances.iloc[highest_noise_idx]['Importance']
    
    print(f"\nHighest ranked noise variable: {noise_features[0]} (Rank: {highest_noise_idx+1}, Importance: {highest_noise_importance:.6f})")
    
    # Filter features more important than the highest noise variable
    better_than_noise = feature_importances.iloc[:highest_noise_idx]
    print(f"Found {len(better_than_noise)} features more important than random noise")
    
    # Save feature importances
    feature_importances.to_csv('output/feature_engineering/rf_feature_importance_with_noise.csv', index=False)
    
    # Use the custom selector based on max noise importance
    selector = NoiseBasedSelector(rf, noise_threshold='max')
    selector.fit(X_train_with_noise, y_train)
    
    # Get selected features (excluding noise variables)
    all_selected = X_train_with_noise.columns[selector.get_support()]
    selected_features = [feat for feat in all_selected if 'random_' not in feat]
    
    print(f"\nRandom Forest selected {len(selected_features)} features (threshold: maximum noise importance)")
    print(f"Top 10 features: {', '.join(selected_features[:10])}")
    
    # Evaluate model with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    rf_selected = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selected.fit(X_train_selected, y_train)
    
    # Evaluate on test set
    y_pred = rf_selected.predict(X_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Performance with {len(selected_features)} selected features:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Also try with mean noise as threshold (usually more conservative)
    selector_mean = NoiseBasedSelector(rf, noise_threshold='mean')
    selector_mean.fit(X_train_with_noise, y_train)
    
    all_selected_mean = X_train_with_noise.columns[selector_mean.get_support()]
    selected_features_mean = [feat for feat in all_selected_mean if 'random_' not in feat]
    
    print(f"\nAlternative selection: {len(selected_features_mean)} features (threshold: mean noise importance)")
    
    return selected_features, feature_importances, selector

def evaluate_final_features(X_train, y_train, X_test, y_test, selected_features):
    """Evaluate the final set of selected features"""
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_selected, y_train)
    
    y_pred = model.predict(X_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nFinal model with {len(selected_features)} features:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X_train_selected, y_train, 
                              cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    print(f"Cross-validation RMSE: {cv_rmse:.4f}")
    
    return model, rmse, r2, cv_rmse

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df_processed = preprocess_data(df)
    
    # Create domain-specific data splits
    print("\nCreating domain-specific data splits...")
    splits = domain_split(df_processed)
    
    # Extract features and target
    X_train, y_train, cat_cols = prepare_features(splits['train'])
    X_test, y_test, _ = prepare_features(splits['test_generic'])
    
    # Encode categorical features
    print("\nEncoding categorical features...")
    X_train, X_test, encoder = encode_categorical(X_train, X_test, cat_cols, y_train)
    
    # Remove missing values
    X_train = X_train.dropna(axis=1)
    X_test = X_test[X_train.columns]
    
    # Remove highly correlated features
    print("\nRemoving highly correlated features...")
    X_train, X_test, correlated_features = remove_correlated_features(X_train, X_test)
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns
    )
    
    # Perform PCA analysis
    print("\nPerforming PCA analysis...")
    pca, n_components, pca_components = pca_analysis(X_train_scaled, X_train.columns)
    
    # Random Forest feature importance with noise
    print("\nAnalyzing Random Forest feature importance with noise variables...")
    rf_features, rf_importance, rf_selector = random_forest_importance(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    selected_features = rf_features
    print(f"Using {len(selected_features)} features selected by Random Forest")
    
    # Evaluate final feature set
    print("\nEvaluating final feature set...")
    final_model, rmse, r2, cv_rmse = evaluate_final_features(
        X_train_scaled, y_train, X_test_scaled, y_test, selected_features
    )
    
    # Save models and transformers
    print("\nSaving models and transformers...")
    joblib.dump(scaler, 'output/feature_engineering/scaler.joblib')
    joblib.dump(encoder, 'output/feature_engineering/encoder.joblib')
    joblib.dump(rf_selector, 'output/feature_engineering/rf_selector.joblib')
    joblib.dump(final_model, 'output/feature_engineering/final_model.joblib')
    
    # Save feature list
    with open('output/feature_engineering/selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))
    
    print("\nFeature engineering completed successfully!")
    print(f"Selected {len(selected_features)} features for modeling")
    print(f"Results saved to output/feature_engineering/")

if __name__ == "__main__":
    main()