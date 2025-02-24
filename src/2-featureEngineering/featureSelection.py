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

def random_forest_importance(X_train, y_train, X_test, y_test):
    """Analyze feature importance using Random Forest"""
    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(30))
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('output/feature_engineering/rf_importance.png', dpi=300)
    
    # Save feature importances
    feature_importances.to_csv('output/feature_engineering/rf_feature_importance.csv', index=False)
    
    # Use SelectFromModel to get relevant features
    selector = SelectFromModel(rf, threshold='mean')
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    
    print(f"\nRandom Forest selected {len(selected_features)} features")
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
    
    return selected_features, feature_importances, selector

def recursive_feature_elimination(X_train, y_train, X_test, y_test):
    """Perform Recursive Feature Elimination with Cross-Validation"""
    # Use RF as the estimator
    estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Set up RFECV
    rfecv = RFECV(
        estimator=estimator,
        step=1,  # Remove one feature at a time
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        min_features_to_select=5,
        n_jobs=-1
    )
    
    # Fit RFECV
    rfecv.fit(X_train, y_train)
    
    # Get optimal number of features
    n_features_optimal = rfecv.n_features_
    print(f"\nRFECV selected {n_features_optimal} optimal features")
    
    # Plot CV scores
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Negative Mean Squared Error")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.axvline(x=n_features_optimal, color='r', linestyle='--', 
               label=f'Optimal number of features: {n_features_optimal}')
    plt.legend()
    plt.title('Recursive Feature Elimination with Cross-Validation')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/feature_engineering/rfecv_scores.png', dpi=300)
    
    # Get selected features
    selected_features = X_train.columns[rfecv.support_]
    
    print(f"RFECV selected features: {', '.join(selected_features[:10])}")
    
    # Evaluate performance with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_selected, y_train)
    
    y_pred = model.predict(X_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Performance with {len(selected_features)} RFECV-selected features:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return selected_features, rfecv

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
    
    # Random Forest feature importance
    print("\nAnalyzing Random Forest feature importance...")
    rf_features, rf_importance, rf_selector = random_forest_importance(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Recursive Feature Elimination
    print("\nPerforming Recursive Feature Elimination...")
    rfe_features, rfecv = recursive_feature_elimination(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Combine feature sets
    combined_features = list(set(rf_features) | set(rfe_features))
    print(f"\nCombined feature set contains {len(combined_features)} features")
    
    # Create final feature importance DataFrame
    final_feature_importance = rf_importance[rf_importance['Feature'].isin(combined_features)]
    final_feature_importance.to_csv('output/feature_engineering/final_selected_features.csv', index=False)
    
    # Evaluate final feature set
    print("\nEvaluating final feature set...")
    final_model, rmse, r2, cv_rmse = evaluate_final_features(
        X_train_scaled, y_train, X_test_scaled, y_test, combined_features
    )
    
    # Save models and transformers
    print("\nSaving models and transformers...")
    joblib.dump(scaler, 'output/feature_engineering/scaler.joblib')
    joblib.dump(encoder, 'output/feature_engineering/encoder.joblib')
    joblib.dump(rf_selector, 'output/feature_engineering/rf_selector.joblib')
    joblib.dump(rfecv, 'output/feature_engineering/rfecv.joblib')
    joblib.dump(final_model, 'output/feature_engineering/final_model.joblib')
    
    # Save feature list
    with open('output/feature_engineering/selected_features.txt', 'w') as f:
        f.write('\n'.join(combined_features))
    
    print("\nFeature engineering completed successfully!")
    print(f"Selected {len(combined_features)} features for modeling")
    print(f"Results saved to output/feature_engineering/")

if __name__ == "__main__":
    main()