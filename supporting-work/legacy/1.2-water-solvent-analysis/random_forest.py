import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import os
from data import water, domain_split
import pandas as pd

# Create output directory for plots and models
os.makedirs('output/water_feature_engineering', exist_ok=True)

def prepare_features(data):
    """Prepare feature matrices X and target y"""
    # Drop identifier columns
    drop_cols = ['api', 'solvent','solubility']
    
    # Create X and y
    X = data.drop(columns=drop_cols)
    y = data['solubility']
    
    return X, y

class NoiseBasedSelector(SelectFromModel):
    """Custom feature selector that uses random noise variables as a threshold."""
    
    def __init__(self, estimator, noise_threshold=None, noise_feature_prefix='random_'):
        self.noise_threshold = noise_threshold
        self.noise_feature_prefix = noise_feature_prefix
        super().__init__(estimator, threshold=None)
    
    def _get_support_mask(self):
        if hasattr(self.estimator, 'feature_importances_'):
            importances = self.estimator.feature_importances_
        else:
            importances = self.estimator.coef_[0] if len(self.estimator.coef_.shape) > 1 else self.estimator.coef_
        
        if hasattr(self.estimator, 'feature_names_in_'):
            feature_names = self.estimator.feature_names_in_
            noise_indices = [i for i, name in enumerate(feature_names) 
                           if self.noise_feature_prefix in str(name)]
        else:
            feature_count = len(importances)
            noise_indices = []
        
        if self.noise_threshold == 'max' and noise_indices:
            noise_importances = [importances[i] for i in noise_indices]
            threshold = max(noise_importances) if noise_importances else 0
        elif self.noise_threshold == 'mean' and noise_indices:
            noise_importances = [importances[i] for i in noise_indices]
            threshold = np.mean(noise_importances) if noise_importances else 0
        else:
            threshold = np.mean(importances)
        
        return importances > threshold

def random_forest_importance(X_train, y_train, X_test, y_test):
    """Analyze feature importance using Random Forest with random noise variables"""
    X_train_with_noise = X_train.copy()
    X_test_with_noise = X_test.copy()
    
    # Add random noise variables
    np.random.seed(42)
    noise_types = {
        'random_uniform': np.random.uniform,
        'random_normal': np.random.normal,
        'random_exponential': np.random.exponential
    }
    
    for name, func in noise_types.items():
        X_train_with_noise[name] = func(size=X_train.shape[0])
        X_test_with_noise[name] = func(size=X_test.shape[0])
    
    # Add random permutation of target
    X_train_with_noise['random_target'] = np.random.permutation(y_train.values)
    X_test_with_noise['random_target'] = np.random.permutation(y_test.values)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_with_noise, y_train)
    rf.feature_names_in_ = X_train_with_noise.columns
    
    # Get and plot feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_train_with_noise.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(14, 12))
    colors = ['red' if 'random_' in feat else 'steelblue' 
             for feat in feature_importances.head(30)['Feature']]
    
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(30), 
                palette=colors)
    plt.title('Random Forest Feature Importance (with Noise Variables)')
    plt.tight_layout()
    plt.savefig('output/water_feature_engineering/rf_importance_with_noise.png')
    
    # Select features using noise-based threshold
    selector = NoiseBasedSelector(rf, noise_threshold='max')
    selector.fit(X_train_with_noise, y_train)
    
    selected_features = [feat for feat in X_train_with_noise.columns[selector.get_support()]
                        if 'random_' not in feat]
    
    return selected_features, feature_importances, selector

def main():
    # Get data from water solvent analysis
    print("Loading water solvent analysis data...")
    df = water.get_data()
    
    # Create domain-specific splits
    print("\nCreating data splits...")
    splits = domain_split(df)
    
    # Prepare features
    X_train, y_train = prepare_features(splits['train'])
    X_test, y_test = prepare_features(splits['test_generic'])
    
    # Random Forest feature importance
    print("\nAnalyzing feature importance...")
    selected_features, importance_df, selector = random_forest_importance(
        X_train, y_train, X_test, y_test
    )
    
    # Train final model with selected features
    print(f"\nTraining final model with {len(selected_features)} features...")
    X_train_final = X_train[selected_features]
    X_test_final = X_test[selected_features]
    
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_train_final, y_train)
    
    # Evaluate model
    y_pred = final_model.predict(X_test_final)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Final model performance on generic test:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    
    X_test, y_test = prepare_features(splits['test_new_apis'])
    # Evaluate model
    y_pred = final_model.predict(X_test[selected_features])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Final model performance on new apis:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    X_test, y_test = prepare_features(splits['test_new_solvents'])
    # Evaluate model
    y_pred = final_model.predict(X_test[selected_features])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Final model performance on new solvents:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    
    # Save results
    print("\nSaving models and results...")
    
    importance_df.to_csv('output/water_feature_engineering/feature_importance.csv', 
                        index=False)
    
    with open('output/water_feature_engineering/selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))
    
    print("Feature selection completed successfully!")

if __name__ == "__main__":
    main()