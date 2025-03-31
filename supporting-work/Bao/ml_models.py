import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from scipy.stats import pearsonr, spearmanr

# ML Models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

# ML Evaluation
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Bayesian Optimization
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

# Define search spaces for the three models
search_spaces = {
    "RF": {
        "n_estimators": Integer(10, 400),
        "max_depth": Integer(3, 20),
        "min_samples_split": Real(0.01, 0.1),
        "min_samples_leaf": Integer(1, 20),
        "max_features": Categorical(['auto', 'sqrt', 'log2']),
        "bootstrap": Categorical([True, False]),
    },
    "XGB": {
        "n_estimators": Integer(10, 400),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "max_depth": Integer(3, 20),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
        "gamma": Real(0, 5),
    },
    "LightGBM": {
        "num_leaves": Integer(10, 400),
        "max_depth": Integer(3, 20),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "n_estimators": Integer(100, 1000),
        "bagging_fraction": Real(0.5, 1),
        "feature_fraction": Real(0.5, 1),
        "min_child_samples": Integer(5, 100),
    }
}

# Create model dictionary
models = {
    "RF": RandomForestRegressor(random_state=0, n_jobs=-1),
    "XGB": XGBRegressor(random_state=0, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(random_state=0, n_jobs=-1)
}

def data_processing(file_path):
    """Load and process data"""
    data = pd.read_csv(file_path)
    
    # Split data
    train = data[data['Type'] == 'Train']
    test = data[data['Type'] == 'Test']
    
    # Process train and test datasets
    X_train, Y_train, G_train = extract_features(train)
    X_test, Y_test, G_test = extract_features(test)
    
    return X_train, Y_train, G_train, X_test, Y_test, G_test

def extract_features(dataset):
    """Extract features, target variables, and groups"""
    X = dataset.drop(['Type', 'Drug', 'Solvent_1', 'Solvent_2', 'Drug-solvent system', 'LogS', 'Class', 'Solubility (g/100g)'], axis=1)
    Y = dataset['LogS']
    G = dataset['Drug-solvent system']
    
    return X, Y, G

def perform_hp_screening(model_name, X_train, Y_train, G_train, n_iter):
    """Perform hyperparameter screening using Bayesian Optimization"""
    model = models[model_name]
    search_space = search_spaces[model_name]
    
    # Define cross-validation strategy
    cv = GroupKFold(n_splits=10)
    
    # Create BayesSearchCV object
    bscv = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_iter=n_iter,
        n_jobs=-1,
        verbose=1,
        random_state=0
    )
    
    # Fit model
    bscv.fit(X_train, Y_train, groups=G_train)
    
    # Get optimization history
    optimization_history = bscv.cv_results_
    
    return bscv.best_estimator_, bscv.best_params_, bscv.best_score_, optimization_history

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_models(model_names, results, X_train, Y_train, G_train, X_test, Y_test):
    """Evaluate models using cross-validation and test set"""
    metrics = ['MAE', 'RMSE', 'PCC', 'SCC']
    train_summary = pd.DataFrame(index=metrics, columns=model_names)
    test_summary = pd.DataFrame(index=metrics, columns=model_names)
    predictions = pd.DataFrame(columns=model_names)
    
    for model_name in model_names:
        model = results[model_name]['best_estimator']
        
        # Cross-validation on training data
        cv = GroupKFold(n_splits=10)
        Y_pred_train = cross_val_predict(model, X_train, Y_train, cv=cv, groups=G_train, n_jobs=-1)
        
        Y_train_array = np.ravel(Y_train)
        Y_pred_train = np.ravel(Y_pred_train)
        
        # Calculate training metrics
        train_summary[model_name]['MAE'] = mean_absolute_error(Y_train_array, Y_pred_train)
        train_summary[model_name]['RMSE'] = rmse(Y_train_array, Y_pred_train)
        train_summary[model_name]['PCC'] = pearsonr(Y_train_array, Y_pred_train)[0]
        train_summary[model_name]['SCC'] = spearmanr(Y_train_array, Y_pred_train)[0]
        
        # Fit model on all training data
        model.fit(X_train, Y_train)
        
        # Predict on test data
        Y_pred_test = model.predict(X_test)
        Y_pred_test = np.ravel(Y_pred_test)
        
        # Calculate test metrics
        test_summary[model_name]['MAE'] = mean_absolute_error(Y_test, Y_pred_test)
        test_summary[model_name]['RMSE'] = rmse(Y_test, Y_pred_test)
        test_summary[model_name]['PCC'] = pearsonr(Y_test, Y_pred_test)[0]
        test_summary[model_name]['SCC'] = spearmanr(Y_test, Y_pred_test)[0]
        
        # Store predictions
        predictions[model_name] = Y_pred_test
    
    # Calculate absolute errors for visualization
    test_AE = predictions.copy()
    test_AE['Y'] = Y_test.values
    
    for model_name in model_names:
        test_AE[model_name] = abs(test_AE[model_name] - test_AE['Y'])
    
    test_AE = test_AE.drop(['Y'], axis=1)
    
    return train_summary, test_summary, test_AE, predictions

def visualize_results(test_AE, test_summary):
    """Visualize model performance"""
    # Sort models by MAE
    test_summary_T = test_summary.T
    sorted_models_ind = test_summary_T.sort_values(by="MAE", ascending=True).index
    
    # Prepare data for visualization
    test_AE_sorted = test_AE[sorted_models_ind]
    
    # Create boxplot of absolute errors
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=test_AE_sorted, showfliers=False, showmeans=True, linewidth=1.0,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", 
                           "markersize": 7, "markeredgewidth": 0.5})
    
    plt.ylabel('Model Absolute Error')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300)
    plt.show()
    
    return sorted_models_ind

def plot_optimization_history(model_names, results):
    """Plot optimization history for each model"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, model_name in enumerate(model_names):
        optimization_history = results[model_name]['optimization_history']
        
        scores = optimization_history['mean_test_score']
        iterations = range(1, len(scores) + 1)
        
        best_scores = np.maximum.accumulate(scores)
        
        ax = axs[idx]
        ax.plot(iterations, scores, marker='o', label='Iteration Scores')
        ax.plot(iterations, best_scores, marker='x', linestyle='--', label='Best Scores')
        ax.set_xlabel('HP Optimization Iteration')
        ax.set_ylabel('Negative Mean Absolute Error')
        ax.set_title(model_name)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_history.png', dpi=300)
    plt.show()

def main():
    # File path
    file_path = 'refined_dataset_NoMP.csv'  # Update this to your actual file path
    
    # Process data
    print("Loading and processing data...")
    X_train, Y_train, G_train, X_test, Y_test, G_test = data_processing(file_path)
    
    # Set number of iterations for Bayesian optimization
    n_iter = 50  # Reduced for demonstration
    
    # Define model names
    model_names = ["RF", "XGB", "LightGBM"]
    
    # Perform hyperparameter optimization
    results = {}
    
    for model_name in model_names:
        print(f"\nOptimizing {model_name} hyperparameters...")
        start = time.time()
        
        best_model, best_params, best_score, history = perform_hp_screening(
            model_name, X_train, Y_train, G_train, n_iter
        )
        
        end = time.time()
        
        print(f"{model_name}: Best Score = {round(best_score, 3)}")
        print(f"Optimization time: {round((end-start)/60, 1)} minutes")
        
        # Store results
        results[model_name] = {
            'best_estimator': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': history
        }
    
    # Save results
    with open('bayesian_opt_results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Plot optimization history
    print("\nPlotting optimization history...")
    plot_optimization_history(model_names, results)
    
    # Evaluate models
    print("\nEvaluating models...")
    train_summary, test_summary, test_AE, predictions = evaluate_models(
        model_names, results, X_train, Y_train, G_train, X_test, Y_test
    )
    
    # Round results for display
    train_summary_round = train_summary.round(2)
    test_summary_round = test_summary.round(2)
    
    # Display results
    print("\nTraining Results:")
    print(train_summary_round)
    
    print("\nTest Results:")
    print(test_summary_round)
    
    # Visualize results
    print("\nVisualizing model performance...")
    sorted_models = visualize_results(test_AE, test_summary_round)
    
    # Save sorted results to Excel
    test_summary_round = test_summary_round[sorted_models]
    test_summary_round.to_excel('test_results.xlsx', index=True)
    
    train_summary_round = train_summary_round[sorted_models]
    train_summary_round.to_excel('train_results.xlsx', index=True)
    
    print("\nDone! Results saved to files.")

if __name__ == "__main__":
    main()