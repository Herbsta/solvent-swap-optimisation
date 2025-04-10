"""
Example script demonstrating how to integrate the optimized models
with an existing data split in the solvent swap optimization project.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Import the optimized models
from models_full import (
    XGBoostModel, 
    NeuralNetworkModel, 
    VAEModel, 
    ModelComparer,
    train_models_with_existing_splits
)

# Import modules from existing project
from data_module import DataProcessor
from feature_module import FeatureProcessor


def load_and_split_data():
    """
    Load and split data using existing DataProcessor.
    For demonstration - in practice, use your existing X_train, y_train, etc.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load data
    data_processor = DataProcessor.CreateDataProcessor(
        raw_data_path='curve_fit_results_x_is_7.csv',
        system_columns=['solvent_1', 'solvent_2', 'temperature'],
    )
    
    # Split data
    target_columns = ['J0', 'J1', 'J2']
    X_train, X_test, y_train, y_test = data_processor.split_data(
        target_columns=target_columns
    )
    
    return X_train, X_test, y_train, y_test


def main():
    """Main execution function."""
    output_path = "optimized_models_output"
    os.makedirs(output_path, exist_ok=True)
    
    print("=== Optimized Model Training for Solvent Swap Optimization ===")
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Option 1: Using existing data splits directly
    # In practice, you would use your own X_train, y_train, etc.
    # print("\n--- Option 1: Using Existing Data Splits ---")
    
    # print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    # print(f"Test set: {X_test.shape[0]} samples")
    # print(f"Target dimensions: {y_train.shape[1]} outputs")
    
    # # Train all models and get comparison
    # results = train_models_with_existing_splits(
    #     X_train, y_train, X_test, y_test, 
    #     optimize=True  # Set to True for Bayesian optimization
    # )
    
    # # Extract results
    # models = results['models']
    # comparer = results['comparer']
    # comparison_results = results['results']
    
    # # Save results
    # comparison_results.to_csv(os.path.join(output_path, "model_comparison.csv"), index=False)
    
    # # Plot comparison
    # comparer.plot_comparison(metric='rmse')
    # plt.savefig(os.path.join(output_path, "model_comparison_rmse.png"))
    # plt.close()
    
    # comparer.plot_comparison(metric='r2')
    # plt.savefig(os.path.join(output_path, "model_comparison_r2.png"))
    # plt.close()
    
    # Option 2: Training models individually with more control
    print("\n--- Option 2: Training Models Individually ---")
    
    # Create XGBoost model with custom settings
    xgb_model = XGBoostModel(
        name="xgb_custom",
        n_iter=50,      # Reduce iterations for faster optimization
        cv=5            # 5-fold cross-validation
    )
    
    # Train with optimization
    print("Training XGBoost model...")
    xgb_model.train(X_train, y_train, optimize=True)
    
    # Evaluate
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    print("XGBoost metrics:")
    for metric, value in xgb_metrics.items():
        if not metric.startswith('mae_') and not metric.startswith('mse_') and not metric.startswith('rmse_'):
            print(f"- {metric}: {value:.4f}")
    
    # Save model
    xgb_model.save(os.path.join(output_path, "xgb_custom"))
    
    # Get feature importance if available
    if hasattr(xgb_model, 'get_feature_importance'):
        importance = xgb_model.get_feature_importance()
        importance.to_csv(os.path.join(output_path, "xgb_feature_importance.csv"), index=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        plt.barh(importance.head(15)['feature'], importance.head(15)['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 15 Features (XGBoost)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "xgb_feature_importance.png"))
        plt.close()
    
    # Create and train Neural Network model with custom settings
    nn_model = NeuralNetworkModel(
        name="neural_network_custom",
        n_iter=30        # Fewer iterations for demonstration
    )
    
    print("Training Neural Network model...")
    nn_model.train(X_train, y_train, optimize=True)
    
    # Evaluate
    nn_metrics = nn_model.evaluate(X_test, y_test)
    print("Neural Network metrics:")
    for metric, value in nn_metrics.items():
        if not metric.startswith('mae_') and not metric.startswith('mse_') and not metric.startswith('rmse_'):
            print(f"- {metric}: {value:.4f}")
    
    # Save model
    nn_model.save(os.path.join(output_path, "nn_custom"))
    
    # Train VAE model with minimal optimization for demonstration
    vae_model = VAEModel(
        name="vae_custom",
        n_iter=20        # Even fewer iterations for VAE (more complex)
    )
    
    print("Training VAE model...")
    vae_model.train(X_train, y_train, optimize=True)
    
    # Evaluate
    vae_metrics = vae_model.evaluate(X_test, y_test)
    print("VAE metrics:")
    for metric, value in vae_metrics.items():
        if not metric.startswith('mae_') and not metric.startswith('mse_') and not metric.startswith('rmse_'):
            print(f"- {metric}: {value:.4f}")
    
    # Save model
    vae_model.save(os.path.join(output_path, "vae_custom"))
    
    # Compare individual models
    individual_models = {
        'xgboost': xgb_model,
        'neural_network': nn_model,
        'vae': vae_model
    }
    
    individual_comparer = ModelComparer(individual_models)
    individual_results = individual_comparer.compare(X_test, y_test)
    
    print("\nIndividual model comparison:")
    print(individual_results)
    individual_results.to_csv(os.path.join(output_path, "individual_comparison.csv"), index=False)
    
    # Visualize predictions for the best model
    best_model_name = individual_results.iloc[0]['model']
    best_model = individual_models[best_model_name]
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Plot predictions vs actual for each target
    target_names = ['J0', 'J1', 'J2']  # Adjust based on your targets
    
    for i, target in enumerate(target_names):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5)
        
        # Add diagonal line (perfect predictions)
        min_val = min(y_test.iloc[:, i].min(), np.min(y_pred[:, i]))
        max_val = max(y_test.iloc[:, i].max(), np.max(y_pred[:, i]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add metrics
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{target} - {best_model_name.upper()} Model')
        plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{best_model_name}_{target}_predictions.png"))
        plt.close()
    
    print(f"\nAll models and results saved to {output_path}")


if __name__ == "__main__":
    # Import metrics for individual plotting
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    main()