import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from skopt.space import Real, Integer, Categorical
import json
import pickle

from base_model import BaseModelWithFeatureSelection

class XGBoostWithFeatureSelection(BaseModelWithFeatureSelection):
    """
    XGBoost model with feature selection capabilities.
    Can be used as a drop-in replacement for NeuralNetworkWithFeatureSelection.
    """
    def __init__(self, feature_selection_method='random_forest', n_features=50, keep_prefixes=None, 
                 objective='reg:squarederror', n_estimators=100, max_depth=6):
        """
        Initialize the XGBoost model with feature selection capabilities.
        
        Parameters:
        -----------
        feature_selection_method : str
            Method for feature selection: 'correlation', 'f_regression', 'rfe', 'random_forest'
        n_features : int
            Number of features to select
        keep_prefixes : list
            List of column prefixes to always keep regardless of feature selection
        objective : str
            XGBoost objective function
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        """
        super().__init__(feature_selection_method, n_features, keep_prefixes)
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.history = None
        self.feature_importances_ = None
        
    def _build_model(self, input_dim, output_dim=3, **params):
        """
        Build XGBoost model with specified parameters
        
        Parameters:
        -----------
        input_dim : int
            Input dimension (number of features)
        output_dim : int
            Output dimension (number of targets)
        params : dict
            Model hyperparameters
        
        Returns:
        --------
        model : list
            List of XGBoost models (one for each target)
        """
        # Extract parameters with defaults
        objective = params.get('objective', self.objective)
        n_estimators = params.get('n_estimators', self.n_estimators)
        max_depth = params.get('max_depth', self.max_depth)
        learning_rate = params.get('learning_rate', 0.1)
        subsample = params.get('subsample', 0.8)
        colsample_bytree = params.get('colsample_bytree', 0.8)
        min_child_weight = params.get('min_child_weight', 1)
        gamma = params.get('gamma', 0)
        
        # Common parameters for all models
        model_params = {
            'objective': objective,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'verbosity': 0,
            'n_jobs': -1,
            'random_state': 42,
            'eval_metric': 'rmse'  # Add evaluation metric here at model creation
        }
        
        # For multioutput regression, we create one model per target
        models = []
        for i in range(output_dim):
            models.append(xgb.XGBRegressor(**model_params))
            
        return models
    
    def _define_parameter_space(self):
        """Define hyperparameter search space for XGBoost"""
        return [
            Integer(3, 15, name='max_depth'),
            Integer(50, 500, name='n_estimators'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree'),
            Integer(1, 10, name='min_child_weight'),
            Real(0.0, 5.0, name='gamma')
        ]
    
    def _evaluate_model_for_optimization(self, models, X_train, y_train, X_val, y_val, params):
        """Evaluate XGBoost for hyperparameter optimization"""
        n_outputs = y_train.shape[1]
        val_scores = []
        
        # For early stopping
        early_stopping_rounds = 20
        
        # Train and evaluate each model (one per target)
        for i in range(n_outputs):
            model = xgb.XGBRegressor(
                objective=params.get('objective', self.objective),
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                gamma=params['gamma'],
                verbosity=0,
                n_jobs=-1,
                random_state=42,
                eval_metric='rmse'  # Add evaluation metric here instead of in fit()
            )
            
            # Train with early stopping - removed eval_metric parameter
            try:
                model.fit(
                    X_train, y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            except TypeError:
                # Fall back to basic fit if advanced parameters not supported
                model.fit(X_train, y_train[:, i])
            
            # Get validation score
            pred = model.predict(X_val)
            val_score = mean_squared_error(y_val[:, i], pred)
            val_scores.append(val_score)
        
        # Return the average validation score across all outputs
        return np.mean(val_scores)
    
    def _extract_best_params(self, optimization_result):
        """Extract best parameters from optimization result for XGBoost"""
        return {
            'max_depth': optimization_result.x[0],
            'n_estimators': optimization_result.x[1],
            'learning_rate': optimization_result.x[2],
            'subsample': optimization_result.x[3],
            'colsample_bytree': optimization_result.x[4],
            'min_child_weight': optimization_result.x[5],
            'gamma': optimization_result.x[6]
        }
    
    def train(self, X, y, validation_data=None, epochs=None, batch_size=None, 
              verbose=1, optimize_hyperparams=False, n_calls=50, **kwargs):
        """
        Train the XGBoost model with selected features
        
        Parameters:
        -----------
        X : DataFrame
            Feature DataFrame
        y : DataFrame
            Target DataFrame
        validation_data : tuple
            Tuple of (X_val, y_val) for validation
        epochs : int
            Not used for XGBoost, kept for API compatibility
        batch_size : int
            Not used for XGBoost, kept for API compatibility
        verbose : int
            Verbosity level
        optimize_hyperparams : bool
            Whether to optimize hyperparameters
        n_calls : int
            Number of iterations for optimization
        kwargs : dict
            Additional parameters for XGBoost
            
        Returns:
        --------
        self : object
            Fitted model
        """
        # Select features if not already done
        if self.selected_features is None:
            X_selected = self.select_features(X, y)
        else:
            X_selected = X[self.selected_features]
        
        # Convert to numpy arrays for XGBoost
        X_np = X_selected.values
        y_np = y.values
        
        # Prepare validation data if provided
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_selected = X_val[self.selected_features].values
            y_val_np = y_val.values
            val_data = (X_val_selected, y_val_np)
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            if verbose:
                print("Optimizing hyperparameters...")
            self.optimize_hyperparameters(X_np, y_np, val_data, n_calls=n_calls)
            
            if self.best_params:
                if verbose:
                    print(f"Best parameters: {self.best_params}")
                self.n_estimators = self.best_params.get('n_estimators', self.n_estimators)
                self.max_depth = self.best_params.get('max_depth', self.max_depth)
        
        # Build models (one per target)
        n_outputs = y_np.shape[1]
        if self.model is None:
            self.model = self._build_model(
                input_dim=X_np.shape[1],
                output_dim=n_outputs,
                **{**kwargs, **(self.best_params if self.best_params else {})}
            )
        
        # Storage for training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        # Train each model separately (one for each target)
        for i in range(n_outputs):
            if verbose:
                print(f"\nTraining model for target {i+1}/{n_outputs}")
            
            # Early stopping if validation data is provided
            eval_set = []
            if val_data is not None:
                eval_set.append((X_np, y_np[:, i]))  # Training data
                eval_set.append((val_data[0], val_data[1][:, i]))  # Validation data
            else:
                eval_set.append((X_np, y_np[:, i]))  # Only training data
            
            # Train the model - try different approaches based on XGBoost version compatibility
            fit_success = False
            fit_with_eval_set = False
            try:
                # First try with eval_set
                self.model[i].fit(
                    X_np, y_np[:, i],
                    eval_set=eval_set,
                    early_stopping_rounds=50 if val_data is not None else None,
                    verbose=verbose > 0
                )
                fit_success = True
                fit_with_eval_set = True
                if verbose > 1:
                    print("Training successful with eval_set")
            except TypeError as e:
                if verbose:
                    print(f"Warning: Error with eval_set: {e}")
                # Try with eval_metric parameter explicitly set to None
                try:
                    self.model[i].fit(
                        X_np, y_np[:, i],
                        eval_set=eval_set,
                        eval_metric=None,  # Explicitly set to None
                        early_stopping_rounds=50 if val_data is not None else None,
                        verbose=verbose > 0
                    )
                    fit_success = True
                    fit_with_eval_set = True
                    if verbose > 1:
                        print("Training successful with eval_set and eval_metric=None")
                except Exception as e:
                    if verbose:
                        print(f"Warning: Error with eval_set and eval_metric=None: {e}")
                    # Fall back to basic fit without eval_set
                    try:
                        self.model[i].fit(X_np, y_np[:, i])
                        fit_success = True
                        if verbose > 1:
                            print("Training successful with basic fit")
                    except Exception as e:
                        if verbose:
                            print(f"ERROR: Could not train model: {e}")
                        fit_success = False
            except Exception as e:
                if verbose:
                    print(f"Warning: Unexpected error during training: {e}")
                # Fall back to basic fit
                try:
                    self.model[i].fit(X_np, y_np[:, i])
                    fit_success = True
                    if verbose > 1:
                        print("Training successful with basic fit")
                except Exception as e:
                    if verbose:
                        print(f"ERROR: Could not train model: {e}")
                    fit_success = False
            
            # Skip evaluation results if training failed
            if not fit_success:
                if verbose:
                    print("WARNING: Training failed for this target, using dummy metrics")
                # Use dummy metrics
                self.history['train_loss'].append([0.0])
                if val_data is not None:
                    self.history['val_loss'].append([0.0])
                self.history['train_mae'].append([0.0])
                if val_data is not None:
                    self.history['val_mae'].append([0.0])
                continue
            
            # Only try to get evaluation results if we used eval_set
            if fit_with_eval_set:
                try:
                    # Try to get evaluation results
                    evals_result = self.model[i].evals_result()
                    
                    # Add training metrics to history if available
                    if evals_result and len(evals_result) > 0:
                        # Process evaluation results
                        if len(eval_set) > 1:  # If we have validation data
                            train_key = list(evals_result.keys())[0]
                            val_key = list(evals_result.keys())[1]
                            
                            # Check which metrics are available
                            available_metrics = list(evals_result[train_key].keys())
                            
                            # Use rmse if available, otherwise use first available metric
                            metric_key = 'rmse' if 'rmse' in available_metrics else available_metrics[0]
                            
                            # Convert from RMSE to MSE for consistency with neural network models
                            if metric_key == 'rmse':
                                self.history['train_loss'].append([x**2 for x in evals_result[train_key][metric_key]])
                                self.history['val_loss'].append([x**2 for x in evals_result[val_key][metric_key]])
                            else:
                                self.history['train_loss'].append(evals_result[train_key][metric_key])
                                self.history['val_loss'].append(evals_result[val_key][metric_key])
                            
                            # Use mae if available
                            if 'mae' in available_metrics:
                                self.history['train_mae'].append(evals_result[train_key]['mae'])
                                self.history['val_mae'].append(evals_result[val_key]['mae'])
                            else:
                                # If no MAE available, use the metric we have
                                self.history['train_mae'].append(evals_result[train_key][metric_key])
                                self.history['val_mae'].append(evals_result[val_key][metric_key])
                                
                        else:  # Only training data
                            train_key = list(evals_result.keys())[0]
                            available_metrics = list(evals_result[train_key].keys())
                            metric_key = 'rmse' if 'rmse' in available_metrics else available_metrics[0]
                            
                            if metric_key == 'rmse':
                                self.history['train_loss'].append([x**2 for x in evals_result[train_key][metric_key]])
                            else:
                                self.history['train_loss'].append(evals_result[train_key][metric_key])
                            
                            if 'mae' in available_metrics:
                                self.history['train_mae'].append(evals_result[train_key]['mae'])
                            else:
                                self.history['train_mae'].append(evals_result[train_key][metric_key])
                    else:
                        # Empty or invalid evaluation results
                        raise ValueError("Empty evaluation results")
                except Exception as e:
                    # Calculate metrics manually if evals_result fails
                    if verbose:
                        print(f"Warning: Could not retrieve evaluation metrics. Using calculated metrics. Error: {e}")
                    
                    # Predict training data
                    y_pred_train = self.model[i].predict(X_np)
                    train_mse = mean_squared_error(y_np[:, i], y_pred_train)
                    train_mae = mean_absolute_error(y_np[:, i], y_pred_train)
                    
                    # Store these metrics
                    self.history['train_loss'].append([train_mse])
                    self.history['train_mae'].append([train_mae])
                    
                    # If we have validation data, calculate validation metrics too
                    if val_data is not None:
                        y_pred_val = self.model[i].predict(val_data[0])
                        val_mse = mean_squared_error(val_data[1][:, i], y_pred_val)
                        val_mae = mean_absolute_error(val_data[1][:, i], y_pred_val)
                        self.history['val_loss'].append([val_mse])
                        self.history['val_mae'].append([val_mae])
            else:
                # We didn't use eval_set, calculate metrics manually
                if verbose:
                    print("Calculating metrics manually as eval_set was not used")
                
                # Predict training data
                y_pred_train = self.model[i].predict(X_np)
                train_mse = mean_squared_error(y_np[:, i], y_pred_train)
                train_mae = mean_absolute_error(y_np[:, i], y_pred_train)
                
                # Store these metrics
                self.history['train_loss'].append([train_mse])
                self.history['train_mae'].append([train_mae])
                
                # If we have validation data, calculate validation metrics too
                if val_data is not None:
                    y_pred_val = self.model[i].predict(val_data[0])
                    val_mse = mean_squared_error(val_data[1][:, i], y_pred_val)
                    val_mae = mean_absolute_error(val_data[1][:, i], y_pred_val)
                    self.history['val_loss'].append([val_mse])
                    self.history['val_mae'].append([val_mae])
        
        # Get feature importances
        self.feature_importances_ = self._get_feature_importances()
        
        return self
    
    def _get_feature_importances(self):
        """
        Get feature importances from all models
        
        Returns:
        --------
        importances : ndarray
            Array of feature importances
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Get feature importances from each model and average them
        importances = np.zeros(len(self.selected_features))
        for model in self.model:
            importances += model.feature_importances_
        importances /= len(self.model)
        
        return importances
    
    def predict(self, X):
        """
        Make predictions using the trained XGBoost models
        
        Parameters:
        -----------
        X : DataFrame
            Feature DataFrame
        
        Returns:
        --------
        predictions : ndarray
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        if self.selected_features is None:
            raise ValueError("Features have not been selected. Call select_features() first.")
            
        # Use only selected features for prediction
        X_selected = X[self.selected_features].values
        
        # Make predictions for each target
        predictions = np.zeros((X_selected.shape[0], len(self.model)))
        for i, model in enumerate(self.model):
            predictions[:, i] = model.predict(X_selected)
        
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate the XGBoost model on test data
        
        Parameters:
        -----------
        X : DataFrame
            Feature DataFrame
        y : DataFrame
            Target DataFrame
            
        Returns:
        --------
        evaluation : dict
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        if self.selected_features is None:
            raise ValueError("Features have not been selected. Call select_features() first.")
            
        # Use only selected features for evaluation
        X_selected = X[self.selected_features].values
        y_np = y.values
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y_np, y_pred)
        mae = mean_absolute_error(y_np, y_pred)
        r2 = r2_score(y_np, y_pred)
        
        # Calculate per-target metrics
        mae_per_target = mean_absolute_error(y_np, y_pred, multioutput='raw_values')
        r2_per_target = r2_score(y_np, y_pred, multioutput='raw_values')
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R²: {r2:.4f}")
        print("\nPer-target metrics:")
        for i, (mae_i, r2_i) in enumerate(zip(mae_per_target, r2_per_target)):
            print(f"Target {i+1}: MAE = {mae_i:.4f}, R² = {r2_i:.4f}")
        
        results = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mae_per_target': mae_per_target,
            'r2_per_target': r2_per_target
        }
        
        return results
    
    def plot_training_history(self):
        """Plot the XGBoost training history"""
        if self.history is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        n_outputs = len(self.model)
        fig, axes = plt.subplots(n_outputs, 2, figsize=(15, 4 * n_outputs))
        # If only one output, make axes 2D
        if n_outputs == 1:
            axes = np.array([axes])
        
        for i in range(n_outputs):
            # Plot loss
            ax1 = axes[i, 0]
            ax1.plot(self.history['train_loss'][i], label='Train')
            if 'val_loss' in self.history and self.history['val_loss']:
                ax1.plot(self.history['val_loss'][i], label='Validation')
            ax1.set_title(f'Model Loss (Target {i+1})')
            ax1.set_ylabel('Loss (MSE)')
            ax1.set_xlabel('Iteration')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot MAE
            ax2 = axes[i, 1]
            ax2.plot(self.history['train_mae'][i], label='Train')
            if 'val_mae' in self.history and self.history['val_mae']:
                ax2.plot(self.history['val_mae'][i], label='Validation')
            ax2.set_title(f'Mean Absolute Error (Target {i+1})')
            ax2.set_ylabel('MAE')
            ax2.set_xlabel('Iteration')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        
        Parameters:
        -----------
        top_n : int
            Number of top features to show
        
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get feature importances
        importance_df = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': self.feature_importances_
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        # Take top N features
        if len(importance_df) > top_n:
            importance_df = importance_df.head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return fig
