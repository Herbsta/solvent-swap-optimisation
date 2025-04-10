import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.utils import use_named_args

class BaseModelWithFeatureSelection(ABC):
    def __init__(self, feature_selection_method='correlation', n_features=50, keep_prefixes=None):
        """
        Base class for models with feature selection capabilities.
        
        Parameters:
        -----------
        feature_selection_method : str
            Method for feature selection: 'correlation', 'f_regression', 'rfe', 'random_forest'
        n_features : int
            Number of features to select
        keep_prefixes : list
            List of column prefixes to always keep regardless of feature selection
        """
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.keep_prefixes = keep_prefixes or []
        self.feature_selector = None
        self.selected_features = None
        self.model = None
        self.history = None
        self.best_params = None
    
    def select_features(self, X, y):
        """Select features based on specified method"""
        print(f"Selecting top {self.n_features} features using {self.feature_selection_method} method...")
        
        # First, identify columns to always keep based on prefixes
        always_keep = []
        for prefix in self.keep_prefixes:
            always_keep.extend([col for col in X.columns if col.startswith(prefix)])
        always_keep = list(set(always_keep))  # Remove duplicates
        
        remaining_features = self.n_features
        
        # Get columns that are not in always_keep for selection
        eligible_columns = [col for col in X.columns if col not in always_keep]
        X_eligible = X[eligible_columns]
        
        selected_eligible = []
        if remaining_features > 0 and len(eligible_columns) > 0:
            if self.feature_selection_method == 'correlation':
                # Calculate correlation between features and targets
                correlations = []
                for column in eligible_columns:
                    corr = abs(np.mean([abs(np.corrcoef(X[column], y[target])[0, 1]) for target in y.columns]))
                    correlations.append((column, corr))
                
                # Sort by absolute correlation and select top remaining_features
                selected_eligible = [col for col, _ in sorted(correlations, key=lambda x: x[1], reverse=True)[:remaining_features]]
                
            elif self.feature_selection_method == 'f_regression':
                # Use f_regression for feature selection
                selector = SelectKBest(score_func=f_regression, k=min(remaining_features, len(eligible_columns)))
                selector.fit(X_eligible, y.values)
                self.feature_selector = selector
                selected_eligible = X_eligible.columns[selector.get_support()].tolist()
                
            elif self.feature_selection_method == 'rfe':
                # Recursive Feature Elimination
                base_model = RandomForestRegressor(n_estimators=100, random_state=42)
                selector = RFE(estimator=base_model, n_features_to_select=min(remaining_features, len(eligible_columns)), step=10)
                selector.fit(X_eligible, y.values)
                self.feature_selector = selector
                selected_eligible = X_eligible.columns[selector.get_support()].tolist()
                
            elif self.feature_selection_method == 'random_forest':
                # Random Forest feature importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_eligible, y.values)
                importance = rf.feature_importances_
                indices = np.argsort(importance)[::-1][:remaining_features]
                selected_eligible = X_eligible.columns[indices].tolist()
        
        # Combine always_keep and selected_eligible
        self.selected_features = always_keep + selected_eligible
        
        print(f"Selected {len(self.selected_features)} features ({len(always_keep)} from prefixes, {len(selected_eligible)} from selection)")
        return X[self.selected_features]
    
    def get_feature_importances(self):
        """
        Get feature importances based on selected method
        
        Returns:
        --------
        importances : DataFrame
            DataFrame with feature names and importance scores
        """
        if self.selected_features is None:
            raise ValueError("Features have not been selected. Call select_features() first.")
        
        # For now, just return the selected features
        importances = pd.DataFrame({
            'Feature': self.selected_features,
            'Selected': True
        })
        
        return importances
    
    @abstractmethod
    def _build_model(self, input_dim, output_dim, **params):
        """
        Build the model architecture
        
        Parameters:
        -----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        params : dict
            Additional parameters for model building
            
        Returns:
        --------
        model : object
            Model object
        """
        pass
    
    @abstractmethod
    def _define_parameter_space(self):
        """
        Define the parameter space for hyperparameter optimization
        
        Returns:
        --------
        space : list
            List of parameter space definitions
        """
        pass
    
    def optimize_hyperparameters(self, X, y, validation_data=None, n_calls=50):
        """
        Optimize hyperparameters using Bayesian Optimization
        
        Parameters:
        -----------
        X : DataFrame
            Feature DataFrame
        y : DataFrame
            Target DataFrame
        validation_data : tuple
            Tuple of (X_val, y_val) for validation
        n_calls : int
            Number of iterations for optimization
        
        Returns:
        --------
        best_params : dict
            Dictionary of best hyperparameters
        """
        print("Starting Bayesian hyperparameter optimization...")
        
        # Define the search space
        space = self._define_parameter_space()
        
        # Prepare validation data
        if validation_data is not None:
            X_val, y_val = validation_data
        else:
            # If no validation data is provided, split the input data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            X, y = X_train, y_train
        
        # Define the objective function
        @use_named_args(space)
        def objective(**params):
            # Build the model with current hyperparameters
            model = self._build_model(
                input_dim=X.shape[1], 
                output_dim=y.shape[1],
                **params
            )
            
            # Train and evaluate the model (implementation specific to each model type)
            val_loss = self._evaluate_model_for_optimization(model, X, y, X_val, y_val, params)
            
            return val_loss
        
        # Run the optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=True
        )
        
        # Extract and store the best parameters (implementation specific)
        self.best_params = self._extract_best_params(result)
        
        print(f"Best hyperparameters found: {self.best_params}")
        return self.best_params
    
    @abstractmethod
    def _evaluate_model_for_optimization(self, model, X_train, y_train, X_val, y_val, params):
        """
        Evaluate model during hyperparameter optimization
        
        Parameters:
        -----------
        model : object
            Model object
        X_train : DataFrame
            Training features
        y_train : DataFrame
            Training targets
        X_val : DataFrame
            Validation features
        y_val : DataFrame
            Validation targets
        params : dict
            Model parameters
            
        Returns:
        --------
        val_loss : float
            Validation loss to minimize
        """
        pass
    
    @abstractmethod
    def _extract_best_params(self, optimization_result):
        """
        Extract best parameters from optimization result
        
        Parameters:
        -----------
        optimization_result : object
            Result object from optimization
            
        Returns:
        --------
        best_params : dict
            Dictionary of best parameters
        """
        pass
    
    @abstractmethod
    def train(self, X, y, validation_data=None, **kwargs):
        """
        Train the model
        
        Parameters:
        -----------
        X : DataFrame
            Feature DataFrame
        y : DataFrame
            Target DataFrame
        validation_data : tuple
            Tuple of (X_val, y_val) for validation
        kwargs : dict
            Additional parameters for training
            
        Returns:
        --------
        history : object
            Training history
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : DataFrame
            Feature DataFrame
            
        Returns:
        --------
        predictions : ndarray
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on test data
        
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
        pass
    
    @abstractmethod
    def plot_training_history(self):
        """
        Plot the training history
        
        Returns:
        --------
        fig : matplotlib Figure
            Figure with training history plots
        """
        pass
