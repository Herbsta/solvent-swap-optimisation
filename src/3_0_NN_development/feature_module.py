"""
Feature engineering module for solvent swap optimization.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureProcessor:
    def __init__(self, categorical_column='system'):
        """
        Initialize feature processor for encoding categorical data and scaling numerical data.
        
        Parameters:
        -----------
        categorical_column : str
            The categorical column to be one-hot encoded
        """
        self.categorical_column = categorical_column
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.input_scalar = RobustScaler()
        self.output_scalar = RobustScaler()
        self.fitted = False
    
    def fit(self, X, y=None):
        """Fit the encoder and scalers to the data"""
        # Fit encoder on categorical column
        self.encoder.fit(X[[self.categorical_column]])
        
        # Fit scaler on numerical columns
        self.input_scalar.fit(X.drop(columns=[self.categorical_column]))
        
        # Fit output scaler if target is provided
        if y is not None:
            self.output_scalar.fit(y)
        
        self.fitted = True
        return self
    
    def transform_inputs(self, X):
        """Transform input features using fitted encoder and scaler"""
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted first")
        
        # Encode categorical column
        cat_encoded = self.encoder.transform(X[[self.categorical_column]])
        cat_encoded_df = pd.DataFrame(
            cat_encoded,
            columns=[f"{self.categorical_column}_{val}" for val in self.encoder.categories_[0]],
            index=X.index
        )
        
        # Scale numerical columns
        num_scaled = self.input_scalar.transform(X.drop(columns=[self.categorical_column]))
        num_scaled_df = pd.DataFrame(
            num_scaled,
            columns=X.drop(columns=[self.categorical_column]).columns,
            index=X.index
        )
        
        # Combine encoded and scaled data
        return pd.concat([num_scaled_df, cat_encoded_df], axis=1)
    
    def transform_outputs(self, y):
        """Scale target values"""
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted first")
        
        y_scaled = self.output_scalar.transform(y)
        return pd.DataFrame(
            y_scaled,
            columns=y.columns,
            index=y.index
        )
    
    def inverse_transform_outputs(self, y_scaled):
        """Inverse transform scaled target values back to original scale"""
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted first")
        
        y_original = self.output_scalar.inverse_transform(y_scaled)
        return pd.DataFrame(
            y_original,
            columns=y_scaled.columns,
            index=y_scaled.index
        )

class FeatureImportance:
    """Analyzes feature importance for different model types."""
    
    @staticmethod
    def compute_permutation_importance(model, X, y, 
                                     n_repeats=10, 
                                     random_state=42) -> pd.DataFrame:
        """
        Compute permutation feature importance.
        
        Args:
            model: Trained model with predict method
            X: Feature data
            y: Target data
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with feature importance scores
        """
        # Convert to numpy for consistency
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.DataFrame) else y
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_array, y_array, 
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        # Create DataFrame
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X_array.shape[1])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    @staticmethod
    def get_model_feature_importance(model, feature_names=None) -> pd.DataFrame:
        """
        Extract feature importance directly from model if available.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
            
        importances = model.feature_importances_
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame,
                              top_n: int = 20,
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            figsize: Figure size (width, height)
        """
        # Take top N features
        top_features = importance_df.head(top_n)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Check which importance metric is available
        if 'importance_mean' in importance_df.columns:
            # Plot with error bars if we have std deviation
            if 'importance_std' in importance_df.columns:
                sns.barplot(
                    x='importance_mean',
                    y='feature',
                    xerr=top_features['importance_std'],
                    data=top_features
                )
                plt.xlabel('Mean Importance (± StdDev)')
            else:
                sns.barplot(
                    x='importance_mean',
                    y='feature',
                    data=top_features
                )
                plt.xlabel('Mean Importance')
        else:
            # Direct importance values
            sns.barplot(
                x='importance',
                y='feature',
                data=top_features
            )
            plt.xlabel('Importance')
            
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def feature_selection(X, y, 
                        k: int = 10,
                        method: str = 'f_regression') -> pd.DataFrame:
        """
        Select top k features based on statistical tests.
        
        Args:
            X: Feature data
            y: Target data
            k: Number of features to select
            method: Method to use ('f_regression', 'mutual_info')
            
        Returns:
            DataFrame with selected features and scores
        """
        # Choose the scoring function
        if method == 'f_regression':
            score_func = f_regression
        elif method == 'mutual_info':
            score_func = mutual_info_regression
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Convert to numpy for consistency
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.DataFrame) else y
        
        # Get feature names
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X_array.shape[1])]
        
        # Apply feature selection
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X_array, y_array)
        
        # Get scores and create DataFrame
        scores = selector.scores_
        
        # Create results DataFrame
        results = pd.DataFrame({
            'feature': feature_names,
            'score': scores,
            'selected': selector.get_support()
        })
        
        # Sort by score
        results = results.sort_values('score', ascending=False)
        
        return results