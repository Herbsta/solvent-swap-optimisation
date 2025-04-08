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


class FeatureEncoder:
    """Handles feature encoding strategies."""
    
    def __init__(self):
        """Initialize encoders dictionary."""
        self.encoders = {}
        self.encoded_features = {}
        self.feature_names = {}
        
    def fit_one_hot_encoder(self, 
                           data: pd.DataFrame, 
                           column: str, 
                           handle_unknown: str = 'ignore') -> None:
        """
        Fit one-hot encoder for a categorical column.
        
        Args:
            data: DataFrame containing the column
            column: Column name to encode
            handle_unknown: Strategy for handling unknown categories
        """
        encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
        
        # Reshape to 2D array for encoder
        X = data[[column]].values
        encoder.fit(X)
        
        # Store the encoder
        self.encoders[column] = {
            'type': 'one_hot',
            'encoder': encoder,
            'categories': encoder.categories_[0]
        }
        
    def transform_one_hot(self, 
                         data: pd.DataFrame, 
                         column: str) -> pd.DataFrame:
        """
        Transform data using fitted one-hot encoder.
        
        Args:
            data: DataFrame containing the column
            column: Column name to encode
            
        Returns:
            DataFrame with encoded features
        """
        if column not in self.encoders or self.encoders[column]['type'] != 'one_hot':
            raise ValueError(f"No one-hot encoder fitted for column {column}")
            
        encoder = self.encoders[column]['encoder']
        X = data[[column]].values
        encoded = encoder.transform(X)
        
        # Create feature names
        categories = self.encoders[column]['categories']
        feature_names = [f"{column}_{cat}" for cat in categories]
        
        # Create DataFrame with encoded values
        encoded_df = pd.DataFrame(
            encoded, 
            columns=feature_names,
            index=data.index
        )
        
        # Store encoded features and names
        self.encoded_features[column] = encoded_df
        self.feature_names[column] = feature_names
        
        return encoded_df
    
    def fit_transform_one_hot(self, 
                             data: pd.DataFrame, 
                             column: str,
                             handle_unknown: str = 'ignore') -> pd.DataFrame:
        """
        Fit and transform in one step for one-hot encoding.
        
        Args:
            data: DataFrame containing the column
            column: Column name to encode
            handle_unknown: Strategy for handling unknown categories
            
        Returns:
            DataFrame with encoded features
        """
        self.fit_one_hot_encoder(data, column, handle_unknown)
        return self.transform_one_hot(data, column)
    
    def fit_scaler(self, 
                  data: pd.DataFrame,
                  columns: List[str],
                  scaler_type: str = 'robust',
                  scaler_name: str = 'default') -> None:
        """
        Fit a scaler for numerical columns.
        
        Args:
            data: DataFrame containing the columns
            columns: Column names to scale
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            scaler_name: Name to identify this scaler
        """
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        # Fit the scaler
        X = data[columns].values
        scaler.fit(X)
        
        # Store the scaler
        self.encoders[scaler_name] = {
            'type': 'scaler',
            'scaler': scaler,
            'columns': columns
        }
        
    def transform_scaler(self, 
                        data: pd.DataFrame,
                        scaler_name: str = 'default') -> pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Args:
            data: DataFrame containing the columns
            scaler_name: Name of the scaler to use
            
        Returns:
            DataFrame with scaled features
        """
        if scaler_name not in self.encoders or self.encoders[scaler_name]['type'] != 'scaler':
            raise ValueError(f"No scaler fitted with name {scaler_name}")
            
        scaler_info = self.encoders[scaler_name]
        scaler = scaler_info['scaler']
        columns = scaler_info['columns']
        
        # Check if all columns exist in data
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in data")
            
        # Transform the data
        X = data[columns].values
        scaled = scaler.transform(X)
        
        # Create DataFrame with scaled values
        scaled_df = pd.DataFrame(
            scaled,
            columns=columns,
            index=data.index
        )
        
        # Store scaled features
        self.encoded_features[scaler_name] = scaled_df
        self.feature_names[scaler_name] = columns
        
        return scaled_df
    
    def fit_transform_scaler(self,
                            data: pd.DataFrame,
                            columns: List[str],
                            scaler_type: str = 'robust',
                            scaler_name: str = 'default') -> pd.DataFrame:
        """
        Fit and transform in one step for scaling.
        
        Args:
            data: DataFrame containing the columns
            columns: Column names to scale
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            scaler_name: Name to identify this scaler
            
        Returns:
            DataFrame with scaled features
        """
        self.fit_scaler(data, columns, scaler_type, scaler_name)
        return self.transform_scaler(data, scaler_name)
    
    def create_system_id(self,
                        data: pd.DataFrame,
                        columns: List[str],
                        new_column: str = 'system_id') -> pd.DataFrame:
        """
        Create a system ID by combining multiple columns.
        
        Args:
            data: DataFrame containing the columns
            columns: Column names to combine
            new_column: Name for the new system ID column
            
        Returns:
            DataFrame with new system ID column
        """
        # Check if all columns exist in data
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in data")
            
        # Create the system ID
        df = data.copy()
        df[new_column] = df[columns].astype(str).agg('-'.join, axis=1)
        
        return df
    
    def get_all_feature_names(self) -> List[str]:
        """
        Get all feature names after encoding.
        
        Returns:
            List of feature names
        """
        all_names = []
        for names in self.feature_names.values():
            if isinstance(names, list):
                all_names.extend(names)
            else:
                all_names.append(names)
                
        return all_names


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