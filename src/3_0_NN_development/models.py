"""
Enhanced model implementation module for solvent swap optimization.
Includes XGBoost and Variational Autoencoder (VAE) models.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import pickle
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Layer
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str = "base_model"):
        """
        Initialize base model.
        
        Args:
            name: Model name
        """
        self.name = name
        self.model = None
        self.feature_names = None
        self.target_names = None
        self.history = None
        
    @abstractmethod
    def build(self, input_dim: int, output_dim: int) -> None:
        """
        Build the model architecture.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
        """
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        pass
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with performance metrics
        """
        y_pred = self.predict(X_test)
        
        # Convert to numpy arrays if needed
        y_test_np = y_test.values if isinstance(y_test, pd.DataFrame) else y_test
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test_np, y_pred),
            'mse': mean_squared_error(y_test_np, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_np, y_pred)),
            'r2': r2_score(y_test_np, y_pred)
        }
        
        # If we have multiple targets, calculate metrics for each
        if y_test_np.ndim > 1 and y_test_np.shape[1] > 1:
            for i, target in enumerate(self.target_names):
                metrics[f'mae_{target}'] = mean_absolute_error(y_test_np[:, i], y_pred[:, i])
                metrics[f'mse_{target}'] = mean_squared_error(y_test_np[:, i], y_pred[:, i])
                metrics[f'rmse_{target}'] = np.sqrt(mean_squared_error(y_test_np[:, i], y_pred[:, i]))
                metrics[f'r2_{target}'] = r2_score(y_test_np[:, i], y_pred[:, i])
                
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Directory path where the model is saved
        """
        # Load metadata
        with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            
        self.name = metadata['name']
        self.feature_names = metadata['feature_names']
        self.target_names = metadata['target_names']
    
    def plot_predictions(self, X_test, y_test, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            X_test: Test features
            y_test: Test targets
            figsize: Figure size (width, height)
        """
        y_pred = self.predict(X_test)
        
        # Convert to numpy arrays if needed
        y_test_np = y_test.values if isinstance(y_test, pd.DataFrame) else y_test
        
        # Get target names
        target_names = self.target_names or [f"Target {i+1}" for i in range(y_test_np.shape[1])]
        
        # Create figure
        fig, axes = plt.subplots(1, len(target_names), figsize=figsize)
        
        # If there's only one target, axes is not iterable
        if len(target_names) == 1:
            axes = [axes]
            
        for i, (ax, target) in enumerate(zip(axes, target_names)):
            ax.scatter(y_test_np[:, i], y_pred[:, i], alpha=0.5)
            
            # Add diagonal line (perfect predictions)
            min_val = min(y_test_np[:, i].min(), y_pred[:, i].min())
            max_val = max(y_test_np[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{target}')
            
            # Add metrics
            mae = mean_absolute_error(y_test_np[:, i], y_pred[:, i])
            r2 = r2_score(y_test_np[:, i], y_pred[:, i])
            ax.text(0.05, 0.95, f'MAE: {mae:.4f}\nR²: {r2:.4f}', 
                   transform=ax.transAxes, verticalalignment='top')
            
        plt.tight_layout()
        plt.show()


class NeuralNetworkModel(BaseModel):
    """Neural network model implementation."""
    
    def __init__(self, name: str = "neural_network", **kwargs):
        """
        Initialize neural network model.
        
        Args:
            name: Model name
            **kwargs: Additional parameters for model configuration
        """
        super().__init__(name)
        self.config = {
            'hidden_layers': [256, 128, 64],
            'dropout_rates': [0.3, 0.2, 0.2],
            'activation': 'relu',
            'output_activation': None,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 20,
            'optimizer': 'adam'
        }
        
        # Update with provided kwargs
        self.config.update(kwargs)
        
    def build(self, input_dim: int, output_dim: int) -> None:
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.config['hidden_layers'][0], 
            activation=self.config['activation'],
            input_dim=input_dim
        ))
        
        model.add(Dropout(self.config['dropout_rates'][0]))
        
        # Hidden layers
        for i in range(1, len(self.config['hidden_layers'])):
            model.add(Dense(
                self.config['hidden_layers'][i],
                activation=self.config['activation']
            ))
            
            if i < len(self.config['dropout_rates']):
                model.add(Dropout(self.config['dropout_rates'][i]))
        
        # Output layer
        model.add(Dense(output_dim, activation=self.config['output_activation']))
        
        # Configure optimizer
        if self.config['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config['learning_rate'])
        else:
            optimizer = self.config['optimizer']
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.build(input_dim, output_dim)
            
        # Store feature and target names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        if isinstance(y_train, pd.DataFrame):
            self.target_names = y_train.columns.tolist()
            
        # Update training parameters with kwargs
        train_params = {
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'verbose': 1
        }
        train_params.update(kwargs)
        
        # Setup early stopping
        callbacks = []
        if self.config['patience'] > 0:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['patience'],
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
        # Prepare validation data if provided
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            callbacks=callbacks,
            **train_params
        )
        
        self.history = history.history
        return self.history
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions with the neural network.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
            
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the neural network to disk.
        
        Args:
            path: Directory path to save the model
        """
        super().save(path)
        
        # Save Keras model
        if self.model is not None:
            self.model.save(os.path.join(path, 'keras_model.h5'))
            
        # Save history
        if self.history is not None:
            with open(os.path.join(path, 'history.pkl'), 'wb') as f:
                pickle.dump(self.history, f)
                
        # Save config
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
    
    def load(self, path: str) -> None:
        """
        Load the neural network from disk.
        
        Args:
            path: Directory path where the model is saved
        """
        super().load(path)
        
        # Load Keras model
        model_path = os.path.join(path, 'keras_model.h5')
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            
        # Load history
        history_path = os.path.join(path, 'history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)
                
        # Load config
        config_path = os.path.join(path, 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot training history.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.history is None:
            raise ValueError("No training history available")
            
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Over Training')
        axes[0].legend()
        
        # Plot MAE
        axes[1].plot(self.history['mae'], label='Training MAE')
        if 'val_mae' in self.history:
            axes[1].plot(self.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Mean Absolute Error Over Training')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self, name: str = "xgboost", **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name
            **kwargs: Additional parameters for model configuration
        """
        super().__init__(name)
        self.config = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'verbosity': 1
        }
        
        # Update with provided kwargs
        self.config.update(kwargs)
        
    def build(self, input_dim: int, output_dim: int) -> None:
        """
        Build the XGBoost model.
        
        Args:
            input_dim: Number of input features (not used)
            output_dim: Number of output targets
        """
        # For multiple targets, we need to create multiple models
        if output_dim > 1:
            self.model = [
                xgb.XGBRegressor(**self.config)
                for _ in range(output_dim)
            ]
        else:
            self.model = xgb.XGBRegressor(**self.config)
        
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        # Store feature and target names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        if isinstance(y_train, pd.DataFrame):
            self.target_names = y_train.columns.tolist()
        
        # Convert to numpy arrays if needed
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        
        # Determine output dimension
        output_dim = y_train_np.shape[1] if len(y_train_np.shape) > 1 else 1
        
        # Build model if not already built
        if self.model is None:
            self.build(X_train_np.shape[1], output_dim)
            
        # Prepare eval set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.DataFrame) else y_val
            
        # Train model(s)
        self.history = {}
        
        if output_dim > 1:
            for i in range(output_dim):
                target_name = self.target_names[i] if self.target_names else f"target_{i}"
                
                # Prepare eval set for this target
                if eval_set is not None:
                    this_eval_set = [(X_val_np, y_val_np[:, i])]
                else:
                    this_eval_set = None
                    
                # Train model
                model_result = self.model[i].fit(
                    X_train_np, 
                    y_train_np[:, i],
                    eval_set=this_eval_set,
                    verbose=self.config['verbosity'],
                    **kwargs
                )
                
                # Store evaluation results if available
                if hasattr(model_result, 'evals_result'):
                    self.history[f'target_{i}'] = model_result.evals_result()
        else:
            # Prepare eval set for single target
            if eval_set is not None:
                this_eval_set = [(X_val_np, y_val_np)]
            else:
                this_eval_set = None
                
            # Train model
            model_result = self.model.fit(
                X_train_np, 
                y_train_np,
                eval_set=this_eval_set,
                verbose=self.config['verbosity'],
                **kwargs
            )
            
            # Store evaluation results if available
            if hasattr(model_result, 'evals_result'):
                self.history['target'] = model_result.evals_result()
            
        return self.history
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions with the XGBoost model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
            
        # Convert to numpy array if needed
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Check if we have multiple models (for multiple targets)
        if isinstance(self.model, list):
            # Predict for each target
            predictions = np.column_stack([
                model.predict(X_np) for model in self.model
            ])
            return predictions
        else:
            # Single target prediction
            return self.model.predict(X_np).reshape(-1, 1)
    
    def save(self, path: str) -> None:
        """
        Save the XGBoost model to disk.
        
        Args:
            path: Directory path to save the model
        """
        super().save(path)
        
        # Save model
        if self.model is not None:
            if isinstance(self.model, list):
                # Save each model separately
                for i, model in enumerate(self.model):
                    model.save_model(os.path.join(path, f'xgb_model_{i}.json'))
            else:
                self.model.save_model(os.path.join(path, 'xgb_model.json'))
                
        # Save history
        if self.history is not None:
            with open(os.path.join(path, 'history.pkl'), 'wb') as f:
                pickle.dump(self.history, f)
                
        # Save config
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
    
    def load(self, path: str) -> None:
        """
        Load the XGBoost model from disk.
        
        Args:
            path: Directory path where the model is saved
        """
        super().load(path)
        
        # Check if we have multiple models
        model_files = [f for f in os.listdir(path) if f.startswith('xgb_model_') and f.endswith('.json')]
        
        if model_files:
            # Load multiple models
            self.model = []
            for i in range(len(model_files)):
                model = xgb.XGBRegressor(**self.config)
                model.load_model(os.path.join(path, f'xgb_model_{i}.json'))
                self.model.append(model)
        else:
            # Load single model
            model_path = os.path.join(path, 'xgb_model.json')
            if os.path.exists(model_path):
                self.model = xgb.XGBRegressor(**self.config)
                self.model.load_model(model_path)
                
        # Load history
        history_path = os.path.join(path, 'history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)
                
        # Load config
        config_path = os.path.join(path, 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the XGBoost model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
            
        # Get feature names
        feature_names = self.feature_names or [f"feature_{i}" for i in range(
            self.model[0].feature_importances_.shape[0] if isinstance(self.model, list) 
            else self.model.feature_importances_.shape[0]
        )]
        
        # Get target names
        target_names = self.target_names or [f"target_{i}" for i in range(len(self.model) if isinstance(self.model, list) else 1)]
        
        # Extract importance
        if isinstance(self.model, list):
            # Multiple targets
            importance_df = pd.DataFrame()
            
            for i, model in enumerate(self.model):
                target = target_names[i]
                df = pd.DataFrame({
                    'feature': feature_names,
                    f'importance_{target}': model.feature_importances_
                })
                
                if importance_df.empty:
                    importance_df = df
                else:
                    importance_df = importance_df.merge(df, on='feature')
                    
            # Add average importance
            importance_columns = [col for col in importance_df.columns if col.startswith('importance_')]
            importance_df['importance_mean'] = importance_df[importance_columns].mean(axis=1)
            
            # Sort by average importance
            importance_df = importance_df.sort_values('importance_mean', ascending=False)
            
        else:
            # Single target
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size (width, height)
        """
        importance_df = self.get_feature_importance()
        
        # Take top N features
        top_features = importance_df.head(top_n)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Check which importance metric is available
        if 'importance_mean' in importance_df.columns:
            plt.barh(top_features['feature'], top_features['importance_mean'])
            plt.xlabel('Mean Importance')
        else:
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance')
            
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()  # To show highest importance at top
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curves(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot learning curves from training history.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.history:
            raise ValueError("No training history available")
            
        # Determine how many targets we have
        target_keys = list(self.history.keys())
        num_targets = len(target_keys)
        
        if num_targets == 0:
            raise ValueError("No evaluation metrics found in history")
            
        # Create figure
        fig, axes = plt.subplots(num_targets, 1, figsize=figsize)
        
        # If only one target, make axes iterable
        if num_targets == 1:
            axes = [axes]
            
        # Plot learning curves for each target
        for i, target_key in enumerate(target_keys):
            target_history = self.history[target_key]
            
            # Get evaluation metrics (assumes validation dataset was used during training)
            eval_metrics = list(target_history.keys())
            metric_values = list(target_history.values())
            
            if not eval_metrics:
                continue
                
            # Plot each metric
            ax = axes[i]
            
            for j, metric in enumerate(eval_metrics):
                values = metric_values[j][0] if isinstance(metric_values[j], list) else metric_values[j]
                ax.plot(values, label=f"{metric}")
                
            ax.set_title(f"Learning Curves for {target_key}")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Metric Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()


class SamplingLayer(Layer):
    """Custom layer for VAE sampling."""
    
    def __init__(self, **kwargs):
        super(SamplingLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAEModel(BaseModel):
    """Variational Autoencoder model implementation for feature representation and regression."""
    
    def __init__(self, name: str = "vae", **kwargs):
        """
        Initialize VAE model.
        
        Args:
            name: Model name
            **kwargs: Additional parameters for model configuration
        """
        super().__init__(name)
        self.config = {
            # Encoder
            'encoder_hidden_layers': [128, 64],
            'encoder_activation': 'relu',
            'latent_dim': 16,
            # Decoder
            'decoder_hidden_layers': [64, 128],
            'decoder_activation': 'relu',
            # Regression
            'regression_hidden_layers': [64, 32],
            'regression_activation': 'relu',
            # Training
            'learning_rate': 0.001,
            'kl_weight': 0.1,
            'epochs': 100,
            'batch_size': 32,
            'patience': 20,
            'optimizer': 'adam'
        }
        
        # Update with provided kwargs
        self.config.update(kwargs)
        
        # Model components
        self.encoder = None
        self.decoder = None
        self.regression_model = None
        self.vae = None
        self.input_dim = None
        self.output_dim = None
        
    def build_encoder(self, input_dim: int, latent_dim: int) -> Model:
        """
        Build the encoder part of VAE.
        
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space
            
        Returns:
            Encoder model
        """
        # Input layer
        inputs = Input(shape=(input_dim,), name='encoder_input')
        
        # Hidden layers
        x = inputs
        for i, units in enumerate(self.config['encoder_hidden_layers']):
            x = Dense(units, activation=self.config['encoder_activation'], 
                     name=f'encoder_dense_{i}')(x)
        
        # Latent space parameters
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        z = SamplingLayer(name='z')([z_mean, z_log_var])
        
        # Define encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        return encoder
    
    def build_decoder(self, latent_dim: int, output_dim: int) -> Model:
        """
        Build the decoder part of VAE.
        
        Args:
            latent_dim: Dimension of latent space
            output_dim: Dimension of output space (original input dimension)
            
        Returns:
            Decoder model
        """
        # Input layer (latent space)
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        
        # Hidden layers
        x = latent_inputs
        for i, units in enumerate(self.config['decoder_hidden_layers']):
            x = Dense(units, activation=self.config['decoder_activation'], 
                     name=f'decoder_dense_{i}')(x)
        
        # Output layer
        outputs = Dense(output_dim, name='decoder_output')(x)
        
        # Define decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        
        return decoder
    
    def build_regression_model(self, latent_dim: int, output_dim: int) -> Model:
        """
        Build regression model that predicts targets from latent space.
        
        Args:
            latent_dim: Dimension of latent space
            output_dim: Number of output targets
            
        Returns:
            Regression model
        """
        # Input layer (latent space)
        latent_inputs = Input(shape=(latent_dim,), name='regression_input')
        
        # Hidden layers
        x = latent_inputs
        for i, units in enumerate(self.config['regression_hidden_layers']):
            x = Dense(units, activation=self.config['regression_activation'], 
                     name=f'regression_dense_{i}')(x)
        
        # Output layer
        outputs = Dense(output_dim, name='regression_output')(x)
        
        # Define regression model
        regression_model = Model(latent_inputs, outputs, name='regression')
        
        return regression_model
    
    def build(self, input_dim: int, output_dim: int) -> None:
        """
        Build the VAE model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        latent_dim = self.config['latent_dim']
        
        # Build encoder
        self.encoder = self.build_encoder(input_dim, latent_dim)
        
        # Build decoder (for reconstruction)
        self.decoder = self.build_decoder(latent_dim, input_dim)
        
        # Build regression model
        self.regression_model = self.build_regression_model(latent_dim, output_dim)
        
        # Set up full VAE model
        inputs = Input(shape=(input_dim,))
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        predictions = self.regression_model(z)
        
        # Define custom loss
        def vae_loss(x, x_decoded_mean):
            # Reconstruction loss
            reconstruction_loss = mse(x, x_decoded_mean)
            reconstruction_loss *= input_dim
            
            # KL divergence loss
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5 * self.config['kl_weight']
            
            return K.mean(reconstruction_loss + kl_loss)
            
        # Create VAE model
        vae = Model(inputs, [reconstructed, predictions])
        
        # Configure optimizer
        if self.config['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config['learning_rate'])
        else:
            optimizer = self.config['optimizer']
            
        # Compile model
        vae.compile(
            optimizer=optimizer,
            loss=[vae_loss, 'mse'],
            loss_weights=[1.0, 1.0],
            metrics={'decoder': 'mse', 'regression': 'mae'}
        )
        
        self.vae = vae
        self.model = vae  # For compatibility with base class
        
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict:
        """
        Train the VAE model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history
        """
        if self.vae is None:
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.build(input_dim, output_dim)
            
        # Store feature and target names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        if isinstance(y_train, pd.DataFrame):
            self.target_names = y_train.columns.tolist()
            
        # Convert to numpy if needed
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        
        # Update training parameters with kwargs
        train_params = {
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'verbose': 1
        }
        train_params.update(kwargs)
        
        # Setup early stopping
        callbacks = []
        if self.config['patience'] > 0:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['patience'],
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.DataFrame) else y_val
            validation_data = (X_val_np, [X_val_np, y_val_np])
        
        # Train the model
        history = self.vae.fit(
            X_train_np,
            [X_train_np, y_train_np],
            validation_data=validation_data,
            callbacks=callbacks,
            **train_params
        )
        
        self.history = history.history
        return self.history
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions with the VAE model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values (output of regression model)
        """
        if self.vae is None:
            raise ValueError("Model not built or trained yet")
            
        # Convert to numpy if needed
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Get predictions
        _, predictions = self.vae.predict(X_np)
        
        return predictions
    
    def encode(self, X) -> np.ndarray:
        """
        Encode input data to latent space.
        
        Args:
            X: Input features
            
        Returns:
            Encoded representation (z)
        """
        if self.encoder is None:
            raise ValueError("Encoder not built or trained yet")
            
        # Convert to numpy if needed
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Encode
        z_mean, _, _ = self.encoder.predict(X_np)
        
        return z_mean
    
    def decode(self, z) -> np.ndarray:
        """
        Decode latent representation to input space.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        if self.decoder is None:
            raise ValueError("Decoder not built or trained yet")
            
        # Decode
        return self.decoder.predict(z)
    
    def reconstruct(self, X) -> np.ndarray:
        """
        Reconstruct input data through the autoencoder.
        
        Args:
            X: Input features
            
        Returns:
            Reconstructed input
        """
        if self.vae is None:
            raise ValueError("VAE not built or trained yet")
            
        # Convert to numpy if needed
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Get reconstruction
        reconstruction, _ = self.vae.predict(X_np)
        
        return reconstruction
    
    def save(self, path: str) -> None:
        """
        Save the VAE model to disk.
        
        Args:
            path: Directory path to save the model
        """
        super().save(path)
        
        # Save model components
        if self.encoder is not None:
            self.encoder.save(os.path.join(path, 'encoder.h5'))
            
        if self.decoder is not None:
            self.decoder.save(os.path.join(path, 'decoder.h5'))
            
        if self.regression_model is not None:
            self.regression_model.save(os.path.join(path, 'regression.h5'))
            
        if self.vae is not None:
            self.vae.save(os.path.join(path, 'vae.h5'))
            
        # Save history
        if self.history is not None:
            with open(os.path.join(path, 'history.pkl'), 'wb') as f:
                pickle.dump(self.history, f)
                
        # Save config
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
            
        # Save dimensions
        with open(os.path.join(path, 'dimensions.pkl'), 'wb') as f:
            pickle.dump({
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'latent_dim': self.config['latent_dim']
            }, f)
    
    def load(self, path: str) -> None:
        """
        Load the VAE model from disk.
        
        Args:
            path: Directory path where the model is saved
        """
        super().load(path)
        
        # Load dimensions
        with open(os.path.join(path, 'dimensions.pkl'), 'rb') as f:
            dimensions = pickle.load(f)
            self.input_dim = dimensions['input_dim']
            self.output_dim = dimensions['output_dim']
            self.config['latent_dim'] = dimensions['latent_dim']
            
        # Load model components
        encoder_path = os.path.join(path, 'encoder.h5')
        if os.path.exists(encoder_path):
            self.encoder = load_model(encoder_path, custom_objects={'SamplingLayer': SamplingLayer})
            
        decoder_path = os.path.join(path, 'decoder.h5')
        if os.path.exists(decoder_path):
            self.decoder = load_model(decoder_path)
            
        regression_path = os.path.join(path, 'regression.h5')
        if os.path.exists(regression_path):
            self.regression_model = load_model(regression_path)
            
        vae_path = os.path.join(path, 'vae.h5')
        if os.path.exists(vae_path):
            # Define custom loss function for loading
            def vae_loss(x, x_decoded_mean):
                return K.mean(mse(x, x_decoded_mean)) * 0.5
                
            self.vae = load_model(vae_path, custom_objects={
                'SamplingLayer': SamplingLayer,
                'vae_loss': vae_loss
            })
            self.model = self.vae
            
        # Load history
        history_path = os.path.join(path, 'history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)
                
        # Load config
        config_path = os.path.join(path, 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                loaded_config = pickle.load(f)
                # Update config while preserving latent_dim from dimensions
                latent_dim = self.config['latent_dim'] 
                self.config.update(loaded_config)
                self.config['latent_dim'] = latent_dim
    
    def plot_training_history(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        """
        Plot training history.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.history is None:
            raise ValueError("No training history available")
            
        # Create figure with 4 subplots (total loss, reconstruction loss, KL loss, regression loss)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot total loss
        axes[0, 0].plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        
        # Plot reconstruction loss
        if 'decoder_loss' in self.history:
            axes[0, 1].plot(self.history['decoder_loss'], label='Training')
            if 'val_decoder_loss' in self.history:
                axes[0, 1].plot(self.history['val_decoder_loss'], label='Validation')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Reconstruction Loss')
            axes[0, 1].legend()
        
        # Plot decoder MSE
        if 'decoder_mse' in self.history:
            axes[1, 0].plot(self.history['decoder_mse'], label='Training')
            if 'val_decoder_mse' in self.history:
                axes[1, 0].plot(self.history['val_decoder_mse'], label='Validation')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MSE')
            axes[1, 0].set_title('Reconstruction MSE')
            axes[1, 0].legend()
        
        # Plot regression MAE
        if 'regression_mae' in self.history:
            axes[1, 1].plot(self.history['regression_mae'], label='Training')
            if 'val_regression_mae' in self.history:
                axes[1, 1].plot(self.history['val_regression_mae'], label='Validation')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].set_title('Regression MAE')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_latent_space(self, X, y=None, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot data in 2D latent space if latent_dim >= 2.
        If latent_dim > 2, use PCA or t-SNE to reduce to 2D.
        
        Args:
            X: Input features
            y: Target values (for coloring)
            figsize: Figure size (width, height)
        """
        if self.encoder is None:
            raise ValueError("Encoder not built or trained yet")
            
        # Convert to numpy if needed
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Get latent representations
        z_mean, _, _ = self.encoder.predict(X_np)
        
        # If latent_dim > 2, reduce to 2D
        if z_mean.shape[1] > 2:
            from sklearn.decomposition import PCA
            
            # Use PCA to reduce to 2D
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_mean)
            
            plt.figure(figsize=figsize)
            scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y if y is not None else 'blue', alpha=0.5)
            
            if y is not None:
                plt.colorbar(scatter, label='Target Value')
                
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title(f'PCA of {z_mean.shape[1]}-D Latent Space')
            
        # If latent_dim is 2, plot directly
        elif z_mean.shape[1] == 2:
            plt.figure(figsize=figsize)
            scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y if y is not None else 'blue', alpha=0.5)
            
            if y is not None:
                plt.colorbar(scatter, label='Target Value')
                
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title('2D Latent Space')
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_reconstruction(self, X, n_samples: int = 5, figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Plot original vs reconstructed features for a few samples.
        
        Args:
            X: Input features
            n_samples: Number of samples to plot
            figsize: Figure size (width, height)
        """
        if self.vae is None:
            raise ValueError("VAE not built or trained yet")
            
        # Convert to numpy if needed
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Get random samples
        indices = np.random.choice(X_np.shape[0], size=n_samples, replace=False)
        X_samples = X_np[indices]
        
        # Get reconstructions
        reconstructions, _ = self.vae.predict(X_samples)
        
        # Get feature names
        feature_names = self.feature_names or [f"Feature {i+1}" for i in range(X_np.shape[1])]
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
        
        # If only one sample, make axes iterable
        if n_samples == 1:
            axes = [axes]
            
        # Plot each sample
        for i, ax in enumerate(axes):
            # Original features
            ax.plot(range(len(feature_names)), X_samples[i], 'b-', label='Original')
            
            # Reconstructed features
            ax.plot(range(len(feature_names)), reconstructions[i], 'r--', label='Reconstructed')
            
            # Add x-ticks for feature names
            if len(feature_names) <= 20:
                ax.set_xticks(range(len(feature_names)))
                ax.set_xticklabels(feature_names, rotation=90)
            else:
                # Show fewer ticks if too many features
                step = len(feature_names) // 20 + 1
                ax.set_xticks(range(0, len(feature_names), step))
                ax.set_xticklabels([feature_names[j] for j in range(0, len(feature_names), step)], rotation=90)
                
            ax.set_title(f'Sample {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()