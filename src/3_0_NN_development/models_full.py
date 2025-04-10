"""
Optimized model implementation with Bayesian hyperparameter optimization.
Focused specifically on XGBoost, Neural Network, and VAE models.

Designed for direct integration with established X and y splits.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import pickle
import time
import matplotlib.pyplot as plt

# Base Model and Neural Network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Layer
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# XGBoost
import xgboost as xgb

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hyperparameter optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor


class BayesianOptimizedModel:
    """Base class for models with Bayesian hyperparameter optimization."""
    
    def __init__(self, name: str = "bayes_opt_model", **kwargs):
        """
        Initialize model with Bayesian optimization capabilities.
        
        Args:
            name: Model name
            **kwargs: Additional parameters
        """
        self.name = name
        self.config = {
            'n_iter': 100,        # Number of optimization iterations
            'cv': 10,             # Number of cross-validation folds
            'n_jobs': -1,         # Use all available cores
            'random_state': 42,   # Random seed
            'verbose': 1          # Verbosity level
        }
        
        # Update with provided kwargs
        self.config.update(kwargs)
        
        # Model and results
        self.model = None
        self.history = None
        self.feature_names = None
        self.target_names = None
        self.opt_results = None
        self.best_params = None
        
    def get_param_space(self) -> Dict:
        """
        Define parameter space for optimization.
        To be implemented by each model subclass.
        
        Returns:
            Dictionary with parameter spaces
        """
        raise NotImplementedError("Subclass must implement get_param_space()")
    
    def create_model(self, params: Dict = None) -> Any:
        """
        Create the underlying model with the given parameters.
        To be implemented by each model subclass.
        
        Args:
            params: Model parameters
            
        Returns:
            Model object
        """
        raise NotImplementedError("Subclass must implement create_model()")
    
    def optimize(self, X, y, groups=None) -> Dict:
        """
        Perform Bayesian hyperparameter optimization.
        
        Args:
            X: Features
            y: Targets
            groups: Group labels for group k-fold cross-validation
            
        Returns:
            Best parameters
        """
        # Get parameter space
        param_space = self.get_param_space()
        
        # Create base model for optimization
        base_model = self.create_model()
        
        # Set up GroupKFold if groups are provided
        if groups is not None:
            cv = GroupKFold(n_splits=min(self.config['cv'], len(np.unique(groups))))
        else:
            cv = self.config['cv']
        
        # Create BayesSearchCV
        opt = BayesSearchCV(
            base_model,
            param_space,
            n_iter=self.config['n_iter'],
            cv=cv,
            n_jobs=self.config['n_jobs'],
            random_state=self.config['random_state'],
            verbose=self.config['verbose'],
            scoring='neg_mean_squared_error'
        )
        
        # Start timing
        start_time = time.time()
        
        # Run optimization
        if groups is not None:
            opt.fit(X, y, groups=groups)
        else:
            opt.fit(X, y)
            
        # End timing
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Store results
        self.opt_results = opt
        self.best_params = opt.best_params_
        
        # Create final model with best parameters
        self.model = self.create_model(self.best_params)
        
        # Print results
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {opt.best_score_:.4f}")
        
        return self.best_params
    
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize=False, groups=None, **kwargs) -> Dict:
        """
        Train the model. If optimize is True, perform hyperparameter optimization first.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            optimize: Whether to perform hyperparameter optimization
            groups: Group labels for group k-fold cross-validation
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        # Store feature and target names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        if isinstance(y_train, pd.DataFrame):
            self.target_names = y_train.columns.tolist()
        elif isinstance(y_train, np.ndarray) and len(y_train.shape) > 1 and y_train.shape[1] > 1:
            self.target_names = [f"target_{i}" for i in range(y_train.shape[1])]
        
        # Convert to numpy arrays if needed
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        
        if optimize:
            # Perform hyperparameter optimization
            self.optimize(X_train_np, y_train_np, groups=groups)
        elif self.model is None:
            # Create model with default parameters if no optimization
            self.model = self.create_model()
        
        # Train the model
        self.model.fit(X_train_np, y_train_np)
        
        # Simple history for compatibility
        self.history = {'trained': True}
        
        return self.history
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
            
        # Convert to numpy array if needed
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        return self.model.predict(X_np)
    
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
            target_names = self.target_names or [f"target_{i}" for i in range(y_test_np.shape[1])]
            for i, target in enumerate(target_names):
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
            
        # Save optimization results
        if self.opt_results is not None:
            with open(os.path.join(path, 'opt_results.pkl'), 'wb') as f:
                pickle.dump(self.opt_results, f)
                
        # Save best parameters
        if self.best_params is not None:
            with open(os.path.join(path, 'best_params.pkl'), 'wb') as f:
                pickle.dump(self.best_params, f)
                
        # Save config
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
    
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
        
        # Load optimization results
        opt_results_path = os.path.join(path, 'opt_results.pkl')
        if os.path.exists(opt_results_path):
            with open(opt_results_path, 'rb') as f:
                self.opt_results = pickle.load(f)
                
        # Load best parameters
        best_params_path = os.path.join(path, 'best_params.pkl')
        if os.path.exists(best_params_path):
            with open(best_params_path, 'rb') as f:
                self.best_params = pickle.load(f)
                
        # Load config
        config_path = os.path.join(path, 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
    
    def plot_optimization_results(self, top_n: int = 10, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot optimization results.
        
        Args:
            top_n: Number of top results to show
            figsize: Figure size (width, height)
        """
        if self.opt_results is None:
            raise ValueError("No optimization results available")
            
        # Get results as DataFrame
        results = pd.DataFrame(self.opt_results.cv_results_)
        
        # Sort by mean test score
        results = results.sort_values('mean_test_score', ascending=False)
        
        # Take top N results
        top_results = results.head(top_n)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot scores
        plt.errorbar(
            range(len(top_results)),
            -top_results['mean_test_score'],  # Convert to positive MSE
            yerr=top_results['std_test_score'],
            fmt='o',
            capsize=5
        )
        
        plt.xlabel('Rank')
        plt.ylabel('Mean Squared Error')
        plt.title('Top Hyperparameter Configurations')
        plt.xticks(range(len(top_results)))
        plt.grid(True, alpha=0.3)
        
        # Add parameter details
        param_names = [p for p in results.columns if p.startswith('param_')]
        
        for i, (_, row) in enumerate(top_results.iterrows()):
            param_text = "\n".join([f"{p.replace('param_', '')}: {row[p]}" for p in param_names])
            plt.annotate(
                param_text,
                xy=(i, -row['mean_test_score']),
                xytext=(0, 30),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5')
            )
        
        plt.tight_layout()
        plt.show()


class XGBoostModel(BayesianOptimizedModel):
    """XGBoost model implementation with Bayesian optimization."""
    
    def __init__(self, name: str = "xgboost", **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        
    def get_param_space(self) -> Dict:
        """
        Define parameter space for XGBoost optimization.
        
        Returns:
            Dictionary with parameter spaces
        """
        param_space = {
            'n_estimators': Integer(50, 500),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 15),
            'min_child_weight': Integer(1, 10),
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'gamma': Real(0.0, 5.0),
            'reg_alpha': Real(0.0, 10.0, prior='log-uniform'),
            'reg_lambda': Real(0.0, 10.0, prior='log-uniform')
        }
        
        return param_space
    
    def create_model(self, params: Dict = None) -> Any:
        """
        Create XGBoost model with the given parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            XGBoost model
        """
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_jobs': -1,
            'random_state': self.config['random_state'],
            'verbosity': 0
        }
        
        # Update with provided parameters
        if params is not None:
            default_params.update(params)
            
        # Create multi-output model if needed
        if hasattr(self, 'target_names') and self.target_names and len(self.target_names) > 1:
            model = MultiOutputRegressor(xgb.XGBRegressor(**default_params))
        else:
            model = xgb.XGBRegressor(**default_params)
            
        return model
    
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
            self.model.estimators_[0].feature_importances_.shape[0] if hasattr(self.model, 'estimators_') 
            else self.model.feature_importances_.shape[0]
        )]
        
        # Extract importance
        if hasattr(self.model, 'estimators_'):
            # Multiple outputs (MultiOutputRegressor)
            importances = np.mean([
                est.feature_importances_ for est in self.model.estimators_
            ], axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
        else:
            # Single output
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            })
            
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
            
        return importance_df


class NeuralNetworkModel(BayesianOptimizedModel):
    """Neural Network model implementation with Bayesian optimization."""
    
    def __init__(self, name: str = "neural_network", **kwargs):
        """
        Initialize Neural Network model.
        
        Args:
            name: Model name
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        
    def get_param_space(self) -> Dict:
        """
        Define parameter space for Neural Network optimization.
        
        Returns:
            Dictionary with parameter spaces
        """
        param_space = {
            'hidden_layer_sizes': Categorical([
                (64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)
            ]),
            'activation': Categorical(['relu', 'tanh']),
            'learning_rate': Real(0.0001, 0.01, prior='log-uniform'),
            'batch_size': Categorical([16, 32, 64, 128]),
            'dropout_rate': Real(0.0, 0.5)
        }
        
        return param_space
    
    def create_model(self, params: Dict = None) -> Any:
        """
        Create Neural Network model with the given parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Neural Network model wrapper
        """
        default_params = {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'dropout_rate': 0.2,
            'epochs': 1000,
            'patience': 20
        }
        
        # Update with provided parameters
        if params is not None:
            default_params.update(params)
            
        # Create a wrapper to use with Bayesian optimization
        # This wrapper will build and train a Keras model
        class KerasRegressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model = None
                self.output_dim = None
                self.history = None
                
            def _build_model(self, input_dim, output_dim):
                model = Sequential()
                
                # Input layer
                model.add(Dense(
                    self.kwargs['hidden_layer_sizes'][0], 
                    activation=self.kwargs['activation'],
                    input_dim=input_dim
                ))
                model.add(Dropout(self.kwargs['dropout_rate']))
                
                # Hidden layers
                for units in self.kwargs['hidden_layer_sizes'][1:]:
                    model.add(Dense(units, activation=self.kwargs['activation']))
                    model.add(Dropout(self.kwargs['dropout_rate']))
                
                # Output layer
                model.add(Dense(output_dim))
                
                # Compile model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.kwargs['learning_rate']
                    ),
                    loss='mse',
                    metrics=['mae']
                )
                
                return model
                
            def fit(self, X, y, **fit_params):
                # Store output dimension
                self.output_dim = y.shape[1] if len(y.shape) > 1 else 1
                
                # Build model
                self.model = self._build_model(X.shape[1], self.output_dim)
                
                # Set up callbacks
                callbacks = []
                
                # Early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.kwargs['patience'],
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
                
                # Learning rate reduction
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.kwargs['patience'] // 2,
                    min_lr=1e-6
                )
                callbacks.append(reduce_lr)
                
                # Train model
                self.history = self.model.fit(
                    X, y,
                    epochs=self.kwargs['epochs'],
                    batch_size=self.kwargs['batch_size'],
                    validation_split=0.1,
                    callbacks=callbacks,
                    verbose=0
                ).history
                
                return self
                
            def predict(self, X):
                return self.model.predict(X, verbose=0)
                
            def score(self, X, y):
                y_pred = self.predict(X)
                return -mean_squared_error(y, y_pred)
                
            def get_params(self, deep=True):
                return self.kwargs
                
            def set_params(self, **params):
                self.kwargs.update(params)
                return self
        
        return KerasRegressor(**default_params)
    
    def save(self, path: str) -> None:
        """
        Save the Neural Network model to disk.
        
        Args:
            path: Directory path to save the model
        """
        super().save(path)
        
        # Save Keras model
        if self.model is not None and hasattr(self.model, 'model'):
            self.model.model.save(os.path.join(path, 'keras_model.h5'))
            
        # Save history
        if hasattr(self.model, 'history') and self.model.history:
            with open(os.path.join(path, 'history.pkl'), 'wb') as f:
                pickle.dump(self.model.history, f)
    
    def load(self, path: str) -> None:
        """
        Load the Neural Network model from disk.
        
        Args:
            path: Directory path where the model is saved
        """
        super().load(path)
        
        # Load Keras model
        model_path = os.path.join(path, 'keras_model.h5')
        if os.path.exists(model_path):
            # Create a wrapper for the Keras model
            class KerasWrapper:
                def __init__(self):
                    self.model = None
                    self.history = None
                
                def predict(self, X):
                    return self.model.predict(X, verbose=0)
            
            self.model = KerasWrapper()
            self.model.model = load_model(model_path)
            
            # Load history
            history_path = os.path.join(path, 'history.pkl')
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    self.model.history = pickle.load(f)


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


class VAEModel(BayesianOptimizedModel):
    """Variational Autoencoder model implementation with Bayesian optimization."""
    
    def __init__(self, name: str = "vae", **kwargs):
        """
        Initialize VAE model.
        
        Args:
            name: Model name
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.encoder = None
        self.decoder = None
        self.regression_model = None
        self.vae = None
        self.input_dim = None
        self.output_dim = None
        
    def get_param_space(self) -> Dict:
        """
        Define parameter space for VAE optimization.
        
        Returns:
            Dictionary with parameter spaces
        """
        param_space = {
            'encoder_hidden_units': Categorical([(128, 64), (256, 128), (128, 64, 32)]),
            'decoder_hidden_units': Categorical([(64, 128), (128, 256), (32, 64, 128)]),
            'latent_dim': Integer(8, 32),
            'learning_rate': Real(0.0001, 0.01, prior='log-uniform'),
            'kl_weight': Real(0.01, 1.0, prior='log-uniform'),
            'batch_size': Categorical([16, 32, 64, 128]),
            'dropout_rate': Real(0.0, 0.5)
        }
        
        return param_space
    
    def create_model(self, params: Dict = None) -> Any:
        """
        Create VAE model with the given parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            VAE model wrapper
        """
        default_params = {
            'encoder_hidden_units': (128, 64),
            'decoder_hidden_units': (64, 128),
            'latent_dim': 16,
            'learning_rate': 0.001,
            'kl_weight': 0.1,
            'batch_size': 32,
            'dropout_rate': 0.2,
            'epochs': 1000,
            'patience': 20
        }
        
        # Update with provided parameters
        if params is not None:
            default_params.update(params)
            
        # Create a wrapper to use with Bayesian optimization
        # This wrapper will build and train a VAE model
        class VAEWrapper:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.encoder = None
                self.decoder = None
                self.regression_model = None
                self.vae = None
                self.history = None
                self.input_dim = None
                self.output_dim = None
                
            def _build_encoder(self, input_dim, latent_dim):
                # Input layer
                inputs = Input(shape=(input_dim,), name='encoder_input')
                
                # Hidden layers
                x = inputs
                for i, units in enumerate(self.kwargs['encoder_hidden_units']):
                    x = Dense(units, activation='relu', name=f'encoder_dense_{i}')(x)
                    x = Dropout(self.kwargs['dropout_rate'])(x)
                
                # Latent space parameters
                z_mean = Dense(latent_dim, name='z_mean')(x)
                z_log_var = Dense(latent_dim, name='z_log_var')(x)
                
                # Sampling layer
                z = SamplingLayer(name='z')([z_mean, z_log_var])
                
                # Define encoder model
                encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
                
                return encoder
                
            def _build_decoder(self, latent_dim, output_dim):
                # Input layer (latent space)
                latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
                
                # Hidden layers
                x = latent_inputs
                for i, units in enumerate(self.kwargs['decoder_hidden_units']):
                    x = Dense(units, activation='relu', name=f'decoder_dense_{i}')(x)
                    x = Dropout(self.kwargs['dropout_rate'])(x)
                
                # Output layer
                outputs = Dense(output_dim, name='decoder_output')(x)
                
                # Define decoder model
                decoder = Model(latent_inputs, outputs, name='decoder')
                
                return decoder
                
            def _build_regression_model(self, latent_dim, output_dim):
                # Input layer (latent space)
                latent_inputs = Input(shape=(latent_dim,), name='regression_input')
                
                # Hidden layers
                x = latent_inputs
                for i, units in enumerate(reversed(self.kwargs['encoder_hidden_units'])):
                    x = Dense(units, activation='relu', name=f'regression_dense_{i}')(x)
                    x = Dropout(self.kwargs['dropout_rate'])(x)
                
                # Output layer
                outputs = Dense(output_dim, name='regression_output')(x)
                
                # Define regression model
                regression_model = Model(latent_inputs, outputs, name='regression')
                
                return regression_model
                
            def _build_vae(self, input_dim, output_dim, latent_dim):
                # Build encoder
                self.encoder = self._build_encoder(input_dim, latent_dim)
                
                # Build decoder (for reconstruction)
                self.decoder = self._build_decoder(latent_dim, input_dim)
                
                # Build regression model
                self.regression_model = self._build_regression_model(latent_dim, output_dim)
                
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
                    kl_loss *= -0.5 * self.kwargs['kl_weight']
                    
                    return K.mean(reconstruction_loss + kl_loss)
                    
                # Create VAE model
                vae = Model(inputs, [reconstructed, predictions])
                
                # Compile model
                vae.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.kwargs['learning_rate']),
                    loss=[vae_loss, 'mse'],
                    loss_weights=[1.0, 1.0],
                    metrics={'decoder': 'mse', 'regression': 'mae'}
                )
                
                return vae
                
            def fit(self, X, y, **fit_params):
                # Store dimensions
                self.input_dim = X.shape[1]
                self.output_dim = y.shape[1] if len(y.shape) > 1 else 1
                
                # Build model
                self.vae = self._build_vae(
                    self.input_dim, 
                    self.output_dim, 
                    self.kwargs['latent_dim']
                )
                
                # Set up callbacks
                callbacks = []
                
                # Early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.kwargs['patience'],
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
                
                # Learning rate reduction
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.kwargs['patience'] // 2,
                    min_lr=1e-6
                )
                callbacks.append(reduce_lr)
                
                # Train model
                self.history = self.vae.fit(
                    X,
                    [X, y],
                    epochs=self.kwargs['epochs'],
                    batch_size=self.kwargs['batch_size'],
                    validation_split=0.1,
                    callbacks=callbacks,
                    verbose=0
                ).history
                
                return self
                
            def predict(self, X):
                _, predictions = self.vae.predict(X, verbose=0)
                return predictions
                
            def encode(self, X):
                z_mean, _, _ = self.encoder.predict(X, verbose=0)
                return z_mean
                
            def score(self, X, y):
                y_pred = self.predict(X)
                return -mean_squared_error(y, y_pred)
                
            def get_params(self, deep=True):
                return self.kwargs
                
            def set_params(self, **params):
                self.kwargs.update(params)
                return self
        
        return VAEWrapper(**default_params)
    
    def save(self, path: str) -> None:
        """
        Save the VAE model to disk.
        
        Args:
            path: Directory path to save the model
        """
        super().save(path)
        
        # Save model components if available
        if self.model is not None:
            if hasattr(self.model, 'encoder') and self.model.encoder is not None:
                self.model.encoder.save(os.path.join(path, 'encoder.h5'))
                
            if hasattr(self.model, 'decoder') and self.model.decoder is not None:
                self.model.decoder.save(os.path.join(path, 'decoder.h5'))
                
            if hasattr(self.model, 'regression_model') and self.model.regression_model is not None:
                self.model.regression_model.save(os.path.join(path, 'regression.h5'))
                
            if hasattr(self.model, 'vae') and self.model.vae is not None:
                self.model.vae.save(os.path.join(path, 'vae.h5'))
                
            # Save VAE dimensions and history
            if hasattr(self.model, 'input_dim') and hasattr(self.model, 'output_dim'):
                dimensions = {
                    'input_dim': self.model.input_dim,
                    'output_dim': self.model.output_dim,
                    'latent_dim': self.model.kwargs['latent_dim']
                }
                with open(os.path.join(path, 'dimensions.pkl'), 'wb') as f:
                    pickle.dump(dimensions, f)
            
            if hasattr(self.model, 'history') and self.model.history:
                with open(os.path.join(path, 'history.pkl'), 'wb') as f:
                    pickle.dump(self.model.history, f)
    
    def load(self, path: str) -> None:
        """
        Load the VAE model from disk.
        
        Args:
            path: Directory path where the model is saved
        """
        super().load(path)
        
        # Load dimensions
        dimensions_path = os.path.join(path, 'dimensions.pkl')
        if os.path.exists(dimensions_path):
            with open(dimensions_path, 'rb') as f:
                dimensions = pickle.load(f)
        
        # Custom loss function for loading
        def vae_loss(x, x_decoded_mean):
            return K.mean(mse(x, x_decoded_mean))
        
        # Create wrapper for the loaded models
        class VAEWrapper:
            def __init__(self):
                self.encoder = None
                self.decoder = None
                self.regression_model = None
                self.vae = None
                self.history = None
                self.input_dim = None
                self.output_dim = None
                self.kwargs = {'latent_dim': 16}  # Default
            
            def predict(self, X):
                _, predictions = self.vae.predict(X, verbose=0)
                return predictions
                
            def encode(self, X):
                z_mean, _, _ = self.encoder.predict(X, verbose=0)
                return z_mean
        
        self.model = VAEWrapper()
        
        # Load dimensions
        if os.path.exists(dimensions_path):
            self.model.input_dim = dimensions['input_dim']
            self.model.output_dim = dimensions['output_dim']
            self.model.kwargs['latent_dim'] = dimensions['latent_dim']
        
        # Load model components
        encoder_path = os.path.join(path, 'encoder.h5')
        if os.path.exists(encoder_path):
            self.model.encoder = load_model(encoder_path, custom_objects={'SamplingLayer': SamplingLayer})
            
        decoder_path = os.path.join(path, 'decoder.h5')
        if os.path.exists(decoder_path):
            self.model.decoder = load_model(decoder_path)
            
        regression_path = os.path.join(path, 'regression.h5')
        if os.path.exists(regression_path):
            self.model.regression_model = load_model(regression_path)
            
        vae_path = os.path.join(path, 'vae.h5')
        if os.path.exists(vae_path):
            self.model.vae = load_model(vae_path, custom_objects={
                'SamplingLayer': SamplingLayer,
                'vae_loss': vae_loss
            })
            
        # Load history
        history_path = os.path.join(path, 'history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.model.history = pickle.load(f)


class ModelComparer:
    """Utility for comparing multiple models."""
    
    def __init__(self, models: Dict[str, BayesianOptimizedModel]):
        """
        Initialize with a dictionary of models.
        
        Args:
            models: Dictionary of model instances
        """
        self.models = models
        self.results = None
        
    def compare(self, X_test, y_test) -> pd.DataFrame:
        """
        Compare model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with performance metrics for each model
        """
        # Convert to numpy arrays if needed
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test_np = y_test.values if isinstance(y_test, pd.DataFrame) else y_test
        
        results = []
        
        for name, model in self.models.items():
            # Skip models that haven't been trained
            if model.model is None:
                continue
                
            # Make predictions
            y_pred = model.predict(X_test_np)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_np, y_pred)
            mse = mean_squared_error(y_test_np, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_np, y_pred)
            
            # Calculate metrics for each target if multi-output
            target_metrics = {}
            if y_test_np.ndim > 1 and y_test_np.shape[1] > 1:
                # Get target names (from first model with target_names attribute)
                target_names = None
                for m in self.models.values():
                    if hasattr(m, 'target_names') and m.target_names:
                        target_names = m.target_names
                        break
                        
                if target_names is None:
                    target_names = [f"Target_{i+1}" for i in range(y_test_np.shape[1])]
                    
                for i, target in enumerate(target_names):
                    target_metrics[f'mae_{target}'] = mean_absolute_error(y_test_np[:, i], y_pred[:, i])
                    target_metrics[f'rmse_{target}'] = np.sqrt(mean_squared_error(y_test_np[:, i], y_pred[:, i]))
                    target_metrics[f'r2_{target}'] = r2_score(y_test_np[:, i], y_pred[:, i])
            
            # Add to results
            results.append({
                'model': name,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                **target_metrics
            })
            
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by RMSE
        results_df = results_df.sort_values('rmse')
        
        self.results = results_df
        return results_df
    
    def plot_comparison(self, metric: str = 'rmse', figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot comparison of models based on a specific metric.
        
        Args:
            metric: Metric to compare ('mae', 'mse', 'rmse', 'r2')
            figsize: Figure size (width, height)
        """
        if self.results is None:
            raise ValueError("No comparison results available. Run compare() first.")
            
        # Check if the metric exists in the results
        if metric not in self.results.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Sort by the selected metric (ascending for error metrics, descending for R²)
        ascending = metric != 'r2'
        sorted_results = self.results.sort_values(metric, ascending=ascending)
        
        # Plot bar chart
        bars = plt.bar(sorted_results['model'], sorted_results[metric])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + (0.01 if metric == 'r2' else 0.05 * height),
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0
            )
            
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison by {metric.upper()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.show()


# Utility function to directly use with existing data splits
def train_models_with_existing_splits(X_train, y_train, X_test, y_test, optimize=True):
    """
    Train XGBoost, NN, and VAE models with existing data splits.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        optimize: Whether to use Bayesian optimization (default: True)
        
    Returns:
        Dictionary of trained models and comparison results
    """
    # Create models
    models = {
        'xgboost': XGBoostModel(),
        'nn': NeuralNetworkModel(),
        'vae': VAEModel()
    }
    
    # Generate groups if possible (for group cross-validation)
    groups = None
    if isinstance(X_train, pd.DataFrame):
        if 'system' in X_train.columns:
            groups = X_train['system'].values
        elif all(col in X_train.columns for col in ['solvent_1', 'solvent_2', 'compound_id']):
            groups = (X_train['solvent_1'].astype(str) + '-' + 
                     X_train['solvent_2'].astype(str) + '-' + 
                     X_train['compound_id'].astype(str)).values
    
    # Train models
    for name, model in models.items():
        print(f"\nTraining {name.upper()} model...")
        start_time = time.time()
        model.train(X_train, y_train, optimize=optimize, groups=groups)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Evaluate on test set
        metrics = model.evaluate(X_test, y_test)
        print(f"Test metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Compare models
    comparer = ModelComparer(models)
    results = comparer.compare(X_test, y_test)
    
    print("\nModel comparison results:")
    print(results)
    
    return {
        'models': models,
        'comparer': comparer,
        'results': results
    }


# Example usage
if __name__ == "__main__":
    # Example with random data for testing
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 3)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20, 3)
    
    # Train models
    result = train_models_with_existing_splits(X_train, y_train, X_test, y_test, optimize=False)
    
    # Get best model
    best_model_name = result['results'].iloc[0]['model']
    print(f"\nBest model: {best_model_name}")