import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from skopt.space import Real, Integer
import json

from base_model import BaseModelWithFeatureSelection

class Sampling(Layer):
    """
    Custom layer for the reparameterization trick in VAE.
    Takes the mean and log variance as inputs and returns a sample from the latent distribution.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VariationalAutoencoderWithFeatureSelection(BaseModelWithFeatureSelection):
    """
    Variational Autoencoder model with feature selection capabilities.
    Can be used as a drop-in replacement for NeuralNetworkWithFeatureSelection.
    """
    def __init__(self, feature_selection_method='random_forest', n_features=50, keep_prefixes=None, 
                 latent_dim=10, kl_weight=0.001):
        """
        Initialize the VAE model with feature selection capabilities.
        
        Parameters:
        -----------
        feature_selection_method : str
            Method for feature selection: 'correlation', 'f_regression', 'rfe', 'random_forest'
        n_features : int
            Number of features to select
        keep_prefixes : list
            List of column prefixes to always keep regardless of feature selection
        latent_dim : int
            Dimension of the latent space
        kl_weight : float
            Weight for the KL divergence loss component
        """
        super().__init__(feature_selection_method, n_features, keep_prefixes)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.encoder = None
        self.decoder = None
        
    def _build_model(self, input_dim, output_dim=3, **params):
        """
        Build VAE architecture with specified parameters
        
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
        model : Model
            Compiled VAE model
        """
        # Extract parameters with defaults
        latent_dim = params.get('latent_dim', self.latent_dim)
        encoder_layers = params.get('encoder_layers', [256, 128])
        decoder_layers = params.get('decoder_layers', [128, 256])
        dropout_rate = params.get('dropout_rate', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        
        # Build encoder
        encoder_inputs = Input(shape=(input_dim,), name='encoder_input')
        x = encoder_inputs
        
        # Add encoder layers
        for i, units in enumerate(encoder_layers):
            x = Dense(units, activation='relu', name=f'encoder_dense_{i}')(x)
            x = Dropout(dropout_rate, name=f'encoder_dropout_{i}')(x)
        
        # Mean and variance for the latent distribution
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        # Sample from the distribution
        z = Sampling()([z_mean, z_log_var])
        
        # Instantiate encoder model
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Build decoder
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        x = latent_inputs
        
        # Add decoder layers
        for i, units in enumerate(decoder_layers):
            x = Dense(units, activation='relu', name=f'decoder_dense_{i}')(x)
            x = Dropout(dropout_rate, name=f'decoder_dropout_{i}')(x)
        
        # Output layer
        outputs = Dense(output_dim, activation='linear', name='decoder_output')(x)
        
        # Instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        
        # Instantiate VAE model
        vae_outputs = self.decoder(self.encoder(encoder_inputs)[2])
        vae = Model(encoder_inputs, vae_outputs, name='vae')
        
        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        vae.add_loss(self.kl_weight * kl_loss)
        
        # Compile model
        vae.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return vae
    
    def _define_parameter_space(self):
        """Define hyperparameter search space for VAE"""
        return [
            Integer(4, 32, name='latent_dim'),
            Integer(1, 4, name='num_encoder_layers'),
            Integer(32, 256, name='encoder_layer_size'),
            Real(0.0, 0.5, name='dropout_rate'),
            Real(0.0001, 0.01, name='learning_rate'),
            Real(0.0001, 0.01, name='kl_weight'),
            Integer(16, 128, name='batch_size')
        ]
    
    def _evaluate_model_for_optimization(self, model, X_train, y_train, X_val, y_val, params):
        """Evaluate VAE for hyperparameter optimization"""
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Reduced epochs for optimization
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Return the validation loss (to be minimized)
        return min(history.history['val_loss'])
    
    def _extract_best_params(self, optimization_result):
        """Extract best parameters from optimization result for VAE"""
        # Convert optimization result to hyperparameter dictionary
        latent_dim = optimization_result.x[0]
        num_encoder_layers = optimization_result.x[1]
        encoder_layer_size = optimization_result.x[2]
        
        # Create encoder and decoder layer configurations
        encoder_layers = [encoder_layer_size for _ in range(num_encoder_layers)]
        decoder_layers = encoder_layers[::-1]  # Mirror the encoder architecture
        
        return {
            'latent_dim': latent_dim,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'dropout_rate': optimization_result.x[3],
            'learning_rate': optimization_result.x[4],
            'kl_weight': optimization_result.x[5],
            'batch_size': optimization_result.x[6]
        }
    
    def train(self, X, y, validation_data=None, epochs=300, batch_size=32, 
              verbose=1, optimize_hyperparams=False, n_calls=50):
        """Train the VAE with selected features"""
        # Select features if not already done
        if self.selected_features is None:
            X_selected = self.select_features(X, y)
        else:
            X_selected = X[self.selected_features]
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            val_data = None
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val_selected = X_val[self.selected_features]
                val_data = (X_val_selected, y_val)
                
            self.optimize_hyperparameters(X_selected, y, val_data, n_calls=n_calls)
            if self.best_params is not None:
                batch_size = self.best_params.get('batch_size', batch_size)
                self.latent_dim = self.best_params.get('latent_dim', self.latent_dim)
                self.kl_weight = self.best_params.get('kl_weight', self.kl_weight)
                    
        # Build the model if not already built
        if self.model is None:
            self.model = self._build_model(
                input_dim=X_selected.shape[1], 
                output_dim=y.shape[1],
                **self.best_params if self.best_params else {}
            )
                
        # Prepare validation data if provided
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_selected = X_val[self.selected_features]
            val_data = (X_val_selected, y_val)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, min_lr=1e-6)
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_selected, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2 if val_data is None else 0.0,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions using the trained VAE"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        if self.selected_features is None:
            raise ValueError("Features have not been selected. Call select_features() first.")
            
        # Use only selected features for prediction
        X_selected = X[self.selected_features]
        
        # Make predictions
        return self.model.predict(X_selected)
    
    def encode(self, X):
        """
        Encode data to the latent space
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        z_mean : ndarray
            Mean values in latent space
        z_log_var : ndarray
            Log variance values in latent space
        z : ndarray
            Sampled points in latent space
        """
        if self.encoder is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        # Use only selected features
        X_selected = X[self.selected_features]
        
        # Encode data
        return self.encoder.predict(X_selected)
    
    def decode(self, z):
        """
        Decode latent space points to output space
        
        Parameters:
        -----------
        z : ndarray
            Points in latent space
            
        Returns:
        --------
        outputs : ndarray
            Decoded outputs
        """
        if self.decoder is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        # Decode latent space points
        return self.decoder.predict(z)
    
    def evaluate(self, X, y):
        """Evaluate the VAE on test data"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        if self.selected_features is None:
            raise ValueError("Features have not been selected. Call select_features() first.")
            
        # Use only selected features for evaluation
        X_selected = X[self.selected_features]
        
        # Evaluate model
        loss, mae = self.model.evaluate(X_selected, y, verbose=0)
        
        # Make predictions for R² calculation
        y_pred = self.model.predict(X_selected)
        r2 = r2_score(y, y_pred)
        
        # Print metrics
        print(f"Test loss (MSE): {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R²: {r2:.4f}")
        
        results = {
            'loss': loss,
            'mae': mae,
            'r2': r2
        }
        
        return results
    
    def plot_training_history(self):
        """Plot the VAE training history"""
        if self.history is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation loss values
        ax1.plot(self.history.history['loss'])
        ax1.plot(self.history.history['val_loss'])
        ax1.set_title('VAE Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation mean absolute error
        ax2.plot(self.history.history['mae'])
        ax2.plot(self.history.history['val_mae'])
        ax2.set_title('VAE Mean Absolute Error')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_latent_space(self, X, y=None, n_samples=1000, dim1=0, dim2=1):
        """
        Plot samples in the latent space
        
        Parameters:
        -----------
        X : DataFrame
            Input data
        y : DataFrame, optional
            Target values for coloring the points
        n_samples : int
            Number of samples to plot
        dim1 : int
            First dimension to plot
        dim2 : int
            Second dimension to plot
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        if self.encoder is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Select a subset of samples if needed
        if X.shape[0] > n_samples:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_subset = X.iloc[indices]
            if y is not None:
                y_subset = y.iloc[indices]
        else:
            X_subset = X
            y_subset = y
        
        # Encode the data
        X_selected = X_subset[self.selected_features]
        z_mean, _, _ = self.encoder.predict(X_selected)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot points
        scatter = ax.scatter(
            z_mean[:, dim1], 
            z_mean[:, dim2],
            c=y_subset.iloc[:, 0] if y is not None else None,
            cmap='viridis',
            alpha=0.8
        )
        
        # Add colorbar if y is provided
        if y is not None:
            plt.colorbar(scatter, ax=ax, label=y_subset.columns[0])
        
        ax.set_xlabel(f'Latent Dimension {dim1}')
        ax.set_ylabel(f'Latent Dimension {dim2}')
        ax.set_title('Visualization of Latent Space')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _save_model_specific(self, filepath):
        """Save VAE specific components"""
        # Save full model
        self.model.save(filepath)
        
        # Save encoder and decoder separately for convenience
        if self.encoder is not None:
            self.encoder.save(f"{filepath}_encoder")
        if self.decoder is not None:
            self.decoder.save(f"{filepath}_decoder")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved VAE model"""
        # Load custom layer
        custom_objects = {"Sampling": Sampling}
        
        # Load full model, encoder and decoder
        model = load_model(filepath, custom_objects=custom_objects)
        encoder = None
        decoder = None
        
        try:
            encoder = load_model(f"{filepath}_encoder", custom_objects=custom_objects)
            decoder = load_model(f"{filepath}_decoder", custom_objects=custom_objects)
        except:
            print("Warning: Could not load separate encoder/decoder models.")
        
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create instance and set attributes
        instance = cls(
            feature_selection_method=metadata['feature_selection_method'],
            n_features=metadata['n_features'],
            keep_prefixes=eval(metadata['keep_prefixes']) if metadata['keep_prefixes'].startswith('[') else []
        )
        
        instance.model = model
        instance.encoder = encoder
        instance.decoder = decoder
        instance.selected_features = eval(metadata['selected_features']) if metadata['selected_features'].startswith('[') else []
        instance.best_params = eval(metadata['best_params']) if metadata['best_params'] != 'None' else None
        
        # Set VAE-specific params from best_params if available
        if instance.best_params:
            instance.latent_dim = instance.best_params.get('latent_dim', 10)
            instance.kl_weight = instance.best_params.get('kl_weight', 0.001)
        
        return instance
