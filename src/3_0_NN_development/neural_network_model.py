import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from skopt.space import Real, Integer

from base_model import BaseModelWithFeatureSelection

class NeuralNetworkWithFeatureSelection(BaseModelWithFeatureSelection):
    def _build_model(self, input_dim, output_dim=3, **params):
        """Build neural network architecture with specified parameters"""
        inputs = Input(shape=(input_dim,))
        
        # First layer
        x = Dense(params.get('first_layer_units', 256), activation='relu')(inputs)
        x = Dropout(params.get('dropout_rate', 0.2))(x)
        
        # Hidden layers
        for _ in range(params.get('num_hidden_layers', 3)):
            x = Dense(params.get('hidden_layer_units', 128), activation='relu')(x)
            x = Dropout(params.get('dropout_rate', 0.2))(x)
        
        outputs = Dense(output_dim, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _define_parameter_space(self):
        """Define hyperparameter search space for neural networks"""
        return [
            Integer(32, 512, name='first_layer_units'),
            Integer(16, 256, name='hidden_layer_units'),
            Integer(2, 5, name='num_hidden_layers'),
            Real(0.0, 0.5, name='dropout_rate'),
            Real(0.0001, 0.01, name='learning_rate'),
            Integer(16, 128, name='batch_size')
        ]
    
    def _evaluate_model_for_optimization(self, model, X_train, y_train, X_val, y_val, params):
        """Evaluate neural network for hyperparameter optimization"""
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=150,  # Reduced epochs for optimization
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Return the validation loss (to be minimized)
        return min(history.history['val_loss'])
    
    def _extract_best_params(self, optimization_result):
        """Extract best parameters from optimization result for neural network"""
        return {
            'first_layer_units': optimization_result.x[0],
            'hidden_layer_units': optimization_result.x[1],
            'num_hidden_layers': optimization_result.x[2],
            'dropout_rate': optimization_result.x[3],
            'learning_rate': optimization_result.x[4],
            'batch_size': optimization_result.x[5]
        }
    
    def train(self, X, y, validation_data=None, epochs=300, batch_size=32, 
              verbose=1, optimize_hyperparams=False, n_calls=50):
        """Train the neural network with selected features"""
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
                batch_size = self.best_params['batch_size']
                    
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
        """Make predictions using the trained neural network"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        if self.selected_features is None:
            raise ValueError("Features have not been selected. Call select_features() first.")
            
        # Use only selected features for prediction
        X_selected = X[self.selected_features]
        
        # Make predictions
        return self.model.predict(X_selected)
    
    def evaluate(self, X, y):
        """Evaluate the neural network on test data"""
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
        """Plot the neural network training history"""
        if self.history is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation loss values
        ax1.plot(self.history.history['loss'])
        ax1.plot(self.history.history['val_loss'])
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation mean absolute error
        ax2.plot(self.history.history['mae'])
        ax2.plot(self.history.history['val_mae'])
        ax2.set_title('Model Mean Absolute Error')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _save_model_specific(self, filepath):
        """Save neural network specific components"""
        # Save Keras model
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved neural network model"""
        # Load Keras model
        keras_model = load_model(filepath)
        
        # Load metadata
        import json
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create instance and set attributes
        instance = cls(
            feature_selection_method=metadata['feature_selection_method'],
            n_features=metadata['n_features'],
            keep_prefixes=eval(metadata['keep_prefixes']) if metadata['keep_prefixes'].startswith('[') else []
        )
        
        instance.model = keras_model
        instance.selected_features = eval(metadata['selected_features']) if metadata['selected_features'].startswith('[') else []
        instance.best_params = eval(metadata['best_params']) if metadata['best_params'] != 'None' else None
        
        return instance
