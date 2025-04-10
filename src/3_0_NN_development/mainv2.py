from data_module import DataProcessor
from sklearn.model_selection import train_test_split
from feature_module import FeatureProcessor

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer

import matplotlib.pyplot as plt

class NeuralNetworkWithFeatureSelection:
    def __init__(self, feature_selection_method='correlation', n_features=50, keep_prefixes=None):
        """
        Neural Network model with feature selection capabilities.
        
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
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        if self.selected_features is None:
            raise ValueError("Features have not been selected. Call select_features() first.")
            
        # Use only selected features for prediction
        X_selected = X[self.selected_features]
        
        # Make predictions
        return self.model.predict(X_selected)
    
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
        """
        Plot the training history
        
        Returns:
        --------
        fig : matplotlib Figure
            Figure with training history plots
        """
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
        # In a more advanced implementation, we could calculate actual importance scores
        importances = pd.DataFrame({
            'Feature': self.selected_features,
            'Selected': True
        })
        
        return importances
    
    def save_model(self, filepath):
        """
        Save the model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        # Save Keras model
        self.model.save(filepath)
        
        # Also save feature selection info and other metadata
        metadata = {
            'selected_features': self.selected_features,
            'feature_selection_method': self.feature_selection_method,
            'n_features': self.n_features,
            'keep_prefixes': self.keep_prefixes,
            'best_params': self.best_params
        }
        
        # Save metadata as JSON
        import json
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump({k: str(v) if isinstance(v, list) else v for k, v in metadata.items()}, f)
        
        print(f"Model and metadata saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        model : NeuralNetworkWithFeatureSelection
            Loaded model
        """
        # Load Keras model
        from tensorflow.keras.models import load_model
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
        space = [
            Integer(32, 512, name='first_layer_units'),
            Integer(16, 256, name='hidden_layer_units'),
            Integer(2, 5, name='num_hidden_layers'),
            Real(0.0, 0.5, name='dropout_rate'),
            Real(0.0001, 0.01, name='learning_rate'),
            Integer(16, 128, name='batch_size')
        ]
        
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
            model = self._build_model_with_params(
                input_dim=X.shape[1], 
                output_dim=y.shape[1],
                **params
            )
            
            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)
            ]
            
            # Train the model
            history = model.fit(
                X, y,
                epochs=150,  # Reduced epochs for optimization
                batch_size=params['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            # Return the validation loss (to be minimized)
            return min(history.history['val_loss'])
        
        # Run the optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=True
        )
        
        # Store the best parameters
        self.best_params = {
            'first_layer_units': result.x[0],
            'hidden_layer_units': result.x[1],
            'num_hidden_layers': result.x[2],
            'dropout_rate': result.x[3],
            'learning_rate': result.x[4],
            'batch_size': result.x[5]
        }
        
        print(f"Best hyperparameters found: {self.best_params}")
        return self.best_params
    
    def _build_model_with_params(self, input_dim, output_dim=3, **params):
        """Build neural network architecture with specified parameters"""
        inputs = Input(shape=(input_dim,))
        
        # First layer
        x = Dense(params['first_layer_units'], activation='relu')(inputs)
        x = Dropout(params['dropout_rate'])(x)
        
        # Hidden layers
        for _ in range(params['num_hidden_layers']):
            x = Dense(params['hidden_layer_units'], activation='relu')(x)
            x = Dropout(params['dropout_rate'])(x)
        
        outputs = Dense(output_dim, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X, y, validation_data=None, epochs=300, batch_size=32, 
              verbose=1, optimize_hyperparams=False, n_calls=50):
        """Train the model with selected features"""
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
            self.model = self._build_model_with_params(
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

class SystemDesign:
    def __init__(self, system_columns=['solvent_2', 'solvent_1','temperature','compound_id'], 
                 raw_data_path='curve_fit_results_x_is_7.csv',
                 extra_fitted_points=0,
                 target_columns=['J0','J1','J2']
                 ):
        """
        Initialize the SystemDesign class with system columns.
        
        Parameters:
        -----------
        system_columns : list
            List of columns that define the system.
            Options of 'solvent_2', solvent_1','temperature','compound_id'
        raw_data_path : str
            Path to the raw data file.
        extra_fitted_points : int
            Number of extra fitted points to consider.
        target_columns : list
            List of target columns to be used for splitting.
        """
        self.system_columns = system_columns
        self.dataprocess, self.dataloader = DataProcessor.CreateDataProcessor(
            raw_data_path=raw_data_path,
            system_columns=self.system_columns,
            extra_points=extra_fitted_points
        )
                    
        self.feature_processor = FeatureProcessor(categorical_column='system')
        self.target_columns = target_columns
        self.model = None
        
        self.fit_feature_processor()
        
    def get_data_split_df(self):
        """
        Get the data split DataFrame.
        
        Parameters:
        -----------
        target_columns : list
            List of target columns to be used for splitting.
        
        Returns:
        --------
        x : DataFrame
            Feature DataFrame
        y : DataFrame
            Target DataFrame.
        """
        x, y = self.dataprocess.get_data_split_df(target_columns=self.target_columns)
        return x, y
    
    def get_train_test_split(self):
        x, y = self.get_data_split_df()
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )
        return x_train, x_test, y_train, y_test
    
    def fit_feature_processor(self):
        x, y = self.get_data_split_df()
        self.feature_processor.fit(x, y)
        
    def transform_inputs(self, x):
        return self.feature_processor.transform_inputs(x)
    
    def transform_outputs(self, y):
        return self.feature_processor.transform_outputs(y)
    
    def train_model(self, feature_selection_method='random_forest', n_features=10, 
                   keep_prefixes=['solvent_1_pure','solvent_2_pure','system','solubility_','temperature'],
                   epochs=1000, batch_size=32, verbose=1, optimize_hyperparams=False, n_calls=20):
        """
        Train a neural network model with feature selection.
        
        Parameters:
        -----------
        feature_selection_method : str
            Method for feature selection: 'correlation', 'f_regression', 'rfe', 'random_forest'
        n_features : int
            Number of features to select
        keep_prefixes : list
            List of column prefixes to always keep regardless of feature selection
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level for training
        optimize_hyperparams : bool
            Whether to perform hyperparameter optimization
        n_calls : int
            Number of iterations for optimization
        
        Returns:
        --------
        model : NeuralNetworkWithFeatureSelection
            Trained neural network model
        """
        # Get train-test split
        x_train, x_test, y_train, y_test = self.get_train_test_split()
        
        # Transform training data
        X_train_processed = self.transform_inputs(x_train)
        y_train_processed = self.transform_outputs(y_train)
        
        # Transform testing data
        X_test_processed = self.transform_inputs(x_test)
        y_test_processed = self.transform_outputs(y_test)

        # Create model with feature selection
        self.model = NeuralNetworkWithFeatureSelection(
            feature_selection_method=feature_selection_method, 
            n_features=n_features,
            keep_prefixes=keep_prefixes)

        # Train the model
        self.model.train(
            X_train_processed, 
            y_train_processed, 
            validation_data=(X_test_processed, y_test_processed), 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=verbose,
            optimize_hyperparams=optimize_hyperparams,
            n_calls=n_calls
        )
        
        return self.model
    
    def predict_model(self,x):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        x : DataFrame
            Feature DataFrame for prediction.
        
        Returns:
        --------
        y_pred : ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
            
        # Transform input data
        X_processed = self.transform_inputs(x)
        
        # Make predictions
        y_pred = self.model.predict(X_processed)
        
        # Convert scaled predictions back to original scale
        y_pred = self.feature_processor.output_scalar.inverse_transform(y_pred)
        y_pred = pd.DataFrame(
            y_pred,
            columns=self.target_columns,
            index=x.index
        )
        
        return y_pred
        
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        
        Returns:
        --------
        results : dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
            
        # Get test data
        _, x_test, _, y_test = self.get_train_test_split()
        
        # Transform test data
        X_test_processed = self.transform_inputs(x_test)
        y_test_processed = self.transform_outputs(y_test)
        
        # Evaluate and plot
        results = self.model.evaluate(X_test_processed, y_test_processed)
        self.model.plot_training_history()
        
        return results
    
    def get_predictions_and_metrics(self):
        """
        Get predictions and calculate metrics in original scale.
        
        Returns:
        --------
        predictions : tuple
            Tuple containing (y_pred_original, y_test_original, mae_original)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
            
        # Get test data
        _, x_test, _, y_test = self.get_train_test_split()
        
        # Transform test data
        X_test_processed = self.transform_inputs(x_test)
        y_test_processed = self.transform_outputs(y_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_processed)
        
        # Convert scaled predictions back to original scale
        y_pred_original = self.feature_processor.output_scalar.inverse_transform(y_pred)
        y_test_original = self.feature_processor.output_scalar.inverse_transform(y_test_processed)
        
        # Calculate metrics
        mae_original = np.mean(np.abs(y_pred_original - y_test_original), axis=0)
        print(f"MAE in original scale - J0: {mae_original[0]:.2f}, J1: {mae_original[1]:.2f}, J2: {mae_original[2]:.2f}")
        
        return y_pred_original, y_test_original, mae_original


if __name__ == "__main__":
    # Create and setup the system
    system = SystemDesign(
        system_columns=['solvent_1','solvent_2','temperature'],
        raw_data_path='curve_fit_results_x_is_7.csv',
        extra_fitted_points=1,
        target_columns=['J0','J1','J2']
    )
    
    # Train the model
    system.train_model(
        feature_selection_method='random_forest',
        n_features=10,
        keep_prefixes=['solvent_1_pure','solvent_2_pure','system','solubility_','temperature'],
        epochs=1000, 
        batch_size=32, 
        verbose=1,
        optimize_hyperparams=True,
        n_calls=11
    )
    
    # Evaluate the model
    system.evaluate_model()
    
    # Get predictions and metrics
    predictions, actuals, mae = system.get_predictions_and_metrics()