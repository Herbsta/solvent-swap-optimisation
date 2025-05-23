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
    
    # The rest of your class remains unchanged
    def build_model(self, input_dim, output_dim=3):
        """Build neural network architecture"""
        inputs = Input(shape=(input_dim,))
        
        x = Dense(256, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(output_dim, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X, y, validation_data=None, epochs=300, batch_size=32, verbose=1):
        """Train the model with selected features"""
        # Select features if not already done
        if self.selected_features is None:
            X_selected = self.select_features(X, y)
        else:
            X_selected = X[self.selected_features]
                    
        # Build the model if not already built
        if self.model is None:
            self.model = self.build_model(input_dim=X_selected.shape[1], output_dim=y.shape[1])
                
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
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        X_selected = X[self.selected_features]
        return self.model.predict(X_selected)
    
    def evaluate(self, X, y):
        """Evaluate the model performance"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        X_selected = X[self.selected_features]
        y_pred = self.model.predict(X_selected)
        
        mae_per_target = mean_absolute_error(y, y_pred, multioutput='raw_values')
        r2_per_target = r2_score(y, y_pred, multioutput='raw_values')
        
        print("\nModel Evaluation:")
        print("-----------------")
        for i, col in enumerate(y.columns):
            print(f"{col}: MAE = {mae_per_target[i]:.4f}, R² = {r2_per_target[i]:.4f}")
        
        print(f"\nAverage MAE: {np.mean(mae_per_target):.4f}")
        print(f"Average R²: {np.mean(r2_per_target):.4f}")
        
        return {
            'mae': mae_per_target,
            'r2': r2_per_target,
            'mae_avg': np.mean(mae_per_target),
            'r2_avg': np.mean(r2_per_target)
        }
    
    def plot_feature_importance(self, feature_names=None, top_n=20):
        """Plot feature importance"""
        if self.selected_features is None:
            raise ValueError("Features have not been selected yet")
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(min(top_n, len(self.selected_features))), 
                [1] * min(top_n, len(self.selected_features)))
        plt.xticks(range(min(top_n, len(self.selected_features))), 
                  self.selected_features[:top_n], rotation=90)
        plt.title(f'Top {min(top_n, len(self.selected_features))} Selected Features')
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("Model has not been trained yet")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation loss values
        axes[0].plot(self.history.history['loss'])
        axes[0].plot(self.history.history['val_loss'])
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation mean absolute error
        axes[1].plot(self.history.history['mae'])
        axes[1].plot(self.history.history['val_mae'])
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()

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
        raw_data_pathh : str
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
            Feature DataFrame.
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
                   epochs=1000, batch_size=32, verbose=1):
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
            verbose=verbose
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
        verbose=1
    )
    
    # Evaluate the model
    system.evaluate_model()
    
    # Get predictions and metrics
    predictions, actuals, mae = system.get_predictions_and_metrics()