from data_module import DataProcessor
from sklearn.model_selection import train_test_split
from feature_module import FeatureProcessor
from neural_network_model import NeuralNetworkWithFeatureSelection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    def train_model(self, model_class=NeuralNetworkWithFeatureSelection, feature_selection_method='random_forest', n_features=10, 
                   keep_prefixes=['solvent_1_pure','solvent_2_pure','system','solubility_','temperature'],
                   epochs=1000, batch_size=32, verbose=1, optimize_hyperparams=False, n_calls=20, **kwargs):
        """
        Train a model with feature selection.
        
        Parameters:
        -----------
        model_class : class
            Model class to use (defaults to NeuralNetworkWithFeatureSelection)
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
        **kwargs : dict
            Additional keyword arguments to pass to the model constructor
        
        Returns:
        --------
        model : object
            Trained model
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
        self.model = model_class(
            feature_selection_method=feature_selection_method, 
            n_features=n_features,
            keep_prefixes=keep_prefixes,
            **kwargs)

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
    
    def predict_model(self, x):
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
    
    system.model.save_model('trained_model.keras')