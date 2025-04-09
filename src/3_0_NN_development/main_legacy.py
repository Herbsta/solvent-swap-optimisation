"""
Main script for solvent swap optimization modeling.

This script demonstrates how to use the modular framework to build,
train, and evaluate different models for solvent swap optimization.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Import modules
from data_module import DataLoader, DataProcessor
from feature_module import FeatureEncoder, FeatureImportance
from model_module import NeuralNetworkModel, ForestModel, GradientBoostingModel
from visualization_module import SolventSwapVisualizer


class SolventSwapModeling:
    """Main class for solvent swap modeling experiments."""
    
    def __init__(self, 
               data_path: str = None,
               database_path: str = None,
               output_path: str = 'models'):
        """
        Initialize the modeling framework.
        
        Args:
            data_path: Path to data files
            database_path: Path to compound/solvent database
            output_path: Path to save models and results
        """
        self.data_path = data_path
        self.database_path = database_path
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_path, database_path)
        self.data_processor = DataProcessor()
        self.feature_encoder = FeatureEncoder()
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.compound_descriptors = None
        self.solvent_descriptors = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models
        self.models = {}
        
    def load_data(self) -> None:
        """Load and prepare data for modeling."""
        # Load raw data
        print("Loading data...")
        self.raw_data = self.data_loader.load_data_from_csvs()
        print(f"Loaded {len(self.raw_data)} data points")
        
        # Load compound descriptors if database path is provided
        if self.database_path:
            print("Loading compound descriptors...")
            self.compound_descriptors = self.data_loader.load_compound_descriptors()
            print(f"Loaded descriptors for {len(self.compound_descriptors)} compounds")
            
            # Extract unique solvent IDs from data
            solvent_ids = pd.concat([
                self.raw_data['solvent_1'].drop_duplicates(),
                self.raw_data['solvent_2'].drop_duplicates()
            ]).drop_duplicates().tolist()
            
            print("Loading solvent descriptors...")
            self.solvent_descriptors = self.data_loader.load_solvent_descriptors(solvent_ids)
            print(f"Loaded descriptors for {len(self.solvent_descriptors)} solvents")
            
    def prepare_fixed_model_data(self, 
                               target_columns: List[str],
                               system_columns: List[str],
                               test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for the fixed model (baseline).
        
        Args:
            target_columns: List of target column names
            system_columns: List of system parameter column names
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test DataFrames
        """
        print("\nPreparing data for fixed model...")
        
        # Set raw data for data processor
        self.data_processor.set_raw_data(self.raw_data)
        
        # Process data
        processed_data = self.data_processor.process_data(
            target_columns=target_columns,
            system_columns=system_columns
        )
        
        # Create system ID feature
        for col in system_columns:
            self.feature_encoder.fit_one_hot_encoder(processed_data, col)
            encoded = self.feature_encoder.transform_one_hot(processed_data, col)
            processed_data = pd.concat([processed_data, encoded], axis=1)
            
        # Get feature columns
        feature_cols = [col for col in processed_data.columns 
                       if col.startswith(tuple(col + '_' for col in system_columns))]
        
        # Split data
        X_train, X_test, y_train, y_test = self.data_processor.split_data(
            target_columns=target_columns,
            feature_columns=feature_cols,
            test_size=test_size
        )
        
        print(f"Created {len(feature_cols)} features for fixed model")
        
        # Store for later use
        self.X_train_fixed = X_train
        self.X_test_fixed = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def prepare_compound_model_data(self, 
                                  target_columns: List[str],
                                  system_columns: List[str],
                                  compound_id_col: str = 'compound_id',
                                  test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for the compound-encoded model.
        
        Args:
            target_columns: List of target column names
            system_columns: List of system parameter column names
            compound_id_col: Column name for compound ID
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test DataFrames
        """
        print("\nPreparing data for compound model...")
        
        if self.compound_descriptors is None:
            raise ValueError("Compound descriptors not loaded. Please load data first.")
            
        # Set raw data for data processor
        self.data_processor.set_raw_data(self.raw_data)
        
        # Process data
        processed_data = self.data_processor.process_data(
            target_columns=target_columns,
            system_columns=system_columns,
            compound_id_col=compound_id_col
        )
        
        # Merge compound descriptors
        processed_data = self.data_processor.merge_compound_descriptors(
            compound_descriptors=self.compound_descriptors,
            compound_id_col=compound_id_col,
            desc_id_col='id'
        )
        
        # Create system ID features
        for col in system_columns:
            self.feature_encoder.fit_one_hot_encoder(processed_data, col)
            encoded = self.feature_encoder.transform_one_hot(processed_data, col)
            processed_data = pd.concat([processed_data, encoded], axis=1)
            
        # Get system feature columns
        system_feature_cols = [col for col in processed_data.columns 
                             if col.startswith(tuple(col + '_' for col in system_columns))]
        
        # Get compound descriptor columns (exclude system, target, and metadata columns)
        exclude_cols = system_columns + target_columns + [compound_id_col, 'system_id', 'id', 'canonical_smiles', 'molecular_name']
        compound_feature_cols = [col for col in processed_data.columns 
                               if col not in exclude_cols and not col.startswith(tuple(col + '_' for col in system_columns))]
        
        # Handle NaN values in compound descriptors
        processed_data[compound_feature_cols] = processed_data[compound_feature_cols].fillna(0)
        
        # Scale compound descriptors
        self.feature_encoder.fit_scaler(
            data=processed_data,
            columns=compound_feature_cols,
            scaler_type='robust',
            scaler_name='compound_descriptors'
        )
        
        scaled_descriptors = self.feature_encoder.transform_scaler(
            data=processed_data,
            scaler_name='compound_descriptors'
        )
        
        processed_data[compound_feature_cols] = scaled_descriptors
        
        # Combine feature columns
        feature_cols = system_feature_cols + compound_feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = self.data_processor.split_data(
            target_columns=target_columns,
            feature_columns=feature_cols,
            test_size=test_size
        )
        
        print(f"Created {len(feature_cols)} features for compound model")
        print(f"- {len(system_feature_cols)} system features")
        print(f"- {len(compound_feature_cols)} compound descriptors")
        
        # Store for later use
        self.X_train_compound = X_train
        self.X_test_compound = X_test
        
        return X_train, X_test, y_train, y_test
    
    def prepare_full_model_data(self, 
                              target_columns: List[str],
                              system_columns: List[str],
                              compound_id_col: str = 'compound_id',
                              solvent1_col: str = 'solvent_1',
                              solvent2_col: str = 'solvent_2',
                              test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for the full model with compound and solvent descriptors.
        
        Args:
            target_columns: List of target column names
            system_columns: List of system parameter column names
            compound_id_col: Column name for compound ID
            solvent1_col: Column name for solvent 1
            solvent2_col: Column name for solvent 2
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test DataFrames
        """
        print("\nPreparing data for full model...")
        
        if self.compound_descriptors is None or self.solvent_descriptors is None:
            raise ValueError("Compound or solvent descriptors not loaded. Please load data first.")
            
        # Set raw data for data processor
        self.data_processor.set_raw_data(self.raw_data)
        
        # Process data
        processed_data = self.data_processor.process_data(
            target_columns=target_columns,
            system_columns=system_columns,
            compound_id_col=compound_id_col
        )
        
        # Merge compound descriptors
        processed_data = self.data_processor.merge_compound_descriptors(
            compound_descriptors=self.compound_descriptors,
            compound_id_col=compound_id_col,
            desc_id_col='id'
        )
        
        # Merge solvent 1 descriptors
        processed_data = self.data_processor.merge_solvent_descriptors(
            solvent_descriptors=self.solvent_descriptors,
            solvent_col=solvent1_col,
            solvent_id_col='id',
            prefix='solv1'
        )
        
        # Merge solvent 2 descriptors
        processed_data = self.data_processor.merge_solvent_descriptors(
            solvent_descriptors=self.solvent_descriptors,
            solvent_col=solvent2_col,
            solvent_id_col='id',
            prefix='solv2'
        )
        
        # Create temperature feature (one-hot encode isn't appropriate for temperature)
        if 'temperature' in system_columns:
            self.feature_encoder.fit_scaler(
                data=processed_data,
                columns=['temperature'],
                scaler_type='standard',
                scaler_name='temperature'
            )
            
            scaled_temp = self.feature_encoder.transform_scaler(
                data=processed_data,
                scaler_name='temperature'
            )
            
            processed_data['temperature_scaled'] = scaled_temp['temperature']
            
            # Remove temperature from system_columns for one-hot encoding
            system_columns_for_encoding = [col for col in system_columns if col != 'temperature']
        else:
            system_columns_for_encoding = system_columns
            
        # Create system ID features (except temperature)
        for col in system_columns_for_encoding:
            self.feature_encoder.fit_one_hot_encoder(processed_data, col)
            encoded = self.feature_encoder.transform_one_hot(processed_data, col)
            processed_data = pd.concat([processed_data, encoded], axis=1)
            
        # Get system feature columns
        system_feature_cols = [col for col in processed_data.columns 
                             if col.startswith(tuple(col + '_' for col in system_columns_for_encoding))]
        
        # Add temperature if scaled
        if 'temperature' in system_columns:
            system_feature_cols.append('temperature_scaled')
            
        # Get compound descriptor columns
        exclude_cols = system_columns + target_columns + [compound_id_col, 'system_id', 'id', 
                                                        'canonical_smiles', 'molecular_name',
                                                        solvent1_col, solvent2_col]
        compound_feature_cols = [col for col in processed_data.columns 
                               if col not in exclude_cols 
                               and not col.startswith(tuple(col + '_' for col in system_columns_for_encoding))
                               and not col.startswith('solv1_')
                               and not col.startswith('solv2_')]
        
        # Get solvent descriptor columns
        solv1_feature_cols = [col for col in processed_data.columns if col.startswith('solv1_') 
                             and col != 'solv1_id']
        solv2_feature_cols = [col for col in processed_data.columns if col.startswith('solv2_') 
                             and col != 'solv2_id']
        
        # Handle NaN values in all descriptors
        all_desc_cols = compound_feature_cols + solv1_feature_cols + solv2_feature_cols
        processed_data[all_desc_cols] = processed_data[all_desc_cols].fillna(0)
        
        # Scale compound descriptors
        self.feature_encoder.fit_scaler(
            data=processed_data,
            columns=compound_feature_cols,
            scaler_type='robust',
            scaler_name='compound_descriptors_full'
        )
        
        scaled_comp_desc = self.feature_encoder.transform_scaler(
            data=processed_data,
            scaler_name='compound_descriptors_full'
        )
        
        processed_data[compound_feature_cols] = scaled_comp_desc
        
        # Scale solvent descriptors
        self.feature_encoder.fit_scaler(
            data=processed_data,
            columns=solv1_feature_cols + solv2_feature_cols,
            scaler_type='robust',
            scaler_name='solvent_descriptors'
        )
        
        scaled_solv_desc = self.feature_encoder.transform_scaler(
            data=processed_data,
            scaler_name='solvent_descriptors'
        )
        
        processed_data[solv1_feature_cols + solv2_feature_cols] = scaled_solv_desc[solv1_feature_cols + solv2_feature_cols]
        
        # Combine feature columns
        feature_cols = system_feature_cols + compound_feature_cols + solv1_feature_cols + solv2_feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = self.data_processor.split_data(
            target_columns=target_columns,
            feature_columns=feature_cols,
            test_size=test_size
        )
        
        print(f"Created {len(feature_cols)} features for full model")
        print(f"- {len(system_feature_cols)} system features")
        print(f"- {len(compound_feature_cols)} compound descriptors")
        print(f"- {len(solv1_feature_cols)} solvent 1 descriptors")
        print(f"- {len(solv2_feature_cols)} solvent 2 descriptors")
        
        # Store for later use
        self.X_train_full = X_train
        self.X_test_full = X_test
        
        return X_train, X_test, y_train, y_test
    
    def train_fixed_model(self, model_type: str = 'nn', **kwargs) -> None:
        """
        Train a model using only system parameters.
        
        Args:
            model_type: Type of model ('nn', 'rf', 'gb')
            **kwargs: Additional model parameters
        """
        if self.X_train_fixed is None:
            raise ValueError("Fixed model data not prepared. Run prepare_fixed_model_data first.")
            
        print(f"\nTraining {model_type} model on fixed data...")
        
        # Create model
        if model_type.lower() == 'nn':
            model = NeuralNetworkModel(name="fixed_nn", **kwargs)
        elif model_type.lower() == 'rf':
            model = ForestModel(name="fixed_rf", **kwargs)
        elif model_type.lower() == 'gb':
            model = GradientBoostingModel(name="fixed_gb", **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Train model
        history = model.train(self.X_train_fixed, self.y_train)
        
        # Evaluate model
        metrics = model.evaluate(self.X_test_fixed, self.y_test)
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
            
        # Store model
        model_name = f"fixed_{model_type}"
        self.models[model_name] = model
        
        # Save model
        model_path = os.path.join(self.output_path, model_name)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model
    
    def train_compound_model(self, model_type: str = 'nn', **kwargs) -> None:
        """
        Train a model using system parameters and compound descriptors.
        
        Args:
            model_type: Type of model ('nn', 'rf', 'gb')
            **kwargs: Additional model parameters
        """
        if self.X_train_compound is None:
            raise ValueError("Compound model data not prepared. Run prepare_compound_model_data first.")
            
        print(f"\nTraining {model_type} model on compound data...")
        
        # Create model
        if model_type.lower() == 'nn':
            model = NeuralNetworkModel(name="compound_nn", **kwargs)
        elif model_type.lower() == 'rf':
            model = ForestModel(name="compound_rf", **kwargs)
        elif model_type.lower() == 'gb':
            model = GradientBoostingModel(name="compound_gb", **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Train model
        history = model.train(self.X_train_compound, self.y_train)
        
        # Evaluate model
        metrics = model.evaluate(self.X_test_compound, self.y_test)
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
            
        # Store model
        model_name = f"compound_{model_type}"
        self.models[model_name] = model
        
        # Save model
        model_path = os.path.join(self.output_path, model_name)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model
    
    def train_full_model(self, model_type: str = 'nn', **kwargs) -> None:
        """
        Train a model using system, compound, and solvent descriptors.
        
        Args:
            model_type: Type of model ('nn', 'rf', 'gb')
            **kwargs: Additional model parameters
        """
        if self.X_train_full is None:
            raise ValueError("Full model data not prepared. Run prepare_full_model_data first.")
            
        print(f"\nTraining {model_type} model on full data...")
        
        # Create model
        if model_type.lower() == 'nn':
            model = NeuralNetworkModel(name="full_nn", **kwargs)
        elif model_type.lower() == 'rf':
            model = ForestModel(name="full_rf", **kwargs)
        elif model_type.lower() == 'gb':
            model = GradientBoostingModel(name="full_gb", **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Train model
        history = model.train(self.X_train_full, self.y_train)
        
        # Evaluate model
        metrics = model.evaluate(self.X_test_full, self.y_test)
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
            
        # Store model
        model_name = f"full_{model_type}"
        self.models[model_name] = model
        
        # Save model
        model_path = os.path.join(self.output_path, model_name)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model
    
    def analyze_feature_importance(self, model_type: str = 'rf') -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance across different model types.
        
        Args:
            model_type: Type of model to analyze ('rf', 'gb')
            
        Returns:
            Dictionary of feature importance DataFrames
        """
        if model_type.lower() not in ['rf', 'gb']:
            raise ValueError("Feature importance analysis requires tree-based models (rf, gb)")
            
        print("\nAnalyzing feature importance...")
        
        # Check if models exist
        model_names = [f"{prefix}_{model_type}" for prefix in ['fixed', 'compound', 'full']]
        existing_models = [name for name in model_names if name in self.models]
        
        if not existing_models:
            raise ValueError(f"No {model_type} models found. Train models first.")
            
        # Get feature importance for each model
        importance_dict = {}
        
        for model_name in existing_models:
            model = self.models[model_name]
            
            if hasattr(model, 'get_feature_importance'):
                importance_df = model.get_feature_importance()
                importance_dict[model_name] = importance_df
                
                print(f"\nTop 10 important features for {model_name}:")
                print(importance_df.head(10))
                
        return importance_dict
    
    def compare_models(self, model_types: List[str] = None) -> Dict[str, Dict]:
        """
        Compare performance of different models.
        
        Args:
            model_types: List of model types to compare (if None, use all available)
            
        Returns:
            Dictionary of model metrics
        """
        if not self.models:
            raise ValueError("No models available. Train models first.")
            
        if model_types is None:
            model_names = list(self.models.keys())
        else:
            prefixes = ['fixed', 'compound', 'full']
            model_names = [f"{prefix}_{type}" for prefix in prefixes for type in model_types]
            model_names = [name for name in model_names if name in self.models]
            
        if not model_names:
            raise ValueError("No matching models found")
            
        print("\nComparing models...")
        
        # Evaluate all models
        metrics_dict = {}
        
        for model_name in model_names:
            model = self.models[model_name]
            
            # Determine which test set to use
            if model_name.startswith('fixed'):
                X_test = self.X_test_fixed
            elif model_name.startswith('compound'):
                X_test = self.X_test_compound
            elif model_name.startswith('full'):
                X_test = self.X_test_full
            else:
                raise ValueError(f"Unknown model prefix: {model_name}")
                
            # Evaluate model
            metrics = model.evaluate(X_test, self.y_test)
            metrics_dict[model_name] = metrics
            
        return metrics_dict
    
    def visualize_predictions(self, model_names: List[str], 
                            example_index: int = 0,
                            solubility_1: float = None,
                            solubility_2: float = None) -> None:
        """
        Visualize predictions for a specific example.
        
        Args:
            model_names: List of model names to visualize
            example_index: Index of example in test set
            solubility_1: Solubility in solvent 1 (if None, use from data if available)
            solubility_2: Solubility in solvent 2 (if None, use from data if available)
        """
        if not self.models:
            raise ValueError("No models available. Train models first.")
            
        models_to_viz = [name for name in model_names if name in self.models]
        
        if not models_to_viz:
            raise ValueError("No matching models found")
            
        print(f"\nVisualizing predictions for example {example_index}...")
        
        # Get predictions from each model
        predictions = {}
        
        for model_name in models_to_viz:
            model = self.models[model_name]
            
            # Determine which test set to use
            if model_name.startswith('fixed'):
                X_test = self.X_test_fixed
            elif model_name.startswith('compound'):
                X_test = self.X_test_compound
            elif model_name.startswith('full'):
                X_test = self.X_test_full
            else:
                raise ValueError(f"Unknown model prefix: {model_name}")
                
            # Get prediction for this example
            example = X_test.iloc[[example_index]]
            pred = model.predict(example)[0]
            predictions[model_name] = pred
            
        # Get actual values
        actual = self.y_test.iloc[example_index].values
        
        # Print predictions vs actual
        print("\nPredictions vs Actual:")
        print(f"Actual: {actual}")
        
        for model_name, pred in predictions.items():
            print(f"{model_name}: {pred}")
            
        # Create visualization
        for model_name, pred in predictions.items():
            print(f"\nVisualization for {model_name}:")
            
            if solubility_1 is None or solubility_2 is None:
                print("Cannot visualize: solubility values not provided")
                continue
                
            # Try to get temperature from data
            if self.raw_data is not None and 'temperature' in self.raw_data.columns:
                temperature = self.raw_data['temperature'].iloc[example_index]
            else:
                temperature = 298.15
                
            # Get experimental data if available
            experimental = None
            
            # Create visualization
            SolventSwapVisualizer.plot_ja_predictions(
                predictions=pred,
                experimental=experimental,
                solubility_1=solubility_1,
                solubility_2=solubility_2,
                temperature=temperature
            )


# Example usage
if __name__ == "__main__":
    # Example paths - adjust these to your environment
    data_path = "../../output"
    database_path = "../../db/MasterDatabase.db"
    
    # Create modeling framework
    modeling = SolventSwapModeling(data_path, database_path)
    
    # Load data
    modeling.load_data()
    
    # Prepare data for different model types
    target_columns = ['J0', 'J1', 'J2']
    system_columns = ['solvent_1', 'solvent_2', 'temperature']
    
    # Fixed model data
    modeling.prepare_fixed_model_data(target_columns, system_columns)
    
    # Compound model data
    modeling.prepare_compound_model_data(target_columns, system_columns)
    
    # Full model data
    modeling.prepare_full_model_data(target_columns, system_columns)
    
    # Train models
    # You can choose which models to train based on your needs
    
    # Fixed models
    modeling.train_fixed_model('nn')
    modeling.train_fixed_model('rf')
    
    # Compound models
    modeling.train_compound_model('nn')
    modeling.train_compound_model('rf')
    
    # Full models
    modeling.train_full_model('nn')
    modeling.train_full_model('rf')
    
    # Analyze feature importance
    importance_dict = modeling.analyze_feature_importance('rf')
    
    # Compare models
    metrics_dict = modeling.compare_models()
    
    # Visualize predictions
    # Example solubility values - replace with actual values
    solubility_1 = 100.0  # g/L in solvent 1
    solubility_2 = 150.0  # g/L in solvent 2
    
    modeling.visualize_predictions(
        model_names=['fixed_nn', 'compound_nn', 'full_nn'],
        example_index=0,
        solubility_1=solubility_1,
        solubility_2=solubility_2
    )
    
    # Compare feature importance across models
    SolventSwapVisualizer.compare_feature_importance(importance_dict)
    
    # Compare model performance
    SolventSwapVisualizer.compare_models(metrics_dict, 
                                        X_test=modeling.X_test_fixed,
                                        y_test=modeling.y_test)
    
    # Plot parity charts
    model_dict = {
        'Fixed NN': modeling.models['fixed_nn'],
        'Compound NN': modeling.models['compound_nn'],
        'Full NN': modeling.models['full_nn']
    }
    
    SolventSwapVisualizer.plot_parity(model_dict,
                                     X_test=modeling.X_test_fixed,  # Use any test set
                                     y_test=modeling.y_test,
                                     target_names=['J0', 'J1', 'J2'])