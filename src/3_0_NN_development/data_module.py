"""
Data loading and processing module for solvent swap optimization.
"""
import os
import pandas as pd
import numpy as np
import sqlite3
from typing import List, Dict, Tuple, Optional, Union, Any


class DataLoader:
    """Handles data loading from various sources."""
    
    def __init__(self, data_path: str = None, database_path: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to CSV data files
            database_path: Path to SQLite database
        """
        self.data_path = data_path
        self.database_path = database_path
        self.raw_data = None
        self.compound_descriptors = None
        self.solvent_descriptors = None
        
    def load_data_from_csvs(self, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load data from CSV files matching the pattern.
        
        Args:
            pattern: Glob pattern to match CSV files
            
        Returns:
            DataFrame containing combined data
        """
        import glob
        
        if not self.data_path:
            raise ValueError("Data path not specified")
            
        all_files = glob.glob(os.path.join(self.data_path, pattern))
        df_list = []
        
        for file in all_files:
            df = pd.read_csv(file)
            df_list.append(df)
            
        if not df_list:
            raise ValueError(f"No CSV files found matching pattern {pattern}")
            
        self.raw_data = pd.concat(df_list, ignore_index=True)
        
        self.raw_data.drop(columns=['rmse','r2','mape','logmape'])
        return self.raw_data
    
    def load_compound_descriptors(self) -> pd.DataFrame:
        """
        Load compound descriptors from database.
            
        Returns:
            DataFrame with compound descriptors
        """
        if not self.database_path:
            raise ValueError("Database path not specified")
            
        conn = sqlite3.connect(self.database_path)
        
        try:
            query = f"SELECT * FROM compounds"
            self.compound_descriptors = pd.read_sql_query(query, conn)
            
            # Remove columns with only NaN values
            self.compound_descriptors = self.compound_descriptors.dropna(axis=1, how='all')
            
            self.compound_descriptors.drop(columns=['canonical_smiles','molecular_name'], inplace=True, errors='ignore')

            
        finally:
            conn.close()
            
        return self.compound_descriptors
    
    def load_solvent_descriptors(self) -> pd.DataFrame:
        """
        Load solvent descriptors from database.
            
        Returns:
            DataFrame with solvent descriptors
        """
        if not self.database_path:
            raise ValueError("Database path not specified")
            
        conn = sqlite3.connect(self.database_path)
        
        try:
            query = f"SELECT * FROM solvents"
            
            self.solvent_descriptors = pd.read_sql_query(query, conn)
            
            # Remove columns with only NaN values
            self.solvent_descriptors = self.solvent_descriptors.dropna(axis=1, how='all')
            
            self.solvent_descriptors.drop(columns=['canonical_smiles','molecular_name'], inplace=True, errors='ignore')

            
        finally:
            conn.close()
            
        return self.solvent_descriptors


class DataProcessor:
    """Processes raw data into model-ready format."""
    
    def __init__(self, raw_data: pd.DataFrame = None):
        """
        Initialize with raw data.
        
        Args:
            raw_data: Raw data DataFrame
        """
        self.raw_data = raw_data
        self.processed_data = None
        
    def set_raw_data(self, raw_data: pd.DataFrame) -> None:
        """
        Set the raw data.
        
        Args:
            raw_data: Raw data DataFrame
        """
        self.raw_data = raw_data
        self.processed_data = None

    def create_system(self, system_columns: List[str], 
                    ) -> pd.DataFrame:
        """
        Process raw data for modeling.
        
        Args:
            system_columns: List of system condition column names            
        Returns:
            Processed DataFrame
        """
        if self.raw_data is None:
            raise ValueError("Raw data not set")
        
        processed_data = self.raw_data.copy()
        
        processed_data['system'] =  self.raw_data[system_columns].astype(str).agg('-'.join, axis=1)
                
        self.processed_data = processed_data.reset_index(drop=True)
        return processed_data
    
    def drop_columns(self, columns: List[str]) -> pd.DataFrame:
        """
        Drop specified columns from the processed data.
        
        Args:
            columns: List of column names to drop
            
        Returns:
            DataFrame with specified columns dropped
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available, run create_system first")
            
        self.processed_data.drop(columns=columns, inplace=True, errors='ignore')
        return self.processed_data
    
    def merge_compound_descriptors(self, 
                                 compound_descriptors: pd.DataFrame,
                                 compound_id_col: str = 'compound_id',
                                 desc_id_col: str = 'id',
                                 prefix: str = None) -> pd.DataFrame:
        """
        Merge compound descriptors with processed data.
        
        Args:
            compound_descriptors: DataFrame with compound descriptors
            compound_id_col: Column name for compound ID in processed data
            desc_id_col: Column name for compound ID in descriptors
            prefix: Prefix to add to descriptor column names
            
        Returns:
            Merged DataFrame
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available, run create_system first")
        
        compound_descriptors = compound_descriptors.copy()
                
        compound_descriptors = compound_descriptors.add_prefix(prefix) if prefix else None
          
        df = self.processed_data.merge(
            compound_descriptors,
            how='inner',
            left_on=compound_id_col,
            right_on=prefix + desc_id_col
        )
                    
        self.processed_data = df
        return self.processed_data
    
    def merge_solvent_descriptors(self, 
                                solvent_descriptors: pd.DataFrame,
                                solvent_col: str,
                                desc_id_col: str = 'id',
                                prefix: str = None) -> pd.DataFrame:
        """
        Merge solvent descriptors with processed data.
        
        Args:
            solvent_descriptors: DataFrame with solvent descriptors
            solvent_col: Column name for solvent ID in processed data
            desc_id_col: Column name for solvent ID in descriptors
            prefix: Prefix to add to descriptor column names
            
        Returns:
            Merged DataFrame
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available, run process_data first")
            
        solv_desc = solvent_descriptors.copy()
        
        solv_desc = solv_desc.add_prefix(prefix) if prefix else None
                    
        # Merge
        df = self.processed_data.merge(
            solv_desc,
            how='inner',
            left_on=solvent_col,
            right_on=prefix + desc_id_col,
        )
            
        self.processed_data = df
        return self.processed_data
    
    def split_data(self, 
                 target_columns: List[str], 
                 feature_columns: Optional[List[str]] = None,
                 test_size: float = 0.2, 
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            target_columns: List of target column names
            feature_columns: List of feature column names (if None, uses all except targets)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        if self.processed_data is None:
            raise ValueError("Processed data not available")
            
        df = self.processed_data.copy()
        
        # Determine feature columns if not specified
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col not in target_columns]
            
        X = df[feature_columns]
        y = df[target_columns]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test



current_folder = os.path.dirname(os.path.abspath(__file__))

def CreateDataProcessor(
    raw_data_path: str = 'curve_fit_results_x_is_6.csv',
    system_columns: list[str] = ['solvent_1','solvent_2','compound_id','temperature'],
    )-> DataProcessor:
    """
    Create a DataProcessor instance with the given parameters.
    """
    data = DataLoader(
        data_path=f'{current_folder}/../../output',
        database_path=f'{current_folder}/../../db/MasterDatabase.db')

    data.load_data_from_csvs(raw_data_path)
    data.load_compound_descriptors()
    data.load_solvent_descriptors()
        
    dataprocessor = DataProcessor(data.raw_data)

    dataprocessor.create_system(system_columns)

    if 'compound_id' not in system_columns:
        dataprocessor.merge_compound_descriptors(
            compound_descriptors=data.compound_descriptors,
            compound_id_col='compound_id',
            desc_id_col='id',
            prefix='compound_'
        )

    if 'solvent_1' not in system_columns:
        dataprocessor.merge_solvent_descriptors(
            solvent_descriptors=data.solvent_descriptors,
            solvent_col='solvent_1',
            desc_id_col='id',
            prefix='solvent_1_'
        )
        
    if 'solvent_2' not in system_columns:
        dataprocessor.merge_solvent_descriptors(
            solvent_descriptors=data.solvent_descriptors,
            solvent_col='solvent_1',
            desc_id_col='id',
            prefix='solvent_1_'
        )
        
    return dataprocessor.processed_data
        