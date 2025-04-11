"""
Data loading and processing module for solvent swap optimization.
"""
import os
import pandas as pd
import numpy as np
import sqlite3
from typing import List, Dict, Tuple, Optional, Union, Any
from groups import ja_groups, javh_groups

current_folder = os.path.dirname(os.path.abspath(__file__))


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

    @staticmethod
    def CreateDataProcessor(
        raw_data_path: str = 'curve_fit_results_x_is_6.csv',
        system_columns: list[str] = ['solvent_1','solvent_2','compound_id','temperature'],
        extra_points: int = 0
        ):
        """
        Create a DataProcessor instance with the given parameters.
        """
        data = DataLoader(
            data_path=f'{current_folder}/../../output',
            database_path=f'{current_folder}/../../db/MasterDatabase.db')

        data.load_data_from_csvs(raw_data_path)
        data.load_compound_descriptors()
        data.load_solvent_descriptors()
            
        dataprocessor = DataProcessor(data.raw_data,system_columns=system_columns)

        dataprocessor.create_system()

        if 'compound_id' not in dataprocessor.system_columns:
            dataprocessor.merge_compound_descriptors(
                compound_descriptors=data.compound_descriptors,
                compound_id_col='compound_id',
                desc_id_col='id',
                prefix='compound_'
            )

        if 'solvent_1' not in dataprocessor.system_columns:
            dataprocessor.merge_solvent_descriptors(
                solvent_descriptors=data.solvent_descriptors,
                solvent_col='solvent_1',
                desc_id_col='id',
                prefix='solvent_1_'
            )
            
        if 'solvent_2' not in dataprocessor.system_columns:
            dataprocessor.merge_solvent_descriptors(
                solvent_descriptors=data.solvent_descriptors,
                solvent_col='solvent_1',
                desc_id_col='id',
                prefix='solvent_1_'
            )
        
        if extra_points > 0:
            dataprocessor.load_extra_solubility_data(extra_points)

        return dataprocessor, data
    
    def __init__(self, raw_data: pd.DataFrame = None, system_columns: List[str] = ['solvent_1','solvent_2','temperature','compound_id'], use_javh_groups: bool = False):
        """
        Initialize with raw data.
        
        Args:
            raw_data: Raw data DataFrame
        """
        self.raw_data = raw_data
        self.processed_data = None
        self.system_columns = system_columns
        self.group = javh_groups if use_javh_groups else ja_groups
        
    def set_raw_data(self, raw_data: pd.DataFrame) -> None:
        """
        Set the raw data.
        
        Args:
            raw_data: Raw data DataFrame
        """
        self.raw_data = raw_data
        self.processed_data = None

    def create_system(self) -> pd.DataFrame:
        """
        Process raw data for modeling.
                  
        Returns:
            Processed DataFrame
        """
        if self.raw_data is None:
            raise ValueError("Raw data not set")
        
        processed_data = self.raw_data.copy()
        
        processed_data['system'] =  self.raw_data[self.system_columns].astype(str).agg('-'.join, axis=1)
                
        self.processed_data = processed_data
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

    def get_column_names_for_split(self,
                                   target: List[str] = ['J0','J1','J2']) -> Tuple[List[str], List[str]]:
        if self.processed_data is None:
            raise ValueError("Processed data not available")
        data = self.processed_data.copy()

        features = [name for name in data.columns if name not in ['group_index','solvent_1','solvent_2','compound_id'] + target]
        features = [name for name in features if name not in ['rmse','r2','mape','logmape','group_index']]

        if 'temperature' in self.system_columns: features.remove('temperature')

        return features, target
    
    def get_data_split_df(self, target_columns: List[str]) -> pd.DataFrame:                
        if self.processed_data is None:
            raise ValueError("Processed data not available")
        
        feature_columns, target_columns = self.get_column_names_for_split(target=target_columns)
        
        df = self.processed_data.copy()
            
        X = df[feature_columns]
        y = df[target_columns]
        
        return X, y
    
    def load_extra_solubility_and_temperature_data(self,num) -> pd.DataFrame:
        pass
    
    def load_extra_solubility_data(self, num,weight=5) -> pd.DataFrame:
        """
        Load additional solubility data from self.groups based on weight_fraction range.
            
        Returns:
            Updated processed DataFrame with added solubility data
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available, run create_system first")
        
        filtered_data = []
        
        def get_solubility(weight_fraction, group_data):
            # Extract weight fractions from the group data
            weight_fractions = group_data['solvent_1_weight_fraction']
            
            # Find the closest weight fraction index
            closest_idx = min(range(len(weight_fractions)), 
                                key=lambda i: abs(weight_fractions[i] - weight_fraction))
            
            # Return the corresponding solubility value
            return group_data['solubility_g_g'][closest_idx]
        
        for _,row in self.processed_data.iterrows():
            group_index = row['group_index']
            
            group = self.group[group_index]

            column_06 = get_solubility(0.6, group)
            column_02 = get_solubility(0.2, group)    
            column_04 = get_solubility(0.4, group)
            column_08 = get_solubility(0.8, group)     
            column_00 = get_solubility(0.0, group)
            column_10 = get_solubility(1.0, group)

            row = {}
            row['group_index'] = group_index
            
            if num == 1:
                row['solubility_0.6'] = column_06
            elif num == 2:
                row['solubility_0.6'] = column_06
                row['solubility_0.2'] = column_02
            elif num == 3:
                row['solubility_0.6'] = column_06
                row['solubility_0.2'] = column_02
                row['solubility_0.4'] = column_04
            elif num == 4:
                row['solubility_0.6'] = column_06
                row['solubility_0.2'] = column_02
                row['solubility_0.4'] = column_04
                row['solubility_0.8'] = column_08
            else:
                row['solubility_0.6'] = column_06
                row['solubility_0.2'] = column_02
                row['solubility_0.4'] = column_04
                row['solubility_0.8'] = column_08
                row['solubility_0.0'] = column_00
                row['solubility_1.0'] = column_10
                 
            filtered_data.append(row)
        
        filtered_data = pd.DataFrame(filtered_data)

        for i in range(1, weight + 1):
            for col in filtered_data.columns:
                if col.startswith('solubility_'):
                    filtered_data[f"{col}_{i}"] = filtered_data[col]
                
        self.processed_data = self.processed_data.merge(
            filtered_data,
            how='inner',
            left_on='group_index',
            right_on='group_index',
        )
                
        
        return self.processed_data

        