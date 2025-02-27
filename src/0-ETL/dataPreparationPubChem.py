import pandas as pd
import sqlite3
import glob
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ProcessingStats:
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_records: int = 0
    single_solvent_records: int = 0
    dual_solvent_records: int = 0

class PubChemToSQLite:
    def __init__(self, db_path: str, sql_script_path: str):
        self.db_path = db_path
        self.sql_script_path = sql_script_path
        self.stats = ProcessingStats()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    def _create_database(self):
        """Create initial SQLite database with imported_data table."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Create the initial imported_data table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS imported_data (
            pub_chem_id TEXT,
            solubility_g_l REAL,
            saturation TEXT,
            temperature_k REAL,
            solvent TEXT,
            ratio_of_solvents TEXT,
            comment TEXT,
            reference TEXT
        )
        ''')
        
        conn.commit()
        conn.close()

    def _safe_string_convert(self, value) -> str:
        """Safely convert a value to string, handling None and NaN."""
        if pd.isna(value):
            return ''
        return str(value)

    def _process_csv_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Process a single CSV file and return a DataFrame."""
        try:
            # Read CSV file with CP1252 encoding as in original script
            df = pd.read_csv(file_path, encoding='cp1252')
            
            # Clean up column names
            df.columns = [col.replace('¡¤', '/').replace('¡ã', '°') for col in df.columns]
            
            # Extract PubChem ID from filename
            pub_chem_id = os.path.splitext(os.path.basename(file_path))[0]
            df.insert(0, 'pub_chem_id', pub_chem_id)
            
            # Convert temperature to Kelvin
            df.insert(5, 'temperature_k', df['Temperature (Solubility (MCS)), °C'].apply(
                lambda x: float(x) + 273.15 if isinstance(x, (int, float)) else None
            ))
            
            # Select and rename columns for our standardized schema
            df = df.rename(columns={
                'Solubility, g/l-1': 'solubility_g_l',
                'Saturation': 'saturation',
                'Solvent (Solubility (MCS))': 'solvent',
                'Ratio of Solvents': 'ratio_of_solvents',
                'Comment (Solubility (MCS))': 'comment',
                'Reference': 'reference'
            })
            
            # Select only the columns we need
            columns_to_keep = [
                'pub_chem_id', 'solubility_g_l', 'saturation', 'temperature_k',
                'solvent', 'ratio_of_solvents', 'comment', 'reference'
            ]
            df = df[columns_to_keep]
            
            # Convert text columns to string type and handle NaN values
            text_columns = ['solvent', 'ratio_of_solvents', 'comment', 'reference', 'saturation']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self._safe_string_convert)
            
            # Drop rows with null temperature
            df = df.dropna(subset=['temperature_k'])
            
            # Now safe to use string operations since we've converted to strings
            df = df[~df['solvent'].str.contains('various solvent', na=False, case=False)]
            
            # Convert numeric columns to float, replacing invalid values with NULL
            numeric_columns = ['solubility_g_l', 'temperature_k']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            self.stats.failed_files += 1
            return None

    def _execute_sql_conversion(self) -> Tuple[int, int]:
        """Execute the SQL conversion script and return record counts."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Read and execute the SQL script
            with open(self.sql_script_path, 'r') as sql_file:
                sql_script = sql_file.read()
                conn.executescript(sql_script)
            
            # Get record counts
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM single_solvent_data")
            single_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM dual_solvent_data")
            dual_count = cur.fetchone()[0]
            
            conn.close()
            return single_count, dual_count
            
        except Exception as e:
            logging.error(f"Error executing SQL conversion: {str(e)}")
            return 0, 0

    def process_directory(self, directory_path: str):
        """Process all CSV files in the specified directory."""
        self._create_database()
        directory = Path(directory_path)
        csv_files = list(directory.glob('*.csv'))
        self.stats.total_files = len(csv_files)
        
        # Create connection for bulk insert
        conn = sqlite3.connect(self.db_path)
        
        # Process each CSV file
        for file_path in csv_files:
            logging.info(f"Processing {file_path.name}")
            
            df = self._process_csv_file(file_path)
            if df is not None:
                # Write to imported_data table
                df.to_sql('imported_data', conn, if_exists='append', index=False)
                self.stats.processed_files += 1
                self.stats.total_records += len(df)
                
                logging.info(f"Successfully processed {file_path.name}: {len(df)} records")
        
        conn.close()
        
        # Execute SQL conversion script
        self.stats.single_solvent_records, self.stats.dual_solvent_records = self._execute_sql_conversion()
        
        self._log_final_stats()

    def _log_final_stats(self):
        """Log final processing statistics."""
        logging.info("\n=== Processing Complete ===")
        logging.info(f"Total files: {self.stats.total_files}")
        logging.info(f"Successfully processed: {self.stats.processed_files}")
        logging.info(f"Failed: {self.stats.failed_files}")
        logging.info(f"Total imported records: {self.stats.total_records}")
        logging.info(f"Single solvent records: {self.stats.single_solvent_records}")
        logging.info(f"Dual solvent records: {self.stats.dual_solvent_records}")

def main():
    # Initialize converter with paths
    converter = PubChemToSQLite(
        db_path='db/pubchemSolubilityDatabase.db',
        sql_script_path='src/1-ETL/Conversion2.sql'
    )
    
    # Process all files in directory
    converter.process_directory('assets/data/pubchemIDFiles')

if __name__ == "__main__":
    main()