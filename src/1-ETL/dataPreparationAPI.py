import pandas as pd
import sqlite3
import os
from pathlib import Path
import re
from typing import List, Optional
from dataclasses import dataclass
import logging

@dataclass
class ProcessingStats:
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_records: int = 0

class ExcelToSQLite:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.stats = ProcessingStats()
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

    def _clean_compound_name(self, filename: str) -> str:
        """Clean compound name from filename."""
        name = filename.replace('.xlsx', '')
        # Add space between numbers and letters
        name = re.sub(r'(\d+)([A-Z])', r'\1 \2', name)
        # Add space between lowercase and uppercase
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return name

    def _create_database(self):
        """Create SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute('''
        CREATE TABLE IF NOT EXISTS solubility_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            compound_name TEXT,
            solubility REAL,
            saturation TEXT,
            temperature REAL,
            solvent TEXT,
            solvent_ratio TEXT,
            location TEXT,
            comments TEXT,
            reference TEXT
        )
        ''')
        
        # Create indices for common search columns
        cur.execute('CREATE INDEX IF NOT EXISTS idx_compound ON solubility_data(compound_name)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_solvent ON solubility_data(solvent)')
        
        conn.commit()
        conn.close()

    def _process_excel_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Process single Excel file and return DataFrame."""
        try:
            df = pd.read_excel(file_path)
            
            # Standardize column names
            column_map = {
                'Solubility': 'solubility',
                'Saturation': 'saturation',
                'Temperature (Solubility (MCS))': 'temperature',
                'Solvent (Solubility (MCS))': 'solvent',
                'Ratio of Solvents': 'solvent_ratio',
                'Location': 'location',
                'Comment (Solubility (MCS))': 'comments',
                'Reference': 'reference'
            }
            df = df.rename(columns=column_map)
            
            # Add compound name and source file
            df['compound_name'] = self._clean_compound_name(file_path.name)
            
            # Convert numeric columns
            df['solubility'] = pd.to_numeric(df['solubility'], errors='coerce')
            df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
            
            return df
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            self.stats.failed_files += 1
            return None

    def process_directory(self, directory_path: str):
        """Process all Excel files in directory."""
        self._create_database()
        directory = Path(directory_path)
        excel_files = list(directory.glob('*.xlsx'))
        self.stats.total_files = len(excel_files)
        
        conn = sqlite3.connect(self.db_path)
        
        for file_path in excel_files:
            logging.info(f"Processing {file_path.name}")
            
            df = self._process_excel_file(file_path)
            if df is not None:
                # Write to database
                df.to_sql('solubility_data', conn, if_exists='append', index=False)
                self.stats.processed_files += 1
                self.stats.total_records += len(df)
                
                logging.info(f"Successfully processed {file_path.name}: {len(df)} records")
        
        conn.close()
        self._log_final_stats()

    def _log_final_stats(self):
        """Log final processing statistics."""
        logging.info("\n=== Processing Complete ===")
        logging.info(f"Total files: {self.stats.total_files}")
        logging.info(f"Successfully processed: {self.stats.processed_files}")
        logging.info(f"Failed: {self.stats.failed_files}")
        logging.info(f"Total records: {self.stats.total_records}")

def main():
    # Initialize converter
    converter = ExcelToSQLite('db/apiSolubilityDatabase.db')
    
    # Process all files in current directory
    converter.process_directory('assets/apiNameFiles')

    # Run the SQL script to further process the database
    sql_script_path = 'src/1-ETL/conversion1.sql'
    try:
        with sqlite3.connect('db/apiSolubilityDatabase.db') as conn:
            with open(sql_script_path, 'r') as sql_file:
                sql_script = sql_file.read()
            conn.executescript(sql_script)
            logging.info(f"Successfully executed SQL script: {sql_script_path}")
    except Exception as e:
        logging.error(f"Error executing SQL script {sql_script_path}: {str(e)}")

if __name__ == "__main__":
    main()