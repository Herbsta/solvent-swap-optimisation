import sqlite3
import logging
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_failed_entries():
    logging.info("Starting to remove failed entries.")
    # Connect to the database
    conn = sqlite3.connect('db/MasterDatabase.db')
    cursor = conn.cursor()

    # Get the list of failed compounds and solvents
    cursor.execute("SELECT compound_name FROM failed_compounds")
    failed_compounds = [row[0] for row in cursor.fetchall()]
    logging.info(f"Found {len(failed_compounds)} failed compounds.")

    cursor.execute("SELECT solvent_name FROM failed_solvents")
    failed_solvents = [row[0] for row in cursor.fetchall()]
    logging.info(f"Found {len(failed_solvents)} failed API solvents.")

    # Remove failed compounds from relevant tables
    for compound in tqdm(failed_compounds, desc="Removing failed compounds"):
        cursor.execute("DELETE FROM api_dual_solvent_data WHERE compound_name=?", (compound,))
        cursor.execute("DELETE FROM api_single_solvent_data WHERE compound_name=?", (compound,))

    # Remove failed solvents from relevant tables
    for solvent in tqdm(failed_solvents, desc="Removing failed solvents"):
        cursor.execute("DELETE FROM api_dual_solvent_data WHERE solvent_1=? OR solvent_2=?", (solvent, solvent))
        cursor.execute("DELETE FROM api_single_solvent_data WHERE solvent=?", (solvent,))
        cursor.execute("DELETE FROM pubchem_dual_solvent_data WHERE solvent_1=? OR solvent_2=?", (solvent, solvent))
        cursor.execute("DELETE FROM pubchem_single_solvent_data WHERE solvent=?", (solvent,))
    
    cursor.execute("DROP TABLE failed_compounds")
    cursor.execute("DROP TABLE failed_solvents")

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    logging.info("Finished removing failed entries.")

def remove_saturation_column():
    logging.info("Starting to remove 'saturation' column from relevant tables.")
    # Connect to the database
    conn = sqlite3.connect('db/masterdatabase.db')
    cursor = conn.cursor()

    # List of tables to remove the 'saturation' column from
    tables = [
        "api_dual_solvent_data",
        "api_single_solvent_data",
        "pubchem_dual_solvent_data",
        "pubchem_single_solvent_data"
    ]

    # Remove the 'saturation' column from each table
    for table in tqdm(tables, desc="Removing 'saturation' column"):
        cursor.execute(f"ALTER TABLE {table} DROP COLUMN saturation")
        logging.info(f"Removed 'saturation' column from {table}.")

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    logging.info("Finished removing 'saturation' column.")

def cleanup_volume_data():
    try:
        conn = sqlite3.connect('db/masterdatabase.db')
        cursor = conn.cursor()
        
        # Delete rows with volume/litre data and get count of deleted rows
        cursor.execute("""
            DELETE FROM pubchem_dual_solvent_data
            WHERE solubility_vol_vol IS NOT NULL 
            OR solvent_1_vol_fraction IS NOT NULL
            OR solubility_mol_litre IS NOT NULL
            OR solubility_g_l IS NOT NULL
        """)
        print(f"Rows deleted from pubchem_dual_solvent_data: {cursor.rowcount}")
        
        cursor.execute("""
            DELETE FROM pubchem_single_solvent_data
            WHERE solubility_vol_vol IS NOT NULL
            OR solubility_mol_litre IS NOT NULL
            OR solubility_g_l IS NOT NULL
        """)
        print(f"Rows deleted from pubchem_dual_solvent_data: {cursor.rowcount}")
        
        cursor.execute("""
            DELETE FROM api_dual_solvent_data
            WHERE solvent_1_vol_fraction IS NOT NULL
        """)
        
        print(f"Rows deleted from pubchem_dual_solvent_data: {cursor.rowcount}")
        
        # Drop volume/litre columns from dual solvent table
        cursor.execute("ALTER TABLE pubchem_dual_solvent_data DROP COLUMN solubility_vol_vol")
        cursor.execute("ALTER TABLE pubchem_dual_solvent_data DROP COLUMN solvent_1_vol_fraction")
        cursor.execute("ALTER TABLE pubchem_dual_solvent_data DROP COLUMN solubility_mol_litre")
        
        # Drop volume/litre columns from single solvent table
        cursor.execute("ALTER TABLE pubchem_single_solvent_data DROP COLUMN solubility_vol_vol")
        cursor.execute("ALTER TABLE pubchem_single_solvent_data DROP COLUMN solubility_mol_litre")
        cursor.execute("ALTER TABLE api_dual_solvent_data DROP COLUMN solvent_1_vol_fraction")
        
        conn.commit()
        
    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        
def execute_conversion_scripts():
    logging.info("Starting to execute conversion scripts.")
    # Connect to the database
    conn = sqlite3.connect('db/masterdatabase.db')
    cursor = conn.cursor()

    # Read and execute Conversion3.sql
    with open('src/1-ETL/Conversion3.sql', 'r') as file:
        conversion3_sql = file.read()
    cursor.executescript(conversion3_sql)
    logging.info("Executed Conversion3.sql script.")

    # Read and execute Conversion4.sql
    with open('src/1-ETL/Conversion4.sql', 'r') as file:
        conversion4_sql = file.read()
    cursor.executescript(conversion4_sql)
    logging.info("Executed Conversion4.sql script.")

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    logging.info("Finished executing conversion scripts.")
    
def remove_old_files():
    logging.info("Starting to remove old database files.")
    files_to_remove = [
        'db/apiSolubilityDatabase.db',
        'db/BaoSolubilityDatabase.db',
        'db/pubchemSolubilityDatabase.db'
    ]

    for file in files_to_remove:
        try:
            os.remove(file)
            logging.info(f"Removed file: {file}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file}")
        except Exception as e:
            logging.error(f"Error removing file {file}: {e}")

    logging.info("Finished removing old database files.")
    

def main():
    remove_failed_entries()
    remove_saturation_column()
    cleanup_volume_data()
    execute_conversion_scripts()
    remove_old_files()  

if __name__ == "__main__":
    main()