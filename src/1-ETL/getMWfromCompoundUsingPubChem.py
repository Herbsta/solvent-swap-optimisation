import sqlite3
import time
from typing import Optional, Dict
import pubchempy as pcp
import time 
import urllib

def get_compound_data(identifier: str) -> Optional[Dict]:
    """
    Fetch compound data from PubChem using PubChemPy.
    
    Args:
        identifier: compound name
    """
    time.sleep(0.2)
    try:
        # Search by compound name to get compound
        compound = pcp.get_compounds(identifier, 'name')
        if not compound:
            print(f"Error fetching data for {identifier}")
            return None
        compound = compound[0]
        result = {"molecular_weight": compound.molecular_weight}
        return result
    
    except (pcp.PubChemHTTPError, urllib.error.URLError) as e:
        if 'SSL' in str(e):
            print(f"SSL error fetching data for {identifier}, retrying...")
            time.sleep(1)
            try:
                compound = pcp.get_compounds(identifier, 'name')
                if not compound:
                    print(f"Error fetching data for {identifier}")
                    return None
                compound = compound[0]
                result = {"molecular_weight": compound.molecular_weight}
                
                return result
            except Exception as e:
                print(f"Error fetching data for {identifier} after retry: {str(e)}")
                return None
        else:
            print(f"Error fetching data for {identifier}: {str(e)}")
            return None
    except Exception as e:
        print(f"Error fetching data for {identifier}: {str(e)}")
        return None

def get_distinct_compounds(db_path: str, table_name: str, column_name: str) -> list[str]:
    """Get distinct values from specified table and column."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = f"SELECT DISTINCT {column_name} FROM {table_name}"
    cursor.execute(query)
    compounds = set([row[0] for row in cursor.fetchall() if row[0] is not None])
    
    conn.close()
    return compounds

def create_database(db_path: str, source_table: str, target_table: str, compound_column: str):
    """
    Create compounds table with molecular weights.
    
    Args:
        db_path: Path to SQLite database
        source_table: Name of source table containing compounds
        target_table: Name of target table to create/update
        compound_column: Column name containing compound identifiers
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {target_table} (
            compound_name TEXT PRIMARY KEY,
            molecular_weight REAL
        )
    """
    
    cursor.execute(create_table_query)
    
    compounds = get_distinct_compounds(db_path, source_table, compound_column)
    
    for compound in compounds:
        print(f"Processing {compound}...")
        data = get_compound_data(compound)
        
        if data:
            cursor.execute(f"""
                INSERT OR REPLACE INTO {target_table} (compound_name, molecular_weight)
                VALUES (?, ?)
            """, (compound, data["molecular_weight"]))
            
        time.sleep(0.2)
    
    conn.commit()
    conn.close()




if __name__ == "__main__":
    db_configs = [
        {
            "db_path": "db/pubchemSolubilityDatabase.db",
            "source_table": "single_solvent_data",
            "target_table": "solvents_single",
            "compound_column": "solvent"
        },
        {
            "db_path": "db/pubchemSolubilityDatabase.db",
            "source_table": "dual_solvent_data",
            "target_table": "solvents_dual",
            "compound_column": "solvent_1"
        },
        {
            "db_path": "db/pubchemSolubilityDatabase.db",
            "source_table": "dual_solvent_data",
            "target_table": "solvents_dual",
            "compound_column": "solvent_2"
        },
        {
            "db_path": "db/apiSolubilityDatabase.db",
            "source_table": "single_solvent_data",
            "target_table": "solvents_single",
            "compound_column": "solvent"
        },
        {
            "db_path": "db/apiSolubilityDatabase.db",
            "source_table": "dual_solvent_data",
            "target_table": "solvents_dual",
            "compound_column": "solvent_1"
        },
        {
            "db_path": "db/apiSolubilityDatabase.db",
            "source_table": "dual_solvent_data",
            "target_table": "solvents_dual",
            "compound_column": "solvent_2"
        }
    ]

    for config in db_configs:
        print(f"Starting processing for {config['db_path']} - {config['source_table']} to {config['target_table']}")
        create_database(
            config["db_path"],
            config["source_table"],
            config["target_table"],
            config["compound_column"]
        )