import requests
import sqlite3
import time
from typing import Optional, Dict

def get_compound_data(identifier: str, use_cid: bool = False, include_density: bool = False) -> Optional[Dict]:
    """
    Fetch compound data from PubChem API using either CID or compound name.
    
    Args:
        identifier: Either a PubChem CID or compound name
        use_cid: If True, treat identifier as CID; if False, treat as compound name
        include_density: Whether to include density data in results
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    try:
        if not use_cid:
            # Search by compound name to get CID
            search_url = f"{base_url}/compound/name/{identifier}/cids/JSON"
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            if "IdentifierList" not in data:
                return None
                
            cid = data["IdentifierList"]["CID"][0]
        else:
            # Use provided CID directly
            cid = identifier
        
        properties = "MolecularWeight,Density" if include_density else "MolecularWeight"
        props_url = f"{base_url}/compound/cid/{cid}/property/{properties}/JSON"
        response = requests.get(props_url)
        response.raise_for_status()
        prop_data = response.json()
        
        if "PropertyTable" not in prop_data:
            return None
            
        properties = prop_data["PropertyTable"]["Properties"][0]
        result = {"molecular_weight": properties.get("MolecularWeight")}
        
        if include_density:
            result["density"] = properties.get("Density")
        
        return result
        
    except (requests.RequestException, KeyError, IndexError) as e:
        print(f"Error fetching data for {identifier}: {str(e)}")
        return None

def get_distinct_compounds(db_path: str, table_name: str, column_name: str) -> list[str]:
    """Get distinct values from specified table and column."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = f"SELECT DISTINCT {column_name} FROM {table_name}"
    cursor.execute(query)
    compounds = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return compounds

def create_database(db_path: str, source_table: str, target_table: str, compound_column: str, 
                   use_cid: bool = False, include_density: bool = False):
    """
    Create compounds table with molecular weights and optional density.
    
    Args:
        db_path: Path to SQLite database
        source_table: Name of source table containing compounds
        target_table: Name of target table to create/update
        compound_column: Column name containing compound identifiers
        use_cid: If True, treat compound identifiers as CIDs
        include_density: Whether to include density data
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {target_table} (
            compound_name TEXT PRIMARY KEY,
            molecular_weight REAL
            {', density REAL' if include_density else ''}
        )
    """
    
    cursor.execute(create_table_query)
    
    compounds = get_distinct_compounds(db_path, source_table, compound_column)
    
    for compound in compounds:
        print(f"Processing {compound}...")
        data = get_compound_data(compound, use_cid, include_density)
        
        if data:
            if include_density:
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {target_table} (compound_name, molecular_weight, density)
                    VALUES (?, ?, ?)
                """, (compound, data["molecular_weight"], data.get("density")))
            else:
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {target_table} (compound_name, molecular_weight)
                    VALUES (?, ?)
                """, (compound, data["molecular_weight"]))
            
        time.sleep(0.2)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = "db/pubchem_solubility_database.db"
    source_table = "single_solvent_data"
    target_table = "pub_chem_id"
    compound_column = "pub_chem_id"
    create_database(db_path, source_table, target_table, compound_column, use_cid=True, include_density=False)