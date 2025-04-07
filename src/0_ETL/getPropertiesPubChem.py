import sqlite3
import time
import pubchempy as pcp
import logging
from tqdm import tqdm
import urllib

def get_cid_from_name(compound_name: str) -> int:
    while True:
        try:
            compound = pcp.get_compounds(compound_name, 'name')
            if compound:
                return compound[0].cid
            else:
                raise ValueError(f"No compound found for name: {compound_name}")
        except (pcp.PubChemHTTPError, urllib.error.URLError) as e:
            if 'SSL' in str(e):
                logging.warning(f"SSL error for compound name {compound_name}, retrying...")
                continue
            else:
                logging.error(f"Error fetching CID for compound name {compound_name}: {e}")
                raise

def get_all_properties(pub_chem_id: int) -> dict:
    while True:
        try:
            time.sleep(0.2)
            compound = pcp.Compound.from_cid(pub_chem_id)
            properties = {
                'canonical_smiles': compound.canonical_smiles,
                'molecular_weight': compound.molecular_weight,
                'molecular_name': compound.iupac_name
            }
            return properties
        except (pcp.PubChemHTTPError, urllib.error.URLError) as e:
            if 'SSL' in str(e):
                logging.warning(f"SSL error for PubChem ID {pub_chem_id}, retrying...")
                continue
            else:
                raise

def main():
    # Connect to the database
    conn = sqlite3.connect('db/MasterDatabase.db')
    cursor = conn.cursor()

    # Query to select distinct compound names from api_dual_solvent_data
    query_dual_solvent_names = "SELECT DISTINCT compound_name FROM api_dual_solvent_data"
    cursor.execute(query_dual_solvent_names)
    dual_solvent_names = cursor.fetchall()

    # Query to select distinct compound names from api_single_solvent_data
    query_single_data_names = "SELECT DISTINCT compound_name FROM api_single_solvent_data"
    cursor.execute(query_single_data_names)
    single_data_names = cursor.fetchall()

    # Convert the fetched names to a flat list of strings
    dual_solvent_names = [name[0] for name in dual_solvent_names]
    single_data_names = [name[0] for name in single_data_names]

    # Combine the lists and get unique names
    unique_names = set(dual_solvent_names + single_data_names)

    # Initialize lists for CIDs and names that couldn't retrieve CIDs
    cids = []
    failed_names = []

    for name in tqdm(unique_names, desc="Fetching CIDs", unit="compound"):
        try:
            cid = get_cid_from_name(name)
            cids.append((name, cid))
        except Exception as e:
            logging.error(f"Error fetching CID for compound name {name}: {e}")
            failed_names.append(name)

    print(f"Failed to retrieve CIDs for {len(failed_names)} compounds")

    # Create a new table for the failed entries
    cursor.execute('DROP TABLE IF EXISTS failed_compounds;')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS failed_compounds (
        compound_name TEXT PRIMARY KEY
    )
    ''')

    # Insert all failed names into the failed_api_compounds table
    cursor.executemany('''
    INSERT OR REPLACE INTO failed_compounds (compound_name)
    VALUES (?)
    ''', [(name,) for name in failed_names])

    # Commit the transaction
    conn.commit()

    # Query to select distinct pubchem_id from pubchem_dual_solvent_data
    query_dual_solvent = "SELECT DISTINCT pub_chem_id FROM pubchem_dual_solvent_data"
    cursor.execute(query_dual_solvent)
    dual_solvent_ids = cursor.fetchall()

    # Query to select distinct pubchem_id from single_data
    query_single_data = "SELECT DISTINCT pub_chem_id FROM pubchem_single_solvent_data"
    cursor.execute(query_single_data)
    single_data_ids = cursor.fetchall()

    # Convert the fetched IDs to a flat list of numbers
    dual_solvent_ids = [id[0] for id in dual_solvent_ids]
    single_data_ids = [id[0] for id in single_data_ids]
    
    # Create a mapping of CIDs to original names
    cid_to_names = {}
    for name, cid in cids:
        if cid not in cid_to_names:
            cid_to_names[cid] = set()
        cid_to_names[cid].add(name)

    # Combine the lists and get unique IDs
    unique_ids = set(dual_solvent_ids + single_data_ids + [cid[1] for cid in cids])

    compounds_data = []

    # Set up logging to stderr
    logging.basicConfig(level=logging.INFO)

    for pub_chem_id in tqdm(unique_ids, desc="Fetching compound properties", unit="compound"):
        try:
            properties = get_all_properties(pub_chem_id)
            original_names = cid_to_names.get(pub_chem_id, set())
            original_names_str = '|'.join(sorted(original_names)) if original_names else None
            compounds_data.append((
                pub_chem_id,
                properties['canonical_smiles'],
                properties['molecular_weight'],
                properties['molecular_name'],
                original_names_str
            ))
        except Exception as e:
            logging.error(f"Error fetching properties for PubChem ID {pub_chem_id}: {e}")

    # Connect to the database
    cursor.execute('DROP TABLE IF EXISTS compounds;')

    # Create the compounds table with the original_names column
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS compounds (
        pubchem_id INTEGER PRIMARY KEY,
        canonical_smiles TEXT,
        molecular_weight REAL,
        molecular_name TEXT,
        original_names TEXT
    )
    ''')

    # Insert all compounds data at once
    cursor.executemany('''
    INSERT OR REPLACE INTO compounds 
    (pubchem_id, canonical_smiles, molecular_weight, molecular_name, original_names)
    VALUES (?, ?, ?, ?, ?)
    ''', compounds_data)

    # Solvent section
    # Query to select distinct solvent names from all tables
    queries = [
        ("SELECT DISTINCT solvent_1 FROM api_dual_solvent_data", "solvent_1"),
        ("SELECT DISTINCT solvent_2 FROM api_dual_solvent_data", "solvent_2"),
        ("SELECT DISTINCT solvent FROM api_single_solvent_data", "solvent"),
        ("SELECT DISTINCT solvent_1 FROM pubchem_dual_solvent_data", "solvent_1"),
        ("SELECT DISTINCT solvent_2 FROM pubchem_dual_solvent_data", "solvent_2"),
        ("SELECT DISTINCT solvent FROM pubchem_single_solvent_data", "solvent")
    ]

    all_solvent_names = set()
    for query, _ in queries:
        cursor.execute(query)
        names = [row[0] for row in cursor.fetchall()]
        all_solvent_names.update(names)

    # Initialize lists for solvent CIDs and names that couldn't retrieve CIDs
    solvent_cids = []
    failed_solvent_names = []
    solvent_name_to_cid = {}

    for name in tqdm(all_solvent_names, desc="Fetching solvent CIDs", unit="solvent"):
        try:
            cid = get_cid_from_name(name)
            solvent_cids.append((name, cid))
            if cid not in solvent_name_to_cid:
                solvent_name_to_cid[cid] = set()
            solvent_name_to_cid[cid].add(name)
        except Exception as e:
            logging.error(f"Error fetching CID for solvent name {name}: {e}")
            failed_solvent_names.append(name)

    print(f"Failed to retrieve CIDs for {len(failed_solvent_names)} solvents")

    # Create and populate failed_api_solvents table
    cursor.execute('DROP TABLE IF EXISTS failed_solvents;')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS failed_solvents (
        solvent_name TEXT PRIMARY KEY
    )
    ''')
    cursor.executemany('''
    INSERT OR REPLACE INTO failed_solvents (solvent_name)
    VALUES (?)
    ''', [(name,) for name in failed_solvent_names])

    # Fetch properties for all unique solvent CIDs
    solvent_properties_data = []
    unique_solvent_cids = set(cid for _, cid in solvent_cids)

    for cid in tqdm(unique_solvent_cids, desc="Fetching solvent properties", unit="solvent"):
        try:
            properties = get_all_properties(cid)
            original_names = solvent_name_to_cid.get(cid, set())
            original_names_str = '|'.join(sorted(original_names)) if original_names else None
            solvent_properties_data.append((
                cid,
                properties['canonical_smiles'],
                properties['molecular_weight'],
                properties['molecular_name'],
                original_names_str
            ))
        except Exception as e:
            logging.error(f"Error fetching properties for solvent CID {cid}: {e}")

    # Create and populate solvents table
    cursor.execute('DROP TABLE IF EXISTS solvents;')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS solvents (
        pubchem_id INTEGER PRIMARY KEY,
        canonical_smiles TEXT,
        molecular_weight REAL,
        molecular_name TEXT,
        original_names TEXT
    )
    ''')

    cursor.executemany('''
    INSERT OR REPLACE INTO solvents 
    (pubchem_id, canonical_smiles, molecular_weight, molecular_name, original_names)
    VALUES (?, ?, ?, ?, ?)
    ''', solvent_properties_data)

    # Commit and close
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()