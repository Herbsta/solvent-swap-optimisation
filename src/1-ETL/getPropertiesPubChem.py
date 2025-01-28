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

def get_all_properties(pub_chem_id : int) -> dict:
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
cursor.execute('''
DROP TABLE IF EXISTS failed_api_compounds;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS failed_api_compounds (
    compound_name TEXT PRIMARY KEY
)
''')

# Insert all failed names into the failed_api_compounds table
cursor.executemany('''
INSERT OR REPLACE INTO failed_api_compounds (compound_name)
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
# Close the connection
conn.close()

# Convert the fetched IDs to a flat list of numbers
dual_solvent_ids = [id[0] for id in dual_solvent_ids]
single_data_ids = [id[0] for id in single_data_ids]
cids = [cid[1] for cid in cids]

# Combine the lists and get unique IDs
unique_ids = set(dual_solvent_ids + single_data_ids + cids)

compounds_data = []

# Set up logging to stderr
logging.basicConfig(level=logging.INFO)

for pub_chem_id in tqdm(unique_ids, desc="Fetching compound properties", unit="compound"):
    try:
        properties = get_all_properties(pub_chem_id)
        compounds_data.append((pub_chem_id, properties['canonical_smiles'], properties['molecular_weight'], properties['molecular_name']))
    except Exception as e:
        logging.error(f"Error fetching properties for PubChem ID {pub_chem_id}: {e}")

# Connect to the database
conn = sqlite3.connect('db/MasterDatabase.db')
cursor = conn.cursor()

cursor.execute('''
DROP TABLE IF EXISTS compounds;
''')

# Create the compounds table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS compounds (
    pubchem_id INTEGER PRIMARY KEY,
    canonical_smiles TEXT,
    molecular_weight REAL,
    molecular_name TEXT
)
''')

# Insert all compounds data at once
cursor.executemany('''
INSERT OR REPLACE INTO compounds (pubchem_id, canonical_smiles, molecular_weight, molecular_name)
VALUES (?, ?, ?, ?)
''', compounds_data)


# Query to select distinct solvent names from api_dual_solvent_data
query_dual_solvent_1_names = "SELECT DISTINCT solvent_1 FROM api_dual_solvent_data"
cursor.execute(query_dual_solvent_1_names)
dual_solvent_1_names = cursor.fetchall()

query_dual_solvent_2_names = "SELECT DISTINCT solvent_2 FROM api_dual_solvent_data"
cursor.execute(query_dual_solvent_2_names)
dual_solvent_2_names = cursor.fetchall()

# Query to select distinct solvent names from api_single_solvent_data
query_single_solvent_1_names = "SELECT DISTINCT solvent FROM api_single_solvent_data"
cursor.execute(query_single_solvent_1_names)
single_solvent_1_names = cursor.fetchall()


# Convert the fetched names to a flat list of strings
dual_solvent_1_names = [name[0] for name in dual_solvent_1_names]
dual_solvent_2_names = [name[0] for name in dual_solvent_2_names]
single_solvent_1_names = [name[0] for name in single_solvent_1_names]

# Combine the lists and get unique names
unique_solvent_names = set(dual_solvent_1_names + dual_solvent_2_names + single_solvent_1_names)

# Initialize lists for solvent CIDs and names that couldn't retrieve CIDs
solvent_cids = []
failed_solvent_names = []

for name in tqdm(unique_solvent_names, desc="Fetching solvent CIDs", unit="solvent"):
    try:
        cid = get_cid_from_name(name)
        solvent_cids.append((name, cid))
    except Exception as e:
        logging.error(f"Error fetching CID for solvent name {name}: {e}")
        failed_solvent_names.append(name)

print(f"Failed to retrieve CIDs for {len(failed_solvent_names)} solvents")

# Create a new table for the failed solvent entries
cursor.execute('''
DROP TABLE IF EXISTS failed_api_solvents;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS failed_api_solvents (
    solvent_name TEXT PRIMARY KEY
)
''')

# Insert all failed solvent names into the failed_solvents table
cursor.executemany('''
INSERT OR REPLACE INTO failed_api_solvents (solvent_name)
VALUES (?)
''', [(name,) for name in failed_solvent_names])

# Commit the transaction
conn.commit()

# Query to select distinct solvent names from pubchem_dual_solvent_data
query_pubchem_dual_solvent_1_names = "SELECT DISTINCT solvent_1 FROM pubchem_dual_solvent_data"
cursor.execute(query_pubchem_dual_solvent_1_names)
pubchem_dual_solvent_1_names = cursor.fetchall()

query_pubchem_dual_solvent_2_names = "SELECT DISTINCT solvent_2 FROM pubchem_dual_solvent_data"
cursor.execute(query_pubchem_dual_solvent_2_names)
pubchem_dual_solvent_2_names = cursor.fetchall()

# Query to select distinct solvent names from pubchem_single_solvent_data
query_pubchem_single_solvent_1_names = "SELECT DISTINCT solvent FROM pubchem_single_solvent_data"
cursor.execute(query_pubchem_single_solvent_1_names)
pubchem_single_solvent_1_names = cursor.fetchall()

# Convert the fetched names to a flat list of strings
pubchem_dual_solvent_1_names = [name[0] for name in pubchem_dual_solvent_1_names]
pubchem_dual_solvent_2_names = [name[0] for name in pubchem_dual_solvent_2_names]
pubchem_single_solvent_1_names = [name[0] for name in pubchem_single_solvent_1_names]

# Combine the lists and get unique names
unique_pubchem_solvent_names = set(pubchem_dual_solvent_1_names + pubchem_dual_solvent_2_names + pubchem_single_solvent_1_names)

# Initialize lists for solvent CIDs and names that couldn't retrieve CIDs
pubchem_solvent_cids = []
failed_pubchem_solvent_names = []

for name in tqdm(unique_pubchem_solvent_names, desc="Fetching PubChem solvent CIDs", unit="solvent"):
    try:
        cid = get_cid_from_name(name)
        pubchem_solvent_cids.append((name, cid))
    except Exception as e:
        logging.error(f"Error fetching CID for PubChem solvent name {name}: {e}")
        failed_pubchem_solvent_names.append(name)

print(f"Failed to retrieve CIDs for {len(failed_pubchem_solvent_names)} PubChem solvents")

# Create a new table for the failed PubChem solvent entries
cursor.execute('''
DROP TABLE IF EXISTS failed_pubchem_solvents;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS failed_pubchem_solvents (
    solvent_name TEXT PRIMARY KEY
)
''')

# Insert all failed PubChem solvent names into the failed_pubchem_solvents table
cursor.executemany('''
INSERT OR REPLACE INTO failed_pubchem_solvents (solvent_name)
VALUES (?)
''', [(name,) for name in failed_pubchem_solvent_names])

# Commit the transaction
conn.commit()

# Combine all solvent CIDs into a big set
all_solvent_cids = set(solvent_cids + pubchem_solvent_cids)

solvent_properties_data = []

for name, cid in tqdm(all_solvent_cids, desc="Fetching solvent properties", unit="solvent"):
    try:
        properties = get_all_properties(cid)
        solvent_properties_data.append((cid, properties['canonical_smiles'], properties['molecular_weight'], properties['molecular_name']))
    except Exception as e:
        logging.error(f"Error fetching properties for solvent CID {cid}: {e}")

cursor.execute('''
DROP TABLE IF EXISTS solvents;
''')

# Create the solvents table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS solvents (
    pubchem_id INTEGER PRIMARY KEY,
    canonical_smiles TEXT,
    molecular_weight REAL,
    molecular_name TEXT
)
''')

# Insert all solvent properties data at once
cursor.executemany('''
INSERT OR REPLACE INTO solvents (pubchem_id, canonical_smiles, molecular_weight, molecular_name)
VALUES (?, ?, ?, ?)
''', solvent_properties_data)
# Commit the transaction and close the connection
conn.commit()
conn.close()
