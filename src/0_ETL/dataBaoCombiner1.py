import sqlite3
from tqdm import tqdm
from pubchempy import get_compounds
import logging

# Set up logging for console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_cid_from_name(cas):
    try:
        compounds = get_compounds(cas, 'name')
        if compounds:
            return compounds[0].cid
        return None
    except:
        logging.error(f"Failed to get CID for CAS: {cas}")
        return None

def get_all_properties(cid):
    if cid is None:
        return None
    
    try:
        compound = get_compounds(cid, 'cid')[0]
        return {
            'pubchem_id': cid,
            'canonical_smiles': compound.canonical_smiles,
            'molecular_weight': compound.molecular_weight,
            'molecular_name': compound.iupac_name
        }
    except:
        logging.error(f"Failed to get properties for CID: {cid}")
        return None

def main():
    conn = sqlite3.connect('db/MasterDatabase.db')
    cursor = conn.cursor()
    
    # Get existing drugs
    cursor.execute("SELECT cas, drug FROM bao_drugs")
    rows = cursor.fetchall()
    
    # Track statistics
    new_entries = 0
    updated_entries = 0
    failed_entries = 0
    
    # Process each drug
    for cas, drug in tqdm(rows, desc="Processing drugs"):
        try:
            cid = get_cid_from_name(cas)
            if cid is None:
                logging.warning(f"No PubChem ID found for CAS {cas} (drug: {drug})")
                failed_entries += 1
                continue
                
            properties = get_all_properties(cid)
            if properties is None:
                logging.warning(f"No properties found for CAS {cas} (drug: {drug})")
                failed_entries += 1
                continue
                
            # Check if pubchem_id already exists
            cursor.execute("""
                SELECT original_names, molecular_name 
                FROM compounds 
                WHERE pubchem_id = ?
            """, (properties['pubchem_id'],))
            
            existing_row = cursor.fetchone()
            
            if existing_row:
                # Update existing row by appending the drug name
                original_names, molecular_name = existing_row
                new_names = f"{original_names}|{drug}" if original_names else drug
                
                logging.info(f"Updating existing entry - PubChem ID: {properties['pubchem_id']}")
                logging.info(f"  Molecular Name: {molecular_name}")
                logging.info(f"  Original Names: {original_names}")
                logging.info(f"  Adding Drug: {drug}")
                logging.info(f"  New Names List: {new_names}")
                
                cursor.execute("""
                    UPDATE compounds 
                    SET original_names = ?
                    WHERE pubchem_id = ?
                """, (new_names, properties['pubchem_id']))
                
                updated_entries += 1
            else:
                # Insert new row
                logging.info(f"Adding new drug - CAS: {cas}, Name: {drug}")
                cursor.execute("""
                    INSERT INTO compounds (
                        pubchem_id, 
                        canonical_smiles, 
                        molecular_weight, 
                        molecular_name, 
                        original_names
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    properties['pubchem_id'],
                    properties['canonical_smiles'],
                    properties['molecular_weight'],
                    properties['molecular_name'],
                    drug
                ))
                new_entries += 1
                
        except Exception as e:
            logging.error(f"Error processing CAS {cas} (drug: {drug}): {str(e)}")
            failed_entries += 1
            continue
    
    # Log final statistics
    logging.info("=== Processing Complete ===")
    logging.info(f"New entries added: {new_entries}")
    logging.info(f"Existing entries updated: {updated_entries}")
    logging.info(f"Failed entries: {failed_entries}")
    logging.info(f"Total processed: {new_entries + updated_entries + failed_entries}")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()