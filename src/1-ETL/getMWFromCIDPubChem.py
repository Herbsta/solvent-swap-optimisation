import sqlite3
import pubchempy as pcp
import logging
import time

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Connect to the database
    conn = sqlite3.connect('db/pubchemSolubilityDatabase.db')
    cursor = conn.cursor()
    logging.info('Connected to the database.')

    # Create the compounds table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS compounds (
        pub_chem_id INTEGER PRIMARY KEY,
        molecular_weight REAL
    )
    ''')
    logging.info('Compounds table created or already exists.')

    # Function to get molecular weight from PubChem
    def get_molecular_weight(pub_chem_id):
        time.sleep(0.2)
        compound = pcp.Compound.from_cid(pub_chem_id)
        return compound.molecular_weight

    # Fetch pub_chem_id from single_solvent_data and dual_solvent_data
    cursor.execute('SELECT DISTINCT pub_chem_id FROM single_solvent_data')
    single_solvent_ids = cursor.fetchall()
    logging.info('Fetched pub_chem_id from single_solvent_data.')

    cursor.execute('SELECT DISTINCT pub_chem_id FROM dual_solvent_data')
    dual_solvent_ids = cursor.fetchall()
    logging.info('Fetched pub_chem_id from dual_solvent_data.')

    # Combine and deduplicate pub_chem_ids
    pub_chem_ids = set([id[0] for id in single_solvent_ids + dual_solvent_ids])
    logging.info(f'Combined and deduplicated pub_chem_ids: {len(pub_chem_ids)} unique IDs found.')

    # Insert molecular weights into the compounds table
    for pub_chem_id in pub_chem_ids:
        try:
            molecular_weight = get_molecular_weight(pub_chem_id)
            cursor.execute('''
            INSERT OR IGNORE INTO compounds (pub_chem_id, molecular_weight)
            VALUES (?, ?)
            ''', (pub_chem_id, molecular_weight))
            logging.info(f'Inserted molecular weight for PubChem ID {pub_chem_id}.')
        except Exception as e:
            logging.error(f"Error fetching data for PubChem ID {pub_chem_id}: {e}")

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    logging.info('Changes committed and database connection closed.')

if __name__ == "__main__":
    main()