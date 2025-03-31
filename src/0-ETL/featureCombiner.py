import sqlite3
import pandas as pd
import os

def main():
    # Connect to the master database
    master_db = sqlite3.connect('db/MasterDatabase.db')

    # Create a new database for working
    working_db = sqlite3.connect('db/WorkingDatabase.db')

    master_db.backup(working_db)
    master_db.close()

    # Drop the tables bao_drugs and bao_solvents if they exist
    working_db.execute("DROP TABLE IF EXISTS bao_drugs")
    working_db.execute("DROP TABLE IF EXISTS bao_solvents")

    df = pd.read_excel('assets/features/compounds_and_solvents.xlsx')
    df.rename(columns={'PUBCHEM_COMPOUND_CID': 'id'}, inplace=True)

    # Remove columns prefixed with 'pubchem_' and the 'mol' column
    df = df.loc[:, ~df.columns.str.startswith('PUBCHEM_')]
    df = df.drop(columns=['mol'])

    solvents_df = pd.read_sql_query("SELECT * FROM solvents", working_db).drop(columns=['original_names'])
    merged_df = pd.merge(solvents_df, df, left_on='pubchem_id', right_on='id', how='inner').drop(columns=['pubchem_id'])
    merged_df.to_sql('solvents', working_db, if_exists='replace', index=False)

    compounds_df = pd.read_sql_query("SELECT * FROM compounds", working_db).drop(columns=['original_names'])
    merged_df = pd.merge(compounds_df, df, left_on='pubchem_id', right_on='id', how='inner').drop(columns=['pubchem_id'])
    merged_df.to_sql('compounds', working_db, if_exists='replace', index=False)

    # Remove rows from solubility table if compound_id, solvent_1, or solvent_2 do not have a match in the respective table
    working_db.execute(
        """
        DELETE FROM solubility
        WHERE compound_id NOT IN (SELECT id FROM compounds)
        OR solvent_1 NOT IN (SELECT id FROM solvents)
        OR solvent_2 NOT IN (SELECT id FROM solvents)
        """
    )

    working_db.commit()
    working_db.close()

    # Remove the MasterDatabase.db file
    os.remove('db/MasterDatabase.db')

    # Rename the working database to MasterDatabase.db
    os.rename('db/WorkingDatabase.db', 'db/MasterDatabase.db')

if __name__ == "__main__":
    main()