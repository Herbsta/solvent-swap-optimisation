import pandas as pd
import sqlite3
from pathlib import Path

def excel_to_sqlite(excel_path, db_path):
    conn = sqlite3.connect(db_path)
    excel_file = pd.ExcelFile(excel_path)
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Convert column names to lowercase and snake case
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '')
        
        df.to_sql(
            name=sheet_name.lower().replace(' ', '_'),
            con=conn,
            if_exists='replace',
            index=False
        )
    
    conn.close()
    print(f"Database created at: {db_path}")
    print("Tables created:", ', '.join(excel_file.sheet_names))

if __name__ == "__main__":
    input_file = "assets/BaoDataset.xlsx"
    output_db = "db/BaoSolubilityDatabase.db"
    
    excel_to_sqlite(input_file, output_db)