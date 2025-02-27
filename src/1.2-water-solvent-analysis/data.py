import sqlite3
import pandas as pd

# Connect to your SQLite database
conn = sqlite3.connect('db\MasterDatabase.db')

query = """
SELECT 
    compound_id AS api,
    CASE 
        WHEN solvent_1 = 962 THEN solvent_2
        ELSE solvent_1
    END AS solvent,
    ROUND(solubility_g_g, 4) as solubility,
    CASE 
        WHEN solvent_1 = 962 THEN 1 - ROUND(solvent_1_weight_fraction, 2)
        ELSE ROUND(solvent_1_weight_fraction, 2)
    END AS fraction,
    ROUND(temperature) as temperature 
FROM
    solubility s
WHERE 
    (solvent_2 = 962 OR solvent_1 = 962)
    AND solvent_2 IS NOT NULL
    AND solubility <> 0
    AND fraction <> 1
    AND fraction <> 0;
"""

# Read the SQL query into a DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

print(df)