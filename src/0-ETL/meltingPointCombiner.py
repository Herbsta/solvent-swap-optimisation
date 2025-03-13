import json
import pandas as pd
from collections import Counter
import sqlite3

def main():

    with open('assets/features/MeltingPoints.json', 'r') as file:
        data = json.load(file)

    melting_point_columns = []
    for item in data['solvents'] + data['compounds']:
        points = item.get('melting_points',False)
        if points:
            melting_point_columns += [point['label'].split('\n')[0] for point in points]

    melting_point_counts = Counter(melting_point_columns)
    ordered_keys = [key for key, _ in melting_point_counts.most_common()]
    ordered_keys.append(ordered_keys.pop(0))


    for molecule_type in data:
        for solvent in data[molecule_type]:
            mps = solvent.get('melting_points', None)
            first_value = None
            if mps:
                mp_dict = {mp['label'].split('\n')[0]: mp['value'] for mp in mps}
                for key in ordered_keys:
                    if key in mp_dict:
                        first_value = mp_dict[key]
                        break
                
                if first_value and isinstance(first_value, str):
                    first_value = first_value.replace('°C', '').strip()
                    first_value = first_value.replace('(Decomposes)', '').strip()
                    first_value = first_value.replace('(Sublimes)', '').strip()
                    first_value = first_value.replace('(Sealed)', '').strip()
                    first_value = first_value.replace('(Literature)', '').strip()
                    
                    if ' –' in first_value:
                        left, right = first_value.split(' –',1)
                        try:
                            first_value = (float(left) + float(right)) / 2
                        except ValueError:
                            raise
                    first_value = float(first_value) + 273.15                  
                else:
                    first_value = None
                    
            solvent['melting_points'] = first_value
                
    # Connect to the SQLite database
    conn = sqlite3.connect('db/MasterDatabase.db')
    cursor = conn.cursor()

    # Add the new column to the solvents table
    cursor.execute("ALTER TABLE solvents ADD COLUMN melting_point REAL")

    # Add the new column to the compounds table
    cursor.execute("ALTER TABLE compounds ADD COLUMN melting_point REAL")

    # Update the solvents table with the new melting point values
    for solvent in data['solvents']:
        cursor.execute("UPDATE solvents SET melting_point = ? WHERE id = ?", (solvent.get('melting_points',None), solvent['ID']))

    # Update the compounds table with the new melting point values
    for compound in data['compounds']:
        cursor.execute("UPDATE compounds SET melting_point = ? WHERE id = ?", (compound.get('melting_points',None), compound['ID']))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    
if __name__ == '__main__':
    main()