import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button


# Connect to the SQLite database
conn = sqlite3.connect('db/MasterDatabase.db')
cursor = conn.cursor()
# Execute the query
cursor.execute('''
SELECT 
compound_id,
solvent_1,
solvent_2,
GROUP_CONCAT(temperature) as temperature,	
GROUP_CONCAT(solvent_1_weight_fraction) as solvent_weight_fraction,	
GROUP_CONCAT(solvent_1_mol_fraction) as solvent_mol_fraction,	
GROUP_CONCAT(solubility_mol_mol) as solubility_mol_mol,	
GROUP_CONCAT(solubility_g_g) as solubility_g_g	
FROM solubility
GROUP BY compound_id, solvent_1, solvent_2
ORDER BY compound_id, solvent_1, solvent_2
''')

results = cursor.fetchall()

# Convert the results to a dictionary
columns = [desc[0] for desc in cursor.description]


results_dict = [
    {col: (list(map(float, row[i].split(','))) if isinstance(row[i], str) and col in ['solubility_mol_mol', 'solubility_g_g','solvent_weight_fraction','solvent_mol_fraction','temperature'] else row[i])
     for i, col in enumerate(columns)}
    for row in results
]

# Remove entries with null compound_id from results_dict
results_dict = [entry for entry in results_dict if entry['compound_id'] is not None]

# Close the connection
conn.close()

data_dict = []
for entry in results_dict:
    try:
        data = pd.DataFrame({
                'temperature': entry['temperature'],
                'solvent_weight_fraction': entry['solvent_weight_fraction'],
                'solvent_mol_fraction': entry['solvent_mol_fraction'],
                'solubility_mol_mol': entry['solubility_mol_mol'],
                'solubility_g_g': entry['solubility_g_g']
        })
        data_dict.append({
            'compound_id': entry['compound_id'],	
            'solvent_1': entry['solvent_1'],	
            'solvent_2': entry['solvent_2'],
            'data': data	
        })
    except:
        continue

print(len(data_dict))


index = 0

import matplotlib.pyplot as plt

class IndexTracker:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.index = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.plot_data()

        axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.01, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev)

    def plot_data(self,dir=1):
        self.ax.clear()
        while self.data_dict[self.index]['data']['solvent_weight_fraction'][0] == 1.0:
            self.index += dir
            if self.index >= len(self.data_dict):
                self.index = 0

        df = self.data_dict[self.index]['data']
        for temp in df['temperature'].unique():
            temp_df = df[df['temperature'] == temp]
            self.ax.scatter(temp_df['solvent_weight_fraction'], temp_df['solubility_g_g'], label=f'Temperature {temp} K')

        self.ax.set_xlabel('Solvent Weight Fraction')
        self.ax.set_ylabel('Solubility (g/g)')
        self.ax.set_title(f'Solvent Weight Fraction vs Solubility (g/g) for Different Temperatures\nCompound ID: {self.data_dict[self.index]["compound_id"]}, Solvent 1: {self.data_dict[self.index]["solvent_1"]}, Solvent 2: {self.data_dict[self.index]["solvent_2"]}')
        self.ax.legend()
        self.fig.canvas.draw()

    def next(self, event):
        self.index += 1
        if self.index >= len(self.data_dict):
            self.index = 0
        self.plot_data()

    def prev(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.data_dict) - 1
        print(self.index)
        self.plot_data(dir=-1)

tracker = IndexTracker(data_dict)
plt.show()
