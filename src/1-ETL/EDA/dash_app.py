import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import sqlite3

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

# Dash app setup
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='solubility-chart'),
    html.Button('Previous', id='prev-button', n_clicks=0),
    html.Button('Next', id='next-button', n_clicks=0),
    dcc.Checklist(
        id='binary-toggle',
        options=[
            {'label': 'Show Only Binary Systems', 'value': 'binary'}
        ],
        value=[]
    ),
    dcc.Store(id='current-index', data=0)
])

@app.callback(
    Output('solubility-chart', 'figure'),
    Output('current-index', 'data'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    Input('binary-toggle', 'value'),
    State('current-index', 'data')
)
def update_chart(prev_clicks, next_clicks, toggle_values, current_index):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    dir = 1
    if button_id == 'next-button':
        current_index = (current_index + 1) % len(data_dict)
    elif button_id == 'prev-button':
        current_index = (current_index - 1) % len(data_dict)
        dir = -1

    if 'binary' in toggle_values:
        while data_dict[current_index]['solvent_2'] == None:
            current_index = (current_index + dir)  % len(data_dict)
        entry = data_dict[current_index]
    else:
        entry = data_dict[current_index]
    
    df = entry['data']
    
    fig = px.scatter(df, x='solvent_weight_fraction', y='solubility_g_g', color='temperature', 
                     title=f'Solvent Weight Fraction vs Solubility (g/g) for Different Temperatures\nCompound ID: {entry["compound_id"]}, Solvent 1: {entry["solvent_1"]}, Solvent 2: {entry["solvent_2"]}')
    return fig, current_index

if __name__ == '__main__':
    app.run_server(debug=True)