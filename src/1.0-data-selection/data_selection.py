import sqlite3
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import random
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import warnings

random.seed(42)

# Connect to the database
db_path = 'db/MasterDatabase.db'
connection = sqlite3.connect(db_path)

# Query to select the required columns
query = f"""
SELECT round(solvent_1_weight_fraction,4) as solvent_1_weight_fraction, solubility_g_g, temperature, solvent_2, solvent_1, compound_id
FROM solubility
WHERE solvent_2 IS NOT NULL 
AND round(solubility_g_g, 4) >= 0.0001
"""

# Execute the query and load the data into a pandas DataFrame
df = pd.read_sql_query(query, connection)

# Close the database connection
connection.close()

groups = [group.reset_index(drop=True) for _, group in df.groupby(['solvent_1', 'solvent_2', 'compound_id','temperature'])]

def has_boundary_points(group):
    min_fraction = group['solvent_1_weight_fraction'].min()
    max_fraction = group['solvent_1_weight_fraction'].max()
    return min_fraction <= 0.01 and max_fraction >= 0.99

# Filter groups to keep only those with proper boundary conditions
filtered_groups = [group for group in groups if has_boundary_points(group)]

groups = filtered_groups

# Process groups to remove duplicate solvent_1_weight_fraction values
cleaned_groups = []

for group in groups:   
    cleaned_group = group.drop_duplicates(subset=['solvent_1_weight_fraction'], keep='last')
    cleaned_groups.append(cleaned_group)

removelist = [
    1,2,3,4,
    259,260,261,262,263,264,265,266,267,268,
    335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,
    397, 398,399,400,401,402,
    449,450,451,452,453,454,455,456,457,
    477,478,479,480,481,482,483,484,485,486,
    494, 495,496,497,498,499,500,
    1220, 1221, 1222, 1223, 1224, 1225, 1226,1227, 1228,
    1333, 1334, 1335, 1336, 1337, 1338, 1339
    ]



new_groups = [group for i,group in enumerate(cleaned_groups) if i not in removelist]

print("Number of groups before: ", len(groups))
print("Number of groups after: ", len(new_groups))

groups = new_groups

results = []
failed_groups = []
skipped_groups = []

for gn in tqdm(range(len(groups)), desc="Processing groups"):
    chosen_df = groups[gn]
            
    solvent_2_pure = chosen_df[chosen_df['solvent_1_weight_fraction'] <= 0.01].iloc[0]['solubility_g_g']
    solvent_1_pure = chosen_df[chosen_df['solvent_1_weight_fraction'] >= 0.99].iloc[0]['solubility_g_g']
    specific_temperature = chosen_df['temperature'].iloc[0]
    
    fitting_df = chosen_df
    
    def jouyban_acree(f1, J0, J1, J2):   
        # Calculate fraction of second solvent
        f2 = 1 - f1
        
        # Modified interaction term that reduces likelihood of bimodal behavior
        interaction_term = J0 * f1 * f2 + J1 * f1 * f2 * (2*f1 - 1) + J2 * f1 * f2 * (2*f1 - 1)**2
        
        # Calculate logarithm of solubility in the mixture
        log_Cm = f1 * np.log(solvent_1_pure) + f2 * np.log(solvent_2_pure) + \
                 interaction_term / specific_temperature
        
        # Return the solubility in the mixture
        return np.exp(log_Cm)
    

    # Suppress warnings during curve fitting
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(jouyban_acree, fitting_df['solvent_1_weight_fraction'], fitting_df['solubility_g_g'])
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        failed_groups.append(gn)
        continue
    
    if (pcov is None or np.isnan(pcov).any() or np.isinf(pcov).any()):
        print(f"Failed to fit group {gn} due to covariance issues")
        failed_groups.append(gn)
        continue
    
    # Extract the fitted parameters
    J0, J1, J2 = popt
    
    # Calculate predicted solubility for all experimental data points
    predicted_solubility = jouyban_acree(chosen_df['solvent_1_weight_fraction'], J0, J1, J2)
    
    # Root Mean Square Error
    rmse = np.sqrt(mean_squared_error(chosen_df['solubility_g_g'], predicted_solubility))
    
    # R² score (coefficient of determination)
    r2 = r2_score(chosen_df['solubility_g_g'], predicted_solubility)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((chosen_df['solubility_g_g'] - predicted_solubility) / chosen_df['solubility_g_g'])) * 100
    
    # Store results in dictionary
    result = {
        'group_index': gn,
        'solvent_1': chosen_df['solvent_1'].iloc[0],
        'solvent_2': chosen_df['solvent_2'].iloc[0],
        'compound_id': chosen_df['compound_id'].iloc[0],
        'temperature': specific_temperature,
        'J0': J0,
        'J1': J1,
        'J2': J2,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
    }
    results.append(result)        

print(f"Processed {len(results)} groups successfully out of {len(groups)} total groups")
print(f"Failed to process {len(failed_groups)} groups")
print(f"Skipped {len(skipped_groups)} groups due to insufficient data points")

def results_describe(results):
    results_df = pd.DataFrame(results).sort_values(by='mape', ascending=False)
    # Calculate average MAPE and other statistics
    average_mape = results_df['mape'].mean()
    median_mape = results_df['mape'].median()
    min_mape = results_df['mape'].min()
    max_mape = results_df['mape'].max()

    print(f"Average MAPE: {average_mape}")
    print(f"Median MAPE: {median_mape}")
    print(f"Min MAPE: {min_mape}")
    print(f"Max MAPE: {max_mape}")

    # Print descriptive statistics for MAPE values
    print("\n--- MAPE Distribution Analysis ---")
    print(f"Count of values: {len(results_df['mape'])}")
    print(f"Number of values above 100%: {sum(results_df['mape'] > 100)}")
    print(f"Number of values above 50%: {sum(results_df['mape'] > 50)}")
    print(f"Number of values below 10%: {sum(results_df['mape'] < 10)}")
    print(f"Number of values below 5%: {sum(results_df['mape'] < 5)}")
    
    
results_describe(results)

mape_groups = pd.DataFrame(results)
mape_groups = mape_groups[mape_groups['mape'] <= 10]

# Get the groups that match the group indices in mape_groups
filtered_groups = []

for group_index in mape_groups['group_index']:
    filtered_groups.append(groups[group_index])

# Display basic information about the filtered groups
print(f"Found {len(filtered_groups)} groups with MAPE ≤ 10%")
print(f"Total number of data points: {sum(len(group) for group in filtered_groups)}")

connection = sqlite3.connect(db_path)
# Drop the table if it already exists
table_name = 'selected_solubility_data'
connection.execute(f"DROP TABLE IF EXISTS {table_name}")
# Append the filtered groups to the database in a new table
pd.concat(filtered_groups).to_sql(table_name, connection, index=False)
connection.close()