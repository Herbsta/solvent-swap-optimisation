from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sqlite3

def domain_split(data, random_state=42):
    """
    Create domain-specific splits for solubility_g_g prediction
    
    Parameters:
    -----------
    data: DataFrame with columns ['compound_id', 'solvent_1', 'solvent_2', 'temperature', 'solvent_1_weight_fraction']
    test_size: fraction of data to use for testing
    random_state: random seed for reproducibility
    """
    
    generic_train, generic_test = train_test_split(
        data, 
        test_size=0.10,
        random_state=random_state
    )
    
    # 1. Split for new API prediction
    unique_apis = data['compound_id'].unique()
    api_train, api_test = train_test_split(
        unique_apis, 
        test_size=0.10,
        random_state=random_state
    )
    
    # Create API-based splits
    api_test_mask = data['compound_id'].isin(api_test)
    test_new_apis = data[api_test_mask]
    train_known_apis = data[~api_test_mask]
    
    # 2. Split for new solvent combinations
    unique_solv_combos = data.groupby(['solvent_1', 'solvent_2']).size().reset_index()
    train_combos, test_combos = train_test_split(
        unique_solv_combos,
        test_size=0.10,
        random_state=random_state
    )
    
    # Create masks for solvent combination splits
    test_combo_mask = data.apply(
        lambda x: any((
            (test_combos['solvent_1'] == x['solvent_1']) & 
            (test_combos['solvent_2'] == x['solvent_2'])
        ).values),
        axis=1
    )
    
    test_new_combinations = data[test_combo_mask & ~api_test_mask]
    train_final = generic_train[~test_combo_mask & ~api_test_mask]
    
    return {
        'train': train_final,
        'test_generic': generic_test,
        'test_new_apis': test_new_apis,
        'test_new_combinations': test_new_combinations
    }

def evaluate_domain_splits(splits):
    """Print statistics about the domain splits"""
    for split_name, split_data in splits.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"Number of samples: {len(split_data)}")
        print(f"Unique APIs: {split_data['compound_id'].nunique()}")
        print(f"Unique solvent_1: {split_data['solvent_1'].nunique()}")
        print(f"Unique solvent_2: {split_data['solvent_2'].nunique()}")
        print(f"Solvent combinations: {len(split_data.groupby(['solvent_1', 'solvent_2']))}")

# Connect to SQLite database
conn = sqlite3.connect('db/MasterDatabase.db')
    
df = pd.read_sql_query("""
SELECT *
FROM solubility
""", conn)
conn.close()

        
# Example usage
splits = domain_split(df)
evaluate_domain_splits(splits)

# For model training
X_train = splits['train'][['solvent_1', 'solvent_2', 'temperature', 'solvent_1_weight_fraction']]
y_train = splits['train']['solubility_g_g']

# For evaluating on new APIs
X_test_apis = splits['test_new_apis'][['solvent_1', 'solvent_2', 'temperature', 'solvent_1_weight_fraction']]
y_test_apis = splits['test_new_apis']['solubility_g_g']

# For evaluating on new solvent combinations
X_test_combinations = splits['test_new_combinations'][['solvent_1', 'solvent_2', 'temperature', 'solvent_1_weight_fraction']]
y_test_combinations = splits['test_new_combinations']['solubility_g_g']