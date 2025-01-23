import pandas as pd
import re
import numpy as np
import sqlite3
import glob
import os

def extract_solubility_ratios(comment):
    if pd.isna(comment):
       return None, None
    # Mass ratio pattern
    pattern1 = r'(\d+\.?\d*)\s*g\s*solvent\s*dissolves\.\s*(\d+\.?\d*)\s*g\s*Substance\.'
    match1 = re.search(pattern1, comment)
   
    mass_ratio = None
    if match1:
       solvent_g = float(match1.group(1))
       solute_g = float(match1.group(2))
       mass_ratio = solute_g / solvent_g
       
    # Mole ratio pattern (mol/1000mol)
    pattern2 = r'Solubility:\s*(\d+\.?\d*E?-?\d*)\s*mol/1000mol'
    match2 = re.search(pattern2, comment)
    mol_ratio = None
    if match2:
       mol_ratio = float(match2.group(1)) / 1000
       
    # Weight percent pattern
    pattern3 = r'(?:Solubility|Solubilty):\s*(\d+\.?\d*E?-?\d*)\s*(?:weight\s*percent|weight-percent)'
    match3 = re.search(pattern3, comment)
    if match3 and mass_ratio is None:
       mass_ratio = float(match3.group(1)) / 100
       
    # Mol percent patterns
    pattern4 = r'(?:Solubility|Solubilty):\s*(\d+\.?\d*E?-?\d*)\s*(?:mol-?percent|molpercent)'
    match4 = re.search(pattern4, comment)
    if match4 and mol_ratio is None:
       mol_ratio = float(match4.group(1)) / 100

    # Gram per gram pattern
    pattern4 = r'Solubility:\s*(\d+\.?\d*E?-?\d*)\s*p\(g/100g\s*(?:solvent|solution)\)'
    match4 = re.search(pattern4, comment)
    if match4 and mass_ratio is None:
        mass_ratio = float(match4.group(1)) / 100
    
    pattern5 = r'(?:Solubility|solubilty):\s*(\d+\.?\d*E?-?\d*)\s*g/100g\s*solvent'
    match5 = re.search(pattern5, comment)
    if match5 and mass_ratio is None:
        mass_ratio = float(match5.group(1)) / 100
    
    pattern6 = r'(?:Solubility|solubilty):\s*(\d+\.?\d*E?-?\d*)\s*mol/mol'
    match6 = re.search(pattern6, comment)
    if match6 and mol_ratio is None:
        mol_ratio = float(match6.group(1))
       
    return mass_ratio, mol_ratio

def extract_solvent_ratios(ratio_comment):
    if pd.isna(ratio_comment):
        return None, None, None
    
    def calculate_ratio(first_number, second_number):
        """Helper function to calculate ratio from two numbers"""
        return float(first_number) / (float(first_number) + float(second_number))

    def extract_with_pattern(pattern, text, find_all=False):
        """Helper function to extract and calculate ratios using regex patterns"""
        if find_all:
            matches = re.findall(pattern, text)
            if len(matches) == 2:  # For patterns that find individual numbers
                return calculate_ratio(matches[0], matches[1])
            elif len(matches) == 1 and isinstance(matches[0], tuple):  # For patterns that capture groups
                return calculate_ratio(matches[0][0], matches[0][1])
        else:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    return calculate_ratio(match.group(1), match.group(2))
                elif len(match.groups()) == 4:  # For mol percent with compound names
                    return calculate_ratio(match.group(2), match.group(4))
        return None

    # Dictionary of patterns for different ratio types
    patterns = {
        'volume': [
            (r"(\d+):(\d+)\s*v/v", False),
            (r'\((\d+)\s*vol\s*percent\)', True)
        ],
        'weight': [
            (r"(\d+\.\d+):(\d+\.\d+)\s*w/w", False),
            (r"(\d+\.?\d*):(\d+\.?\d*)\s*wtpercent", False),
            (r"(\d+):(\d+)\s*percentWt", False),
            (r'\((\d*\.?\d+)\s*g\)', True),
            (r'\((\d*\.?\d+)\s*weight percent\)', True),
            (r'(\d+)\s*wt\s*percent', True),
            (r"(\d+):(\d+)\s*wt", False),
            (r"(\d+\.\d+):(\d+\.\d+)\s*\(w/w\)", False)
        ],
        'mole': [
            (r"(\w+)\s*\((\d+\.\d+)\s*molpercent\);\s*(\w+)\s*\((\d+\.\d+)\s*molpercent\)", False),
            (r'\((\d*\.?\d+)\s*mol\)', True),
            (r'\((\d+)\s*molpercent\)', True),
            (r'(\d+\.\d+)\/(\d+\.\d+) mol\/mol', False)
        ]
    }

    # Special case for single mol percent
    single_mol_pattern = r"(\w+)\s*\((\d+\.\d+)\s*molpercent\)"
    single_mol_match = re.search(single_mol_pattern, ratio_comment)
    if single_mol_match:
        first_number = float(single_mol_match.group(2))
        mole_ratio = calculate_ratio(first_number, 100 - first_number)
    else:
        mole_ratio = None

    # Special case for single weight percent
    single_weight_pattern = r"\((\d+)\sweight\spercent\)"
    single_weight_match = re.search(single_weight_pattern, ratio_comment)
    if single_weight_match:
        first_number = float(single_weight_match.group(1))
        weight_ratio = calculate_ratio(first_number, 100 - first_number)
    else:
        weight_ratio = None
    
    # Special case for mole per dm3
    mole_dm3_ratio = None
    mole_dm3_pattern = r"(\d*\.?\d+)\s*mol\s*dm-3"
    mole_dm3_match = re.search(mole_dm3_pattern, ratio_comment)
    if mole_dm3_match:
        mole_dm3_ratio = float(mole_dm3_match.group(1))

    mole_dm3_patterns = [
        r"(\d+)\s*mol\s*:\s*(\d+)\s*l",
        r"(\w+)\s*\((\d+\.?\d*)\s*mol\);\s*water\s*\(1\s*l\)"
    ]
    for pattern in mole_dm3_patterns:
        match = re.search(pattern, ratio_comment)
        if match:
            mole_dm3_ratio = float(match.group(2))
            break

    # Special case for HMPT concentration
    hmpt_pattern = r"x\(HMPT\)\s*=\s*(\d+\.\d+)"
    hmpt_match = re.search(hmpt_pattern, ratio_comment)
    if hmpt_match:
        mole_ratio = float(hmpt_match.group(1))

    # Try all patterns for each ratio type
    vol_ratio = None
    for pattern, find_all in patterns['volume']:
        vol_ratio = extract_with_pattern(pattern, ratio_comment, find_all)
        if vol_ratio is not None:
            break

    if weight_ratio is None:  # Only try if special case didn't work
        for pattern, find_all in patterns['weight']:
            weight_ratio = extract_with_pattern(pattern, ratio_comment, find_all)
            if weight_ratio is not None:
                break

    if mole_ratio is None:  # Only try if special case didn't work
        for pattern, find_all in patterns['mole']:
            mole_ratio = extract_with_pattern(pattern, ratio_comment, find_all)
            if mole_ratio is not None:
                break

    return weight_ratio, mole_ratio, vol_ratio, mole_dm3_ratio

def process_data(df):
    solubility_g_g, solubility_mol_mol = zip(*df['Comment (Solubility (MCS))'].apply(extract_solubility_ratios))
    df.insert(1, 'solubility_g_g', solubility_g_g)
    df.insert(2, 'solubility_mol_mol', solubility_mol_mol)
    return df


def process_csv_files(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, encoding='cp1252')
        df.columns = [col.replace('¡¤', '/').replace('¡ã', '°') for col in df.columns]
        df.insert(0,'pub_chem_id',os.path.splitext(os.path.basename(file))[0])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def create_databases(combined_df):
    # Convert to Kelvin
    combined_df.insert(5, 'temperature_k', combined_df['Temperature (Solubility (MCS)), °C'].apply(lambda x: float(x) + 273.15 if isinstance(x, (int, float)) else np.nan))
    # Drop Celsius column
    combined_df = combined_df.drop(columns=['Temperature (Solubility (MCS)), °C'])
    combined_df = combined_df.drop(columns=['Location'])
    # Drop rows where Temperature_K is NaN
    combined_df.dropna(subset=['temperature_k'],inplace=True)



    #Extract from the Comment Column
    combined_df = process_data(combined_df)

    def set_target_to_null(row):
        if pd.notnull(row['solubility_g_g']) or pd.notnull(row['solubility_mol_mol']) or pd.notnull(row['Solubility, g/l-1']):
            row['found'] = np.nan
        return row

    # Apply the custom function to each row
    combined_df['found'] = 1
    combined_df = combined_df.apply(set_target_to_null, axis=1)

    # Cull values where the units could not be standardised
    combined_df = combined_df[combined_df['found'].isnull()]
    combined_df.dropna(axis=1,how='all',inplace=True)

    #rename to ensure ease of sql compatability
    combined_df = combined_df.rename(columns={
        'Saturation': 'saturation',
        'Solubility, g/l-1': 'solubility_g_l',
        'Ratio of Solvents': 'ratio_of_solvents',
        'Reference': 'reference',
        'Solvent (Solubility (MCS))': 'solvent',
        'Comment (Solubility (MCS))': 'comment',
    })

    # drop rows that say various solvents
    combined_df = combined_df[~combined_df['solvent'].str.contains('various solvent', na=False)]

    # Identify single and binary solvent systems
    binary_mask = combined_df['ratio_of_solvents'].notna() & (combined_df['ratio_of_solvents'] != '') # FIX THE RATIO OF SOLVENTS CASE!!!
    # combined_df['ratio_of_solvents'].notna() & (combined_df['ratio_of_solvents'] != '') & ~combined_df['comment'].str.contains('ratio of solvents', case=False, na=False)
    
    # Split data
    single_solvent_df = combined_df[~binary_mask]

    binary_solvent_df = combined_df[binary_mask]
    
    # Create databases
    # Single solvent system

    
    essential_columns = ['solubility_g_l', 'saturation', 'temperature_k',
       'solvent']

    single_solvent_df = single_solvent_df.dropna(subset=essential_columns, how='all')    
    single_solvent_df = single_solvent_df.drop(columns=['ratio_of_solvents'])
    conn_single = sqlite3.connect('db/pubchemSolubilityDatabase.db')
    single_solvent_df.to_sql('single_solvent_data', conn_single, if_exists='replace', index=False)
    conn_single.close()
    
    # Binary solvent system
    # Separate out the 2 solvents
    mask = binary_solvent_df['solvent'].str.contains(', ')
    binary_solvent_df = binary_solvent_df[mask].copy()

    df = binary_solvent_df['solvent'].str.split(', ',n=1, expand=True)
    binary_solvent_df.insert(1,'solvent_1',df[0])
    binary_solvent_df.insert(2,'solvent_2',df[1])
    binary_solvent_df.drop('solvent', axis=1,inplace=True)


    # Extract the solvent fractions
    solvent_1_weight_fraction, solvent_1_mol_fraction,solvent_1_vol_fraction, solvent_2_mol_dm3 = zip(*binary_solvent_df['ratio_of_solvents'].apply(extract_solvent_ratios))

    binary_solvent_df.insert(2,'solvent_1_weight_fraction',solvent_1_weight_fraction)
    binary_solvent_df.insert(3,'solvent_1_mol_fraction',solvent_1_mol_fraction)
    binary_solvent_df.insert(4,'solvent_1_vol_fraction',solvent_1_vol_fraction)
    binary_solvent_df.insert(5,'solvent_2_mol_dm3',solvent_2_mol_dm3)

    def set_target_to_null(row):
        if pd.notnull(row['solvent_1_weight_fraction']) or pd.notnull(row['solvent_1_mol_fraction']) or pd.notnull(row['solvent_1_vol_fraction']) or pd.notnull(row['solvent_2_mol_dm3']):
            row['found'] = np.nan
        return row

    # Apply the custom function to each row
    binary_solvent_df['found'] = 1
    binary_solvent_df = binary_solvent_df.apply(set_target_to_null, axis=1)

    # Cull values where the units could not be standardised
    binary_solvent_df = binary_solvent_df[binary_solvent_df['found'].isnull()]
    binary_solvent_df.dropna(axis=1,how='all',inplace=True)


    conn_binary = sqlite3.connect('db/pubchemSolubilityDatabase.db')
    binary_solvent_df.to_sql('dual_solvent_data', conn_binary, if_exists='replace', index=False)
    conn_binary.close()
    
    return len(single_solvent_df), len(binary_solvent_df)

def main():
    # Get all CSV files
    folder_path = "assets/pubchemIDFiles"
    csv_files = glob.glob(os.path.join(folder_path,'*.csv'))
    
    # Process all CSV files
    combined_df = process_csv_files(csv_files)
    
    # Create databases and get counts
    single_count, binary_count = create_databases(combined_df)
    
    print(f"Processed {len(csv_files)} CSV files")
    print(f"Created single solvent database with {single_count} records")
    print(f"Created binary solvent database with {binary_count} records")

if __name__ == "__main__":
    main()