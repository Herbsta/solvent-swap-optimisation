import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import os

# Set style for scientific plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper")

def print_data_summary(df, name):
    print(f"\n=== {name} Data Summary ===")
    print(f"Min solubility: {df['solubility_g_100g'].min():.2e}")
    print(f"Max solubility: {df['solubility_g_100g'].max():.2e}")
    print(f"Number of records: {len(df)}")

def create_solubility_violin_plots():
    os.makedirs('violins', exist_ok=True)
    # Connect to SQLite database
    conn = sqlite3.connect('db/MasterDatabase.db')

    # Read the solubility data
    query = """
    SELECT COALESCE(s1.molecular_name, 'Unknown') as solvent_1, 
        COALESCE(s2.molecular_name, 'Unknown') as solvent_2, 
        sol.solubility_g_g * 100 as solubility_g_100g
    FROM solubility sol
    LEFT JOIN solvents s1 ON sol.solvent_1 = s1.id
    LEFT JOIN solvents s2 ON sol.solvent_2 = s2.id
    WHERE sol.solubility_g_g IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, dtype={'solubility_g_100g': np.float64})
    print_data_summary(df, "Split by Solvent Dataset")

    # Create violin plots for solvent distribution
    df_melted = pd.melt(df, id_vars=['solubility_g_100g'], 
                       value_vars=['solvent_1', 'solvent_2'], 
                       var_name='solvent_type', value_name='solvent')
    
    # Plot for all solvents
    fig, ax = plt.subplots(figsize=(10, 6))
    df_melted['log_solubility_g_100g'] = np.log10(df_melted['solubility_g_100g'])
    df_cleaned_all = df_melted[df_melted['solvent'] != 'Unknown']
    sns.violinplot(x='solvent', y='log_solubility_g_100g', 
                  data=df_cleaned_all, ax=ax, inner='quartile', 
                  scale='width', color='lightgray')
    ax.set_yticklabels([f'$10^{{{int(tick)}}}$' for tick in ax.get_yticks()])
    ax.set_ylabel('Solubility (g/100g)', fontsize=9)
    ax.set_title('Solubility Distribution for All Solvents', fontsize=12, pad=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig('output/violins/solubility_violin_all_solvents.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot for IQR
    df_stats = df_cleaned_all.groupby('solvent')['log_solubility_g_100g'].agg(
        ['var', 'std', lambda x: np.percentile(x, 75) - np.percentile(x, 25)]).reset_index()
    df_stats.columns = ['solvent', 'variance_log_solubility', 
                       'std_log_solubility', 'iqr_log_solubility']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(y='iqr_log_solubility', data=df_stats, 
                  ax=ax, inner='quartile', scale='width', color='lightgray')
    ax.set_ylabel('IQR of Log10 Solubility (g/100g)', fontsize=9)
    ax.set_title('IQR of Solubility Distribution for All Solvents', fontsize=12, pad=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig('output/violins/solubility_iqr_violin_all_solvents.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot for solvent weight fraction
    query_weight_fraction = """
    SELECT solvent_1_weight_fraction * 100 as solvent_weight_fraction
    FROM solubility
    """
    df_weight = pd.read_sql_query(query_weight_fraction, conn, 
                                 dtype={'solvent_weight_fraction': np.float64})
    
    fig, ax = plt.subplots(figsize=(4, 5))
    sns.violinplot(y=df_weight['solvent_weight_fraction'], ax=ax, 
                  color='lightgray', inner='quartile')
    plt.ylim(0,100)
    plt.title('Distribution of Solvent Weight Fraction', fontsize=10, pad=10)
    plt.ylabel('Solvent Weight Fraction (%)', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig('output/violins/solvent_weight_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot for temperature distribution
    query_temperature = """
    SELECT temperature
    FROM solubility
    WHERE temperature IS NOT NULL
    """
    df_temperature = pd.read_sql_query(query_temperature, conn, 
                                     dtype={'temperature': np.float64})
    
    fig, ax = plt.subplots(figsize=(4, 5))
    sns.violinplot(y=df_temperature['temperature'], ax=ax, 
                  inner='quartile', color='lightgray')
    plt.ylim(250, 400)
    plt.title('Distribution of Temperature Values', fontsize=10, pad=10)
    plt.ylabel('Temperature', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig('output/violins/temperature_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

    conn.close()

if __name__ == "__main__":
    create_solubility_violin_plots()