import sqlite3
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import warnings
import os
import glob
import argparse
import scipy.stats as stats


current_folder = os.path.dirname(os.path.abspath(__file__))
connection = sqlite3.connect(f'{current_folder}/../../db/MasterDatabase.db')
# Execute the query and load the data into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM selected_solubility_data", connection)

# Close the database connection
connection.close()


groups = [group.reset_index(drop=True) for _, group in df.groupby(['solvent_1', 'solvent_2', 'compound_id','temperature'])]

class JAModelEmpirical:
    groups = groups
    
    def __init__(self,x,random_seed=42):
        self.results_df = None
        self.random_seed = random_seed
        self.x = x
        
    def __repr__(self):
        if self.results_df is None:
            return f"JAModelEmpirical(x={self.x}, random_seed={self.random_seed})"
        else:
            return f"JAModelEmpirical(x={self.x}, random_seed={self.random_seed}, median mape={self.results_df['mape'].median():.4f})"
    
    @staticmethod
    def load_from_csvs(input_folder='output'):
        loaded = []

        # Get all CSV files that match the pattern for curve fit results
        csv_files = glob.glob(f'{input_folder}/curve_fit_results_x_is_*_random_seed_is_*.csv')

        # Load each file into a dictionary with keys indicating parameters
        for file in csv_files:
            # Extract x and random_seed from filename
            filename = os.path.basename(file)
            parts = filename.replace('curve_fit_results_x_is_', '').replace('.csv', '').split('_random_seed_is_')
            
            if len(parts) == 2:
                x_value, random_seed = parts                
                # Load the CSV file into a dataframe
                df = pd.read_csv(file)
                
                new_model = JAModelEmpirical(int(x_value),int(random_seed))
                new_model.results_df = df
                loaded.append(new_model)
        return loaded
    
    def desc_by_mape(self):
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        sorted_results = self.results_df.sort_values(by='mape', ascending=False)
        return sorted_results.reset_index(drop=True)
        
    
    def results_describe(self):
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        # Calculate average MAPE and other statistics
        average_mape = self.results_df['mape'].mean()
        median_mape = self.results_df['mape'].median()
        min_mape = self.results_df['mape'].min()
        max_mape = self.results_df['mape'].max()

        print(f"Average MAPE: {average_mape}")
        print(f"Median MAPE: {median_mape}")
        print(f"Min MAPE: {min_mape}")
        print(f"Max MAPE: {max_mape}")

        # Print descriptive statistics for MAPE values
        print("\n--- MAPE Distribution Analysis ---")
        print(f"Count of values: {len(self.results_df['mape'])}")
        print(f"Number of values above 100%: {sum(self.results_df['mape'] > 100)}")
        print(f"Number of values above 50%: {sum(self.results_df['mape'] > 50)}")
        print(f"Number of values below 10%: {sum(self.results_df['mape'] < 10)}")
        print(f"Number of values below 5%: {sum(self.results_df['mape'] < 5)}")
    
    def curve_fitter(self):
        '''
        where x is the number of random points to select from each group, excluding the first and last points.
        The first and last points are always included in the fitting.
        '''
        random.seed(self.random_seed)  # Set the random seed for reproducibility
        results = []
        failed_groups = []
        skipped_groups = []

        for gn in tqdm(range(len(self.groups)), desc="Processing groups"):
            chosen_df = self.groups[gn]
                
            n = len(chosen_df)
            if n < self.x+2:  # Skip groups that don't have enough points
                print(f"Skipping group {gn} due to insufficient data points")
                skipped_groups.append(gn)
                continue
                
            random_indices = random.sample(range(1, n-1), self.x)
            
            solvent_2_pure = chosen_df[chosen_df['solvent_1_weight_fraction'] <= 0.01].iloc[0]['solubility_g_g']
            solvent_1_pure = chosen_df[chosen_df['solvent_1_weight_fraction'] >= 0.99].iloc[0]['solubility_g_g']
            specific_temperature = chosen_df['temperature'].iloc[0]
            
            # Create the random dataframe with x rows
            fitting_df = chosen_df.iloc[[0] + random_indices + [n-1]].reset_index(drop=True)
            
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
                'solvent_1_pure': solvent_1_pure,
                'solvent_2_pure': solvent_2_pure,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'logmape': np.log10(mape) if mape > 0 else np.inf,
            }
            results.append(result)  
        
        self.results_df = pd.DataFrame(results)
    
    def save_results(self,output_folder='output'):
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.results_df.to_csv(f'{output_folder}/curve_fit_results_x_is_{self.x}_random_seed_is_{self.random_seed}.csv', index=False)
        print(f"curve_fit_results_x_is_{self.x}_random_seed_is_{self.random_seed}.csv")
        
    def plot_log_mape(self):
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        # Apply log transformation to the MAPE values
        self.results_df['log_mape'] = np.log10(self.results_df['mape'])

        # Create histogram
        plt.figure(figsize=(12, 6))
        plt.hist(self.results_df['log_mape'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('log10(MAPE)')
        plt.ylabel('Frequency')
        plt.title('Histogram of log10(MAPE) Values')
        plt.grid(True, alpha=0.3)

        # Add vertical lines for reference points
        plt.axvline(x=np.log10(1), color='green', linestyle='--', label='MAPE = 1%')
        plt.axvline(x=np.log10(10), color='orange', linestyle='--', label='MAPE = 10%')
        plt.axvline(x=np.log10(100), color='red', linestyle='--', label='MAPE = 100%')

        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print some statistics about the log-transformed MAPE
        print(f"Log10(MAPE) statistics:")
        print(f"Mean: {np.mean(self.results_df['log_mape']):.4f}")
        print(f"Median: {np.median(self.results_df['log_mape']):.4f}")
        print(f"Min: {np.min(self.results_df['log_mape']):.4f}")
        print(f"Max: {np.max(self.results_df['log_mape']):.4f}")
    
    def plot(self,n):
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        selected = groups[self.results_df['group_index'].iloc[n]]
        J0 = self.results_df['J0'].iloc[n]
        J1 = self.results_df['J1'].iloc[n]
        J2 = self.results_df['J2'].iloc[n]
        solvent_1_pure = self.results_df['solvent_1_pure'].iloc[n]
        solvent_2_pure = self.results_df['solvent_2_pure'].iloc[n]
        specific_temperature = self.results_df['temperature'].iloc[n]

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
            
        x_values = np.linspace(0, 1, 101)

        jouyban_acree_fit_values = jouyban_acree(x_values, J0, J1, J2)

        # Plot the JA model
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, jouyban_acree_fit_values, label='Jouyban-Acree Model', color='blue')

        # Add the experimental data points to the plot
        plt.scatter(selected['solvent_1_weight_fraction'], selected['solubility_g_g'], color='red', label='Experimental Data', zorder=5)
        plt.xlabel('Solvent 1 Weight Fraction')
        plt.ylabel('Solubility (g/g)')
        plt.title('Solubility vs Solvent 1 Weight Fraction (JA Model)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def paired_t_test(self, other_model):
        if self.results_df is None or other_model.results_df is None:
            raise ValueError("Both models must have results to perform the paired t-test.")
        
        # Merge the two dataframes on the group index
        merged_df = pd.merge(self.results_df, other_model.results_df, on='group_index', suffixes=('_model1', '_model2'), how='inner')
        
        # Perform paired t-test on logmape values
        t_statistic, p_value = stats.ttest_rel(merged_df['logmape_model1'], merged_df['logmape_model2'], alternative='less')
                
        print("\nPaired t-test results:")
        print(f"t-statistic: {t_statistic:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.025:
            print(f"There is a statistically significant difference with {self} having lower logmape values (p < {0.025}).")
        else:
            print(f"There is no statistically significant evidence that {self} has lower logmape values (p >= {0.025}).")

        # Visualize the differences
        plt.figure(figsize=(8, 4))

        # Histogram of differences
        plt.subplot(1, 2, 1)
        
        merged_df['diff'] = merged_df['logmape_model1'] - merged_df['logmape_model2']
        
        plt.hist(merged_df['diff'], bins=30, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('Difference in logmape')
        plt.ylabel('Frequency')
        plt.title('Histogram of Differences')

        # Scatter plot comparing the two sets
        plt.subplot(1, 2, 2)
        plt.scatter(merged_df['logmape_model1'], merged_df['logmape_model2'], alpha=0.5)
        plt.plot([-15, 5], [-15, 5], 'r--')  # Line y=x for reference
        plt.xlabel(f'{self}')
        plt.ylabel(f'{other_model}')
        plt.title('Comparison of logmape Values')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return t_statistic, p_value
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JAModelEmpirical with a specified random seed.")
    parser.add_argument("-s", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # save data for values of data point numbers 5 to 12
    
    for x in range(3,10):
        model = JAModelEmpirical(x, random_seed=args.s)
        model.curve_fitter()
        model.save_results()