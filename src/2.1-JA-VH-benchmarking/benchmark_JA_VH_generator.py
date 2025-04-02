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


groups = [group.reset_index(drop=True) for _, group in df.groupby(['solvent_1', 'solvent_2', 'compound_id'])]


class JAVHModelEmpirical:
    groups = groups
    
    def __init__(self,VH_number=3,JA_number=3,random_seed=42):
        self.results_df = None
        self.random_seed = random_seed
        self.VH_number = VH_number
        self.JA_number = JA_number
        
    def __repr__(self):
        if self.results_df is None:
            return f"JAModelEmpirical(VH_number={self.VH_number}, JA_number={self.JA_number}, random_seed={self.random_seed})"
        else:
            return f"JAModelEmpirical(VH_number={self.VH_number}, JA_number={self.JA_number}, random_seed={self.random_seed}, median mape={self.results_df['mape'].median():.4f})"
    
    @staticmethod
    def load_from_csvs(input_folder='output'):
        loaded = []

        # Get all CSV files that match the pattern for curve fit results
        csv_files = glob.glob(f'{input_folder}/curve_fit_results_VH_number_is_*_JA_number_is_*_random_seed_is_*.csv')

        # Load each file into a dictionary with keys indicating parameters
        for file in csv_files:
            # Extract VH_number, JA_number, and random_seed from filename
            filename = os.path.basename(file)
            parts = filename.replace('curve_fit_results_VH_number_is_', '').replace('.csv', '').split('_JA_number_is_')
            
            if len(parts) == 2:
                vh_part, rest = parts
                ja_part, random_seed_part = rest.split('_random_seed_is_')
                            
                # Load the CSV file into a dataframe
                df = pd.read_csv(file)
                
                new_model = JAVHModelEmpirical(VH_number=int(vh_part), JA_number=int(ja_part), random_seed=int(random_seed_part))
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
        random.seed(self.random_seed)  # Set the random seed for reproducibility
        results = []
        failed_groups = []
        skipped_groups = []

        for gn in tqdm(range(len(groups)), desc="Processing groups"):
            chosen_df = groups[gn]
            
            solvent_2_pure_samples = chosen_df[chosen_df['solvent_1_weight_fraction'] <= 0.01][['solubility_g_g','temperature','solvent_1_weight_fraction']]
            solvent_1_pure_samples = chosen_df[chosen_df['solvent_1_weight_fraction'] >= 0.99][['solubility_g_g','temperature','solvent_1_weight_fraction']]

            if len(solvent_2_pure_samples) < self.VH_number or len(solvent_1_pure_samples) < self.VH_number:
                skipped_groups.append(gn)
                continue

            random_temp_indexes = random.sample(range(0, len(solvent_1_pure_samples)), self.VH_number)

            fitting_df_solvent_1 = solvent_1_pure_samples.iloc[random_temp_indexes].reset_index(drop=True)
            fitting_df_solvent_2 = solvent_2_pure_samples.iloc[random_temp_indexes].reset_index(drop=True)

            def vant_hoff_model(T, alpha, beta):
                return np.exp(alpha + beta/T)

            # Suppress warnings during curve fitting
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params_solvent_1, pcov_solvent_1 = curve_fit(vant_hoff_model, fitting_df_solvent_1['temperature'], fitting_df_solvent_1['solubility_g_g'])
                    params_solvent_2, pcov_solvent_2 = curve_fit(vant_hoff_model, fitting_df_solvent_2['temperature'], fitting_df_solvent_2['solubility_g_g'])
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                failed_groups.append(gn)
                continue
            
            if (pcov_solvent_1 is None or np.isnan(pcov_solvent_1).any() or np.isinf(pcov_solvent_1).any()) or (pcov_solvent_2 is None or np.isnan(pcov_solvent_2).any() or np.isinf(pcov_solvent_2).any()):
                print(f"Failed to fit group {gn} due to covariance issues")
                failed_groups.append(gn)
                continue
            
            alpha1, beta1 = params_solvent_1
            alpha2, beta2 = params_solvent_2

            # Choose a random temperature out of the ones used for fitting solvent the temperature
            # re use the 2 same values to fit to JA parameters
            chosen_temp = random.choice(fitting_df_solvent_1['temperature'].values)

            fitting_df = chosen_df[chosen_df['temperature'] == chosen_temp] \
                .sort_values(by='solvent_1_weight_fraction') \
                .reset_index(drop=True) 
            
            n = len(fitting_df)
            if n < self.JA_number+2:  # Skip groups that don't have enough points
                print(f"Skipping group {gn} due to insufficient data points")
                skipped_groups.append(gn)
                continue
            
            fitting_df = fitting_df.iloc[[0] + random.sample(range(1, len(fitting_df)-1), self.JA_number) + [-1]].reset_index(drop=True)

            def jouyban_acree_vant_hoff(independent_vars, J0, J1, J2):
                """    
                Parameters:
                independent_vars (tuple): Tuple containing (f1, T) where:
                    f1 (array-like): Volume or mole fraction of solvent 1
                    T (array-like): Absolute temperature in Kelvin
                alpha1, beta1 (float): van't Hoff constants for solvent 1
                alpha2, beta2 (float): van't Hoff constants for solvent 2
                J0, J1, J2 (float): Jouyban-Acree model parameters
                
                Returns:
                array-like: Natural logarithm of solubility in the mixed solvent
                """   
                # Unpack independent variables
                f1, T = independent_vars
                
                # Calculate fraction of second solvent
                f2 = 1 - f1
                
                # Calculate van't Hoff terms for each pure solvent
                vant_hoff_term1 = f1 * (alpha1 + beta1/T)
                vant_hoff_term2 = f2 * (alpha2 + beta2/T)
                
                # Calculate Jouyban-Acree interaction term
                interaction_term = f1 * f2 * (J0/T + J1*(f1-f2)/T + J2*(f1-f2)**2/T)
                
                # Calculate logarithm of solubility in the mixture
                ln_Xm = vant_hoff_term1 + vant_hoff_term2 + interaction_term
                
                return np.exp(ln_Xm)

            # Suppress warnings during curve fitting
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, pcov = curve_fit(jouyban_acree_vant_hoff, (fitting_df['solvent_1_weight_fraction'],fitting_df['temperature']), fitting_df['solubility_g_g'])

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
            predicted_solubility = jouyban_acree_vant_hoff((chosen_df['solvent_1_weight_fraction'],chosen_df['temperature']), J0, J1, J2)
            
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
                'J0': J0,
                'J1': J1,
                'J2': J2,
                'alpha1': alpha1,
                'beta1': beta1,
                'alpha2': alpha2,
                'beta2': beta2,
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

        self.results_df.to_csv(f'{output_folder}/curve_fit_results_VH_number_is_{self.VH_number}_JA_number_is_{self.JA_number}_random_seed_is_{self.random_seed}.csv', index=False)
        print(f"curve_fit_results_x_is_{self.VH_number}_JA_number_is_{self.JA_number}_random_seed_is_{self.random_seed}.csv")
        
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
    
    def plot_VH(self,n):
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        selected = groups[self.results_df['group_index'].iloc[n]]
        alpha1 = self.results_df['alpha1'].iloc[n]
        beta1 = self.results_df['beta1'].iloc[n]
        alpha2 = self.results_df['alpha2'].iloc[n]
        beta2 = self.results_df['beta2'].iloc[n]

        def vant_hoff_model(T, alpha, beta):
            return np.exp(alpha + beta/T)
        
        # Generate temperature values for the plot
        temperature_values = np.linspace(selected['temperature'].min(),selected['temperature'].max(),100)

        # Calculate solubility values for the plot
        solubility_solvent_1 = vant_hoff_model(temperature_values, alpha1, beta1)
        solubility_solvent_2 = vant_hoff_model(temperature_values, alpha2, beta2)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(temperature_values, solubility_solvent_1, label=f'Solvent 1: {selected["solvent_1"].iloc[0]}', color='blue')
        plt.plot(temperature_values, solubility_solvent_2, label=f'Solvent 2: {selected["solvent_2"].iloc[0]}', color='red')
        # Add the experimental data points to the plot
        pure_solvents = selected[(selected['solvent_1_weight_fraction'] >= 0.99) | (selected['solvent_1_weight_fraction'] <= 0.01)]
        plt.scatter(pure_solvents['temperature'], pure_solvents['solubility_g_g'], label='Experimental Data', zorder=5)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Solubility (g/g)')
        plt.title('Solubility vs Temperature (van\'t Hoff Model)')
        plt.legend()
        plt.grid(True)
        plt.show()
        

    def plot(self,n,temperature=298.15):
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        selected = groups[self.results_df['group_index'].iloc[n]]
        J0 = self.results_df['J0'].iloc[n]
        J1 = self.results_df['J1'].iloc[n]
        J2 = self.results_df['J2'].iloc[n]
        alpha1 = self.results_df['alpha1'].iloc[n]
        beta1 = self.results_df['beta1'].iloc[n]
        alpha2 = self.results_df['alpha2'].iloc[n]
        beta2 = self.results_df['beta2'].iloc[n]

        def jouyban_acree_vant_hoff(independent_vars, J0, J1, J2):
                """    
                Parameters:
                independent_vars (tuple): Tuple containing (f1, T) where:
                    f1 (array-like): Volume or mole fraction of solvent 1
                    T (array-like): Absolute temperature in Kelvin
                alpha1, beta1 (float): van't Hoff constants for solvent 1
                alpha2, beta2 (float): van't Hoff constants for solvent 2
                J0, J1, J2 (float): Jouyban-Acree model parameters
                
                Returns:
                array-like: Natural logarithm of solubility in the mixed solvent
                """   
                # Unpack independent variables
                f1, T = independent_vars
                
                # Calculate fraction of second solvent
                f2 = 1 - f1
                
                # Calculate van't Hoff terms for each pure solvent
                vant_hoff_term1 = f1 * (alpha1 + beta1/T)
                vant_hoff_term2 = f2 * (alpha2 + beta2/T)
                
                # Calculate Jouyban-Acree interaction term
                interaction_term = f1 * f2 * (J0/T + J1*(f1-f2)/T + J2*(f1-f2)**2/T)
                
                # Calculate logarithm of solubility in the mixture
                ln_Xm = vant_hoff_term1 + vant_hoff_term2 + interaction_term
                
                return np.exp(ln_Xm)
            
        x_values = np.linspace(0, 1, 101)

        jouyban_acree_VH_fit_values = jouyban_acree_vant_hoff((x_values,temperature), J0, J1, J2)

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, jouyban_acree_VH_fit_values, label='Extended Log-Linear Model', color='orange')
        # Add the experimental data points to the plot
        scatter = plt.scatter(selected['solvent_1_weight_fraction'], selected['solubility_g_g'], c=selected['temperature'], cmap='viridis', label='Experimental Data', zorder=5)
        plt.xlabel('Solvent 1 Weight Fraction')
        plt.ylabel('Solubility (g/g)')
        plt.title('Solubility vs Solvent 1 Weight Fraction (JA Model)')
        plt.legend()
        plt.colorbar(scatter, label='Temperature (K)', ticks=selected['temperature'].unique())  # Add a colorbar with exact temperature values
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

    # save the values for Van Hoft with data points between 3 and 12
    # save the values for JA with data points between 5 and 12
    for VH_number in range(3,10):
        for JA_number in range(3,10):
            print(f"VH_number: {VH_number}, JA_number: {JA_number}")
            model = JAVHModelEmpirical(VH_number=VH_number, JA_number=JA_number, random_seed=args.s)
            model.curve_fitter()
            model.save_results()
