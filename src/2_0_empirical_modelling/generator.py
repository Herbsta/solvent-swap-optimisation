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

random.seed(123)

# Database connection and data loading
current_folder = os.path.dirname(os.path.abspath(__file__))
connection = sqlite3.connect(f'{current_folder}/../../db/MasterDatabase.db')
df = pd.read_sql_query("SELECT * FROM selected_solubility_data", connection)
connection.close()

# Groups for JA model (grouped by temperature)
ja_groups = [group.reset_index(drop=True) for _, group in df.groupby(['solvent_1', 'solvent_2', 'compound_id', 'temperature'])]

# Groups for JAVH model (no temperature grouping)
javh_groups = [group.reset_index(drop=True) for _, group in df.groupby(['solvent_1', 'solvent_2', 'compound_id'])]

def uniform_sample_indices(n, k):
    """
    Select k indices with approximately uniform spacing from a range of n points.
    
    Parameters:
    n (int): Total number of points
    k (int): Number of points to sample
    
    Returns:
    list: Sorted list of selected indices
    """
    if k >= n:
        return list(range(n))
    
    # Calculate equal spacing between points
    indices = [int(i * (n-1) / (k-1)) for i in range(k)] if k > 1 else [0]
    return indices

#----------------------------
# Base Model Class
#----------------------------
class BaseModelEmpirical:
    """Base class for empirical solubility models."""
    
    def __init__(self):
        self.results_df = None
    
    def results_describe(self):
        """Print descriptive statistics for the model results."""
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
    
    def desc_by_mape(self):
        """Return results sorted by MAPE in descending order."""
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        sorted_results = self.results_df.sort_values(by='mape', ascending=False)
        return sorted_results.reset_index(drop=True)
    
    def plot_log_mape(self):
        """Plot histogram of log10(MAPE) values."""
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        # Apply log transformation to the MAPE values
        self.results_df['log_mape'] = np.log10(self.results_df['mape'])

        # Create histogram
        plt.figure(figsize=(16, 9))
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
    
    def paired_t_test(self, other_model, verbose=True):
        """Perform paired t-test between this model and another model.
        
        Parameters:
        other_model (BaseModelEmpirical): Another model to compare with
        verbose (bool): If True, prints detailed output and shows plots. If False, returns results quietly.
        
        Returns:
        tuple: (t_statistic, p_value)
        """
        if self.results_df is None or other_model.results_df is None:
            raise ValueError("Both models must have results to perform the paired t-test.")
        
        # Merge the two dataframes on the group index
        merged_df = pd.merge(self.results_df, other_model.results_df, 
                             on='group_index', suffixes=('_model1', '_model2'), how='inner')
        
        # Perform paired t-test on logmape values
        t_statistic, p_value = stats.ttest_rel(merged_df['logmape_model1'], 
                                              merged_df['logmape_model2'], 
                                              alternative='less')
        
        if verbose:
            print("\nPaired t-test results:")
            print(f"t-statistic: {t_statistic:.4f}")
            print(f"p-value: {p_value:.4f}")
            
            if p_value < 0.025:
                print(f"There is a statistically significant difference with {self} having lower logmape values (p < {0.025}).")
            else:
                print(f"There is no statistically significant evidence that {self} has lower logmape values (p >= {0.025}).")

            # Calculate additional statistics for verbose mode
            diff_mean = merged_df['logmape_model1'].mean() - merged_df['logmape_model2'].mean()
            num_better = sum(merged_df['logmape_model1'] < merged_df['logmape_model2'])
            total_cases = len(merged_df)
            percentage_better = (num_better / total_cases) * 100
            
            print(f"\nAdditional statistics:")
            print(f"Mean difference in logmape: {diff_mean:.4f}")
            print(f"Cases where {self} performs better: {num_better} out of {total_cases} ({percentage_better:.1f}%)")

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

        return t_statistic, p_value, merged_df['mape_model1'] - merged_df['mape_model2']
    
    def curve_fitter(self):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_results(self, output_folder='output'):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def plot(self, n):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


#----------------------------
# Van't Hoff Model
#----------------------------
class VanHoffModel:
    """Model for temperature-dependent solubility based on van't Hoff equation."""
    
    @staticmethod
    def vant_hoff_equation(T, alpha, beta):
        """Calculate solubility using van't Hoff equation.
        
        Parameters:
        T (float or array): Temperature in Kelvin
        alpha (float): First parameter of the van't Hoff equation
        beta (float): Second parameter of the van't Hoff equation
        
        Returns:
        float or array: Solubility calculated by the van't Hoff equation
        """
        return np.exp(alpha + beta/T)
    
    def fit(self, temperatures, solubilities):
        """Fit the van't Hoff equation to experimental data.
        
        Parameters:
        temperatures (array-like): Temperature values in Kelvin
        solubilities (array-like): Corresponding solubility values
        
        Returns:
        tuple: (params, pcov) where params are (alpha, beta) and pcov is the covariance matrix
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, pcov = curve_fit(self.vant_hoff_equation, temperatures, solubilities)
        
        return params, pcov
    
    def predict(self, temperatures, alpha, beta):
        """Predict solubility at given temperatures.
        
        Parameters:
        temperatures (array-like): Temperature values in Kelvin
        alpha (float): First parameter of the van't Hoff equation
        beta (float): Second parameter of the van't Hoff equation
        
        Returns:
        array-like: Predicted solubility values
        """
        return self.vant_hoff_equation(temperatures, alpha, beta)


#----------------------------
# Jouyban-Acree Model
#----------------------------
class JouybanAcreeModel:
    """Model for mixed solvent solubility based on Jouyban-Acree equation."""
    
    @staticmethod
    def jouyban_acree_equation(f1, solubility_1, solubility_2, temp, J0, J1, J2):
        """Calculate solubility using the Jouyban-Acree model.
        
        Parameters:
        f1 (float or array): Volume or weight fraction of solvent 1
        solubility_1 (float): Solubility in pure solvent 1
        solubility_2 (float): Solubility in pure solvent 2
        temp (float): Temperature in Kelvin
        J0, J1, J2 (float): Jouyban-Acree model parameters
        
        Returns:
        float or array: Predicted solubility in the mixed solvent
        """
        f2 = 1 - f1
        
        # Calculate logarithm of solubility in the mixture
        # Modified interaction term that reduces likelihood of bimodal behavior
        interaction_term = J0 * f1 * f2 + J1 * f1 * f2 * (2*f1 - 1) + J2 * f1 * f2 * (2*f1 - 1)**2
        
        log_Cm = f1 * np.log(solubility_1) + f2 * np.log(solubility_2) + interaction_term / temp
        
        return np.exp(log_Cm)
    
    def fit(self, fractions, solubilities, solubility_1, solubility_2, temp):
        """Fit the Jouyban-Acree model to mixed solvent data.
        
        Parameters:
        fractions (array-like): Solvent 1 weight fractions
        solubilities (array-like): Corresponding solubility values
        solubility_1 (float): Solubility in pure solvent 1
        solubility_2 (float): Solubility in pure solvent 2
        temp (float): Temperature in Kelvin
        
        Returns:
        tuple: (params, pcov) where params are (J0, J1, J2) and pcov is the covariance matrix
        """
        def fit_function(f1, J0, J1, J2):
            return self.jouyban_acree_equation(f1, solubility_1, solubility_2, temp, J0, J1, J2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, pcov = curve_fit(fit_function, fractions, solubilities)
        
        return params, pcov
    
    def predict(self, fractions, solubility_1, solubility_2, temp, J0, J1, J2):
        """Predict solubility at given solvent fractions.
        
        Parameters:
        fractions (array-like): Solvent 1 weight fractions
        solubility_1 (float): Solubility in pure solvent 1
        solubility_2 (float): Solubility in pure solvent 2
        temp (float): Temperature in Kelvin
        J0, J1, J2 (float): Jouyban-Acree model parameters
        
        Returns:
        array-like: Predicted solubility values
        """
        if np.isscalar(fractions):
            return self.jouyban_acree_equation(fractions, solubility_1, solubility_2, temp, J0, J1, J2)
        else:
            return np.array([
                self.jouyban_acree_equation(f, solubility_1, solubility_2, temp, J0, J1, J2) 
                for f in fractions
            ])


#----------------------------
# Jouyban-Acree-Van't Hoff Model
#----------------------------
class JouybanAcreeVanHoffModel:
    """Combined model for mixed solvent solubility with temperature dependence."""
    
    @staticmethod
    def jouyban_acree_vant_hoff_equation(f1, temp, alpha1, beta1, alpha2, beta2, J0, J1, J2):
        """Calculate solubility using the combined Jouyban-Acree and van't Hoff model.
        
        Parameters:
        f1 (float): Volume or weight fraction of solvent 1
        temp (float): Temperature in Kelvin
        alpha1, beta1 (float): van't Hoff parameters for solvent 1
        alpha2, beta2 (float): van't Hoff parameters for solvent 2
        J0, J1, J2 (float): Jouyban-Acree model parameters
        
        Returns:
        float: Predicted solubility in the mixed solvent
        """
        f2 = 1 - f1
        
        # Calculate van't Hoff terms for each pure solvent
        vant_hoff_term1 = f1 * (alpha1 + beta1/temp)
        vant_hoff_term2 = f2 * (alpha2 + beta2/temp)
        
        # Calculate Jouyban-Acree interaction term
        interaction_term = f1 * f2 * (J0/temp + J1*(f1-f2)/temp + J2*(f1-f2)**2/temp)
        
        # Calculate logarithm of solubility in the mixture
        ln_Xm = vant_hoff_term1 + vant_hoff_term2 + interaction_term
        
        return np.exp(ln_Xm)
    
    def fit(self, fractions, temps, solubilities, alpha1, beta1, alpha2, beta2):
        """Fit the Jouyban-Acree-Van't Hoff model to mixed solvent data.
        
        Parameters:
        fractions (array-like): Solvent 1 weight fractions
        temps (array-like): Temperature values in Kelvin
        solubilities (array-like): Corresponding solubility values
        alpha1, beta1 (float): van't Hoff parameters for solvent 1
        alpha2, beta2 (float): van't Hoff parameters for solvent 2
        
        Returns:
        tuple: (params, pcov) where params are (J0, J1, J2) and pcov is the covariance matrix
        """
        def fit_function(independent_vars, J0, J1, J2):
            f1, T = independent_vars
            return self.jouyban_acree_vant_hoff_equation(f1, T, alpha1, beta1, alpha2, beta2, J0, J1, J2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, pcov = curve_fit(fit_function, (fractions, temps), solubilities)
        
        return params, pcov
    
    def predict(self, fractions, temps, alpha1, beta1, alpha2, beta2, J0, J1, J2):
        """Predict solubility at given conditions.
        
        Parameters:
        fractions (array-like): Solvent 1 weight fractions
        temps (float or array-like): Temperature values in Kelvin
        alpha1, beta1 (float): van't Hoff parameters for solvent 1
        alpha2, beta2 (float): van't Hoff parameters for solvent 2
        J0, J1, J2 (float): Jouyban-Acree model parameters
        
        Returns:
        array-like: Predicted solubility values
        """
        if isinstance(temps, (int, float)):
            temps = np.ones_like(fractions) * temps
        
        return np.array([
            self.jouyban_acree_vant_hoff_equation(f, t, alpha1, beta1, alpha2, beta2, J0, J1, J2)
            for f, t in zip(fractions, temps)
        ])


#----------------------------
# JAModel (Original JAModelEmpirical)
#----------------------------
class JAModel(BaseModelEmpirical):
    """Jouyban-Acree model for solubility in mixed solvents at constant temperature."""
    
    groups = ja_groups
    
    
    def __init__(self, x):
        """Initialize the JAModel.
        
        Parameters:
        x (int): Number of data points to use for fitting, excluding endpoints
        """
        super().__init__()
        self.JA_number = x
    
    def __repr__(self):
        if self.results_df is None:
            return f"JAModel(x={self.JA_number})"
        else:
            return f"JAModel(x={self.JA_number}, median mape={self.results_df['mape'].median():.4f})"
    
    @classmethod
    def load_from_csvs(cls, input_folder='output'):
        """Load JAModel instances from CSV files."""
        loaded = []

        # Get all CSV files that match the pattern for curve fit results
        csv_files = glob.glob(f'{input_folder}/curve_fit_results_x_is_*.csv')

        # Load each file into a dictionary with keys indicating parameters
        for file in csv_files:
            # Extract x from filename
            filename = os.path.basename(file)
            x_value = filename.replace('curve_fit_results_x_is_', '').replace('.csv', '')
            
            # Load the CSV file into a dataframe
            df = pd.read_csv(file)
            
            new_model = cls(int(x_value))
            new_model.results_df = df
            loaded.append(new_model)
        return loaded
    
    def curve_fitter(self):
        """Fit the Jouyban-Acree model to experimental data.
        
        The method selects x uniform points from each group, excluding the first and last points,
        which are always included in the fitting.
        """
        results = []
        failed_groups = []
        skipped_groups = []
        
        ja_model = JouybanAcreeModel()

        for gn in tqdm(range(len(self.groups)), desc="Processing groups"):
            chosen_df = self.groups[gn]
                            
            solvent_2_pure = chosen_df[chosen_df['solvent_1_weight_fraction'] <= 0.01].iloc[0]['solubility_g_g']
            solvent_1_pure = chosen_df[chosen_df['solvent_1_weight_fraction'] >= 0.99].iloc[0]['solubility_g_g']
            specific_temperature = chosen_df['temperature'].iloc[0]
            
            # Prepare data for Jouyban-Acree fitting at the chosen temperature
            fitting_df = chosen_df[chosen_df['temperature'] == specific_temperature] \
                .sort_values(by='solvent_1_weight_fraction') \
                .reset_index(drop=True) 

            n = len(fitting_df)
            if n < self.JA_number+2:  # Skip groups that don't have enough points
                print(f"Skipping group {gn} due to insufficient data points for JA fitting")
                skipped_groups.append(gn)
                continue

            # Select points for Jouyban-Acree fitting - keeping endpoints and adding uniform middle points
            if self.JA_number > 0:
                middle_indices = uniform_sample_indices(n-2, self.JA_number)
                middle_indices = [idx+1 for idx in middle_indices]  # Shift indices to account for endpoints
                fitting_indices = [0] + middle_indices + [n-1]
            else:
                fitting_indices = [0, n-1]  # Just endpoints if JA_number = 0

            fitting_df = fitting_df.iloc[fitting_indices].reset_index(drop=True)
            
            # Fit the Jouyban-Acree model
            try:
                params, pcov = ja_model.fit(
                    fitting_df['solvent_1_weight_fraction'], 
                    fitting_df['solubility_g_g'],
                    solvent_1_pure, 
                    solvent_2_pure, 
                    specific_temperature
                )
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                failed_groups.append(gn)
                continue
            
            if (pcov is None or np.isnan(pcov).any() or np.isinf(pcov).any()):
                print(f"Failed to fit group {gn} due to covariance issues")
                failed_groups.append(gn)
                continue
            
            # Extract the fitted parameters
            J0, J1, J2 = params
            
            # Calculate predicted solubility for all experimental data points
            predicted_solubility = ja_model.predict(
                chosen_df['solvent_1_weight_fraction'],
                solvent_1_pure,
                solvent_2_pure,
                specific_temperature,
                J0, J1, J2
            )
            
            # Calculate error metrics
            rmse = np.sqrt(mean_squared_error(chosen_df['solubility_g_g'], predicted_solubility))
            r2 = r2_score(chosen_df['solubility_g_g'], predicted_solubility)
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
        
        print(f"Number of skipped groups: {len(skipped_groups)}")
        print(f"Number of failed groups: {len(failed_groups)}")
        
        self.results_df = pd.DataFrame(results)
    
    def save_results(self, output_folder='output'):
        """Save the results to a CSV file."""
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.results_df.to_csv(f'{output_folder}/curve_fit_results_x_is_{self.JA_number}.csv', index=False)
        print(f"curve_fit_results_x_is_{self.JA_number}.csv")
    
    
    def plot(self, n):
        """Plot the experimental data and fitted model for a specific group.
        
        Parameters:
        n (int): Index of the group to plot
        """
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        selected = self.groups[self.results_df['group_index'].iloc[n]]
        J0 = self.results_df['J0'].iloc[n]
        J1 = self.results_df['J1'].iloc[n]
        J2 = self.results_df['J2'].iloc[n]
        solvent_1_pure = self.results_df['solvent_1_pure'].iloc[n]
        solvent_2_pure = self.results_df['solvent_2_pure'].iloc[n]
        specific_temperature = self.results_df['temperature'].iloc[n]

        ja_model = JouybanAcreeModel()  
        x_values = np.linspace(0, 1, 101)
        
        jouyban_acree_fit_values = ja_model.predict(
            x_values, 
            solvent_1_pure, 
            solvent_2_pure, 
            specific_temperature, 
            J0, J1, J2
        )

        # Plot the JA model
        plt.figure(figsize=(16*1.3/3, 9*1.3/3))
        plt.plot(x_values, jouyban_acree_fit_values, label='Jouyban-Acree Model', color='blue')

        # Add the experimental data points to the plot
        plt.scatter(selected['solvent_1_weight_fraction'], selected['solubility_g_g'], color='red', label='Experimental Data', zorder=5)
        plt.xlabel('Solvent 1 Weight Fraction')
        plt.ylabel('Solubility (g/g)')
        plt.title('Solubility vs Solvent 1 Weight Fraction (JA Model)')
        plt.legend()
        plt.grid(True)
        plt.show()


#----------------------------
# JAVHModel (Original JAVHModelEmpirical)
#----------------------------
class JAVHModel(BaseModelEmpirical):
    """Jouyban-Acree-Van't Hoff model for solubility in mixed solvents with temperature dependence."""
    
    groups = javh_groups
    
    def __init__(self, VH_number=3, JA_number=3):
        """Initialize the JAVHModel.
        
        Parameters:
        VH_number (int): Number of data points to use for van't Hoff fitting
        JA_number (int): Number of data points to use for Jouyban-Acree fitting
        """
        super().__init__()
        self.VH_number = VH_number
        self.JA_number = JA_number
    
    def __repr__(self):
        if self.results_df is None:
            return f"JAVHModel(VH_number={self.VH_number}, JA_number={self.JA_number})"
        else:
            return f"JAVHModel(VH_number={self.VH_number}, JA_number={self.JA_number}, median mape={self.results_df['mape'].median():.4f})"

    @classmethod
    def load_from_csvs(cls, input_folder='output'):
        """Load JAVHModel instances from CSV files."""
        loaded = []

        # Get all CSV files that match the pattern for curve fit results
        csv_files = glob.glob(f'{input_folder}/curve_fit_results_VH_number_is_*_JA_number_is_*.csv')

        # Load each file into a dictionary with keys indicating parameters
        for file in csv_files:
            # Extract VH_number and JA_number from filename
            filename = os.path.basename(file)
            parts = filename.replace('curve_fit_results_VH_number_is_', '').replace('.csv', '').split('_JA_number_is_')
            
            if len(parts) == 2:
                vh_part, ja_part = parts
                            
                # Load the CSV file into a dataframe
                df = pd.read_csv(file)
                
                new_model = cls(VH_number=int(vh_part), JA_number=int(ja_part))
                new_model.results_df = df
                loaded.append(new_model)
        return loaded
    
    def curve_fitter(self):
        """Fit the Jouyban-Acree-Van't Hoff model to experimental data."""
        results = []
        failed_groups = []
        skipped_groups = []
        
        vh_model = VanHoffModel()
        javh_model = JouybanAcreeVanHoffModel()

        for gn in tqdm(range(len(self.groups)), desc="Processing groups"):
            chosen_df = self.groups[gn]
            
            # Get pure solvent samples
            solvent_2_pure_samples = chosen_df[chosen_df['solvent_1_weight_fraction'] <= 0.01][['solubility_g_g','temperature','solvent_1_weight_fraction']]
            solvent_1_pure_samples = chosen_df[chosen_df['solvent_1_weight_fraction'] >= 0.99][['solubility_g_g','temperature','solvent_1_weight_fraction']]

            if len(solvent_2_pure_samples) < self.VH_number or len(solvent_1_pure_samples) < self.VH_number:
                skipped_groups.append(gn)
                continue

            # Select temperature points for van't Hoff fitting
            uniform_temp_indexes = uniform_sample_indices(len(solvent_1_pure_samples), self.VH_number)

            fitting_df_solvent_1 = solvent_1_pure_samples.iloc[uniform_temp_indexes].reset_index(drop=True)
            fitting_df_solvent_2 = solvent_2_pure_samples.iloc[uniform_temp_indexes].reset_index(drop=True)

            chosen_temp_index = random.randint(0, len(fitting_df_solvent_1)-1)
            chosen_temp = float(fitting_df_solvent_1['temperature'].iloc[chosen_temp_index])
            
            # Fit van't Hoff model for both pure solvents
            try:
                params_solvent_1, pcov_solvent_1 = vh_model.fit(
                    fitting_df_solvent_1['temperature'], 
                    fitting_df_solvent_1['solubility_g_g']
                )
                
                params_solvent_2, pcov_solvent_2 = vh_model.fit(
                    fitting_df_solvent_2['temperature'], 
                    fitting_df_solvent_2['solubility_g_g']
                )
            except RuntimeError as e:
                print(f"RuntimeError in van't Hoff fitting: {e}")
                failed_groups.append(gn)
                continue
            
            if (pcov_solvent_1 is None or np.isnan(pcov_solvent_1).any() or np.isinf(pcov_solvent_1).any()) or \
               (pcov_solvent_2 is None or np.isnan(pcov_solvent_2).any() or np.isinf(pcov_solvent_2).any()):
                print(f"Failed to fit group {gn} due to covariance issues in van't Hoff fitting")
                failed_groups.append(gn)
                continue
            
            alpha1, beta1 = params_solvent_1
            alpha2, beta2 = params_solvent_2

            # Prepare data for Jouyban-Acree fitting at the chosen temperature
            fitting_df = chosen_df[chosen_df['temperature'] == chosen_temp] \
                .sort_values(by='solvent_1_weight_fraction') \
                .reset_index(drop=True) 
            
            n = len(fitting_df)
            if n < self.JA_number+2:  # Skip groups that don't have enough points
                print(f"Skipping group {gn} due to insufficient data points for JA fitting")
                skipped_groups.append(gn)
                continue
            
            # Select points for Jouyban-Acree fitting - keeping endpoints and adding uniform middle points
            if self.JA_number > 0:
                middle_indices = uniform_sample_indices(n-2, self.JA_number)
                middle_indices = [idx+1 for idx in middle_indices]  # Shift indices to account for endpoints
                fitting_indices = [0] + middle_indices + [n-1]
            else:
                fitting_indices = [0, n-1]  # Just endpoints if JA_number = 0

            fitting_df = fitting_df.iloc[fitting_indices].reset_index(drop=True)
            
            # Fit Jouyban-Acree-Van't Hoff model
            try:
                params, pcov = javh_model.fit(
                    fitting_df['solvent_1_weight_fraction'],
                    fitting_df['temperature'],
                    fitting_df['solubility_g_g'],
                    alpha1, beta1, alpha2, beta2
                )
            except RuntimeError as e:
                print(f"RuntimeError in JA-VH fitting: {e}")
                failed_groups.append(gn)
                continue
            
            if (pcov is None or np.isnan(pcov).any() or np.isinf(pcov).any()):
                print(f"Failed to fit group {gn} due to covariance issues in JA-VH fitting")
                failed_groups.append(gn)
                continue

            # Extract the fitted parameters
            J0, J1, J2 = params
            
            # Calculate predicted solubility for all experimental data points
            predicted_solubility = javh_model.predict(
                chosen_df['solvent_1_weight_fraction'],
                chosen_df['temperature'],
                alpha1, beta1, alpha2, beta2,
                J0, J1, J2
            )
            
            # Calculate error metrics
            rmse = np.sqrt(mean_squared_error(chosen_df['solubility_g_g'], predicted_solubility))
            r2 = r2_score(chosen_df['solubility_g_g'], predicted_solubility)
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

        print(f"Number of skipped groups: {len(skipped_groups)}")
        print(f"Number of failed groups: {len(failed_groups)}")  
        
        self.results_df = pd.DataFrame(results)
    
    def save_results(self, output_folder='output'):
        """Save the results to a CSV file."""
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.results_df.to_csv(
            f'{output_folder}/curve_fit_results_VH_number_is_{self.VH_number}_JA_number_is_{self.JA_number}.csv', 
            index=False
        )
        print(f"curve_fit_results_VH_number_is_{self.VH_number}_JA_number_is_{self.JA_number}.csv")
    
    def plot_VH(self, n,solvent_1=None, solvent_2=None):
        """Plot the van't Hoff model for a specific group.
        
        Parameters:
        n (int): Index of the group to plot
        """
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        selected = self.groups[self.results_df['group_index'].iloc[n]]
        alpha1 = self.results_df['alpha1'].iloc[n]
        beta1 = self.results_df['beta1'].iloc[n]
        alpha2 = self.results_df['alpha2'].iloc[n]
        beta2 = self.results_df['beta2'].iloc[n]

        vh_model = VanHoffModel()
        
        # Generate temperature values for the plot
        temperature_values = np.linspace(selected['temperature'].min(), selected['temperature'].max(), 100)

        # Calculate solubility values for the plot
        solubility_solvent_1 = vh_model.predict(temperature_values, alpha1, beta1)
        solubility_solvent_2 = vh_model.predict(temperature_values, alpha2, beta2)

        # Plotting
        plt.figure(figsize=(16*1.3/3, 9*1.3/3))
        plt.plot(temperature_values, solubility_solvent_1, label=f'Solvent 1: {solvent_1 if solvent_1 else selected["solvent_1"].iloc[0]  }', color='blue')
        plt.plot(temperature_values, solubility_solvent_2, label=f'Solvent 2: {solvent_2 if solvent_2 else selected["solvent_2"].iloc[0]}', color='red',linestyle='--')
        
        # Add the experimental data points to the plot
        pure_solvents = selected[(selected['solvent_1_weight_fraction'] >= 0.99) | (selected['solvent_1_weight_fraction'] <= 0.01)]
        plt.scatter(pure_solvents['temperature'], pure_solvents['solubility_g_g'], label='Experimental Data', zorder=5, color='grey')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Solubility (g/g)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot(self, n, temperature=298.15,all_experimental_data=True):
        """Plot the Jouyban-Acree-Van't Hoff model for a specific group at a given temperature.
        
        Parameters:
        n (int): Index of the group to plot
        temperature (float): Temperature in Kelvin for which to plot the model
        """
        if self.results_df is None:
            raise ValueError("No results available. Please run the curve_fitter method first.")
        
        selected = self.groups[self.results_df['group_index'].iloc[n]]
        J0 = self.results_df['J0'].iloc[n]
        J1 = self.results_df['J1'].iloc[n]
        J2 = self.results_df['J2'].iloc[n]
        alpha1 = self.results_df['alpha1'].iloc[n]
        beta1 = self.results_df['beta1'].iloc[n]
        alpha2 = self.results_df['alpha2'].iloc[n]
        beta2 = self.results_df['beta2'].iloc[n]

        javh_model = JouybanAcreeVanHoffModel()
        
        x_values = np.linspace(0, 1, 101)
        temps = np.ones_like(x_values) * temperature

        jouyban_acree_VH_fit_values = javh_model.predict(
            x_values, temps, alpha1, beta1, alpha2, beta2, J0, J1, J2
        )

        plt.figure(figsize=(16*1.3/3, 9*1.3/3))
        plt.plot(x_values, jouyban_acree_VH_fit_values, label='Jouyban-Acree Model', color='blue')
        
        # Add the experimental data points to the plot
        if all_experimental_data:
            scatter = plt.scatter(
                selected['solvent_1_weight_fraction'], 
                selected['solubility_g_g'], 
                c=selected['temperature'], 
                cmap='viridis', 
                label='Experimental Data', 
                zorder=5
            )
            plt.colorbar(scatter, label='Temperature (K)', ticks=selected['temperature'].unique())
        else:
            scatter = plt.scatter(
                selected[selected['temperature'] == temperature]['solvent_1_weight_fraction'], 
                selected[selected['temperature'] == temperature]['solubility_g_g'], 
                color='slategrey', 
                label='Experimental Data', 
                zorder=5
            )
        
        plt.xlabel('Solvent 1 Weight Fraction')
        plt.ylabel('Solubility (g/g)')
        plt.legend()
        plt.grid(True)
        plt.show()


#----------------------------
# Main execution
#----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run solubility models with specified parameters.")
    parser.add_argument("--model", type=str, choices=["ja", "javh", "both"], default="both", 
                       help="Which model to run: 'ja', 'javh', or 'both'")
    args = parser.parse_args()

    if args.model in ["ja", "both"]:
        print("Running Jouyban-Acree model...")
        # Run JA model for different numbers of fitting points
        for x in range(3, 10):
            print(f"Running with x={x}")
            model = JAModel(x=x)
            model.curve_fitter()
            model.save_results()
    
    if args.model in ["javh", "both"]:
        print("Running Jouyban-Acree-Van't Hoff model...")
        # Run JAVH model for different numbers of fitting points
        for VH_number in range(3, 10):
            for JA_number in range(3, 10):
                print(f"Running with VH_number={VH_number}, JA_number={JA_number}")
                model = JAVHModel(VH_number=VH_number, JA_number=JA_number)
                model.curve_fitter()
                model.save_results()