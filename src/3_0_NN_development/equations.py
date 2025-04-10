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