"""
Jouyban-Acree Model for Predicting Drug Solubility in Binary Solvent Mixtures

This script implements the Jouyban-Acree model and Extended Hildebrand Solubility
Approach (EHSA) for predicting drug solubility in binary solvent mixtures.

The Jouyban-Acree model follows the equation:
log10(Sm) = f1*log10(C1) + f2*log10(C2) + (f1*f2/T)*[J0 + J1*(f1-f2) + J2*(f1-f2)^2]

Where:
- Sm is the solubility in mixed solvent
- f1, f2 are volume/weight fractions of solvents
- C1, C2 are solubilities in pure solvents
- T is absolute temperature
- J0, J1, J2 are model constants determined by regression
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Global constants
# Temperature in Kelvin and gas constant
T = 298       # 25°C
R = 1.987     # cal/(mol·K)

# Solvent properties
# Densities (g/ml)
DENSITY = {
    'water': 0.997,
    'ethanol': 0.785,
    'pg': 1.0273  # propylene glycol
}

# Molecular weights (g/mol)
MOL_WEIGHT = {
    'water': 18.0,
    'ethanol': 46.0,
    'pg': 76.09  # propylene glycol
}

# Solubility parameters (cal/cm³)^0.5
SOL_PARAM = {
    'water': 23.4,
    'ethanol': 13.0,
    'pg': 14.8  # propylene glycol
}

def mole_fraction(M, d1, d2, mw_1, mw_2):
    """
    Converts molar concentrations to mole fractions.
    
    Parameters:
    -----------
    M : numpy.ndarray
        Array of solute concentrations (g/L)
    d1, d2 : float
        Densities of solvents 1 and 2 (g/ml)
    mw_1, mw_2 : float
        Molecular weights of solvents 1 and 2 (g/mol)
    
    Returns:
    --------
    tuple
        (Xe, MW, d) where:
        Xe : numpy.ndarray - mole fractions of solute
        MW : numpy.ndarray - average molecular weight of solvent mixture
        d : numpy.ndarray - density of solvent mixture
    """
    # Define volume fractions of solvents
    if len(M) == 7:  # For experimental data points (7 different compositions)
        f1 = [1, 0.8, 0.6, 0.5, 0.4, 0.2, 0]  # Volume fraction of solvent 1
        f2 = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]  # Volume fraction of solvent 2
        f1 = np.array(f1).reshape(-1, 1)
        f2 = np.array(f2).reshape(-1, 1)
    else:  # For continuous prediction (101 points from 0 to 1)
        f1 = np.arange(0, 1.01, 0.01)
        f2 = 1 - f1
    
    # Convert volume fractions to molar quantities
    mol_1 = (f1 * d1) / mw_1  # moles of solvent 1 per unit volume
    mol_2 = (f2 * d2) / mw_2  # moles of solvent 2 per unit volume
    
    # Calculate mole fractions of solvents in the mixture (excluding solute)
    X2 = mol_2 / (mol_2 + mol_1)  # mole fraction of solvent 2
    X1 = mol_1 / (mol_2 + mol_1)  # mole fraction of solvent 1
    
    # Calculate average molecular weight and density of the solvent mixture
    MW = X2 * mw_2 + X1 * mw_1
    d = (f1 * d1) + (f2 * d2)
    
    # Convert solute concentration to mole fraction
    m = (M / (1000 * d)) * MW  # molality of solute (mol solute/kg solvent)
    Xe = m / (m + 1)  # mole fraction of solute
    
    return (Xe, MW, d)

def jouyban_acree(Cm, C1, C2, d1, d2):
    """
    Jouyban-Acree model for predicting solubility in binary solvent mixtures.
    
    Parameters:
    -----------
    Cm : numpy.ndarray
        Experimental solubility data in mixed solvents (g/L)
    C1, C2 : float
        Solubility in pure solvents 1 and 2 (g/L)
    d1, d2 : float
        Densities of solvents 1 and 2 (g/ml)
    
    Returns:
    --------
    numpy.ndarray
        Predicted solubility values across the entire composition range (g/L)
    """
    # Define volume fractions for experimental points
    F1 = [1, 0.8, 0.6, 0.5, 0.4, 0.2, 0]  # Volume fraction of solvent 1
    F2 = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]  # Volume fraction of solvent 2
    F1 = np.array(F1).reshape(-1, 1)
    F2 = np.array(F2).reshape(-1, 1)
    
    # Convert volume fractions to weight fractions
    W1 = F1 * d1
    W2 = F2 * d2
    f1 = W1 / (W2 + W1)  # Weight fraction of solvent 1
    f2 = W2 / (W2 + W1)  # Weight fraction of solvent 2
    
    # Take logarithms of solubility values (base 10)
    LCm = [math.log10(x) for x in Cm]  # Log of experimental solubility in mixtures
    LC1 = math.log10(C1)               # Log of solubility in pure solvent 1
    LC2 = math.log10(C2)               # Log of solubility in pure solvent 2
    LCm = np.array(LCm).reshape(-1, 1)
    
    # Calculate deviation from log-linear mixing rule
    x = LCm - (f1 * LC1 + f2 * LC2)
    
    # Regression variables for Jouyban-Acree model constants
    y0 = (f1 * f2) / T                     # For J0 term
    y1 = (f1 * f2 * (f1 - f2)) / T         # For J1 term
    y2 = (f1 * f2 * (f1 - f2) * (f1 - f2)) / T  # For J2 term
    
    # Linear regression to find model parameters (J values)
    reg0 = LinearRegression().fit(y0, x)
    reg1 = LinearRegression().fit(y1, x)
    reg2 = LinearRegression().fit(y2, x)
    
    j0 = float(reg0.coef_)  # Constant term
    j1 = float(reg1.coef_)  # First-order term
    j2 = float(reg2.coef_)  # Second-order term
    
    # Print model parameters for reference
    print(f"Jouyban-Acree model parameters: J0 = {j0:.2f}, J1 = {j1:.2f}, J2 = {j2:.2f}")
    
    # Define composition range for predictions (101 points)
    f12 = np.arange(0, 1.01, 0.01)  # Weight fraction of solvent 1
    f22 = 1 - f12                   # Weight fraction of solvent 2
    
    # Calculate predicted log solubility using Jouyban-Acree equation
    LSm = (f12 * LC1) + (f22 * LC2) + (f12 * f22 / T) * (j0 + (j1 * (f12 - f22)) + (j2 * (f12 - f22) * (f12 - f22)))
    
    # Convert back from logarithm to actual solubility
    Sm = 10**LSm
    
    return Sm

def ehsa(exp_data, T_fusion, dH_fusion, D_SP, D_V, sp1, sp2, d1, d2, mw_1, mw_2):
    """
    Extended Hildebrand Solubility Approach (EHSA) for predicting solubility
    in binary solvent mixtures.
    
    Parameters:
    -----------
    exp_data : numpy.ndarray
        Experimental solubility data in mixed solvents (g/L)
    T_fusion : float
        Melting point of the drug (K)
    dH_fusion : float
        Heat of fusion of the drug (cal/mol)
    D_SP : float
        Solubility parameter of the drug (cal/cm³)^0.5
    D_V : float
        Molar volume of the drug (cm³/mol)
    sp1, sp2 : float
        Solubility parameters of solvents 1 and 2 (cal/cm³)^0.5
    d1, d2 : float
        Densities of solvents 1 and 2 (g/ml)
    mw_1, mw_2 : float
        Molecular weights of solvents 1 and 2 (g/mol)
    
    Returns:
    --------
    tuple
        (X_calc, spmix, spmix2) where:
        X_calc : numpy.ndarray - predicted mole fractions across composition range
        spmix : numpy.ndarray - solubility parameters for experimental points
        spmix2 : numpy.ndarray - solubility parameters for prediction points
    """
    # Calculate mole fractions from experimental data
    Xe, MW, d = mole_fraction(exp_data, d1, d2, mw_1, mw_2)
    
    # Define volume fractions for experimental points
    f1 = [1, 0.8, 0.6, 0.5, 0.4, 0.2, 0]  # Volume fraction of solvent 1
    f2 = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]  # Volume fraction of solvent 2
    f1 = np.array(f1).reshape(-1, 1)
    f2 = np.array(f2).reshape(-1, 1)
    
    # Calculate solubility parameters for solvent mixtures
    spmix = (f1 * sp1) + (f2 * sp2)
    
    # Parameters for ideal solubility calculation
    T0 = T_fusion    # Melting point (K)
    dHf = dH_fusion  # Heat of fusion (cal/mol)
    sp = D_SP        # Drug solubility parameter (cal/cm³)^0.5
    V2 = D_V         # Drug molar volume (cm³/mol)
    
    # Calculate solvent molar volume
    V1 = MW / d
    
    # Volume fraction of drug in solution (approximated as 1)
    Q = 1
    
    # Coefficient for solubility parameter term
    A = (V2 * Q) / (R * T)
    
    # Calculate ideal solubility (mole fraction)
    LXi = (dHf / (2.303 * R * T)) * ((T0 - T) / T0)
    Xi = 10**(-LXi)
    
    # Calculate W parameter from experimental data
    W = ((np.log10(Xi / Xe) / A) + sp**2 + spmix**2) * 0.5
    
    # Polynomial regression to model W as function of solubility parameter
    x = spmix
    x1 = x.reshape(-1, 1)
    y = W
    
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly_features.fit_transform(x1)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    
    # Generate predictions across full composition range
    Q2 = 1  # Volume fraction of drug (simplified)
    f12 = np.arange(0, 1.01, 0.01)  # Volume fraction of solvent 1
    f22 = 1 - f12                   # Volume fraction of solvent 2
    spmix2 = (f22 * sp2) + (f12 * sp1)  # Solubility parameters across range
    
    A2 = (V2 * Q2) / (R * T)  # Coefficient for prediction
    
    # Calculate molar quantities and mixture properties
    mol_2 = (f22 * d2) / mw_2
    mol_1 = (f12 * d1) / mw_1
    X2 = mol_2 / (mol_2 + mol_1)  # Mole fraction of solvent 2
    X1 = mol_1 / (mol_2 + mol_1)  # Mole fraction of solvent 1
    MW = X2 * mw_2 + X1 * mw_1    # Average molecular weight
    d = (f22 * d2) + (f12 * d1)   # Density of mixture
    V1 = MW / d                   # Molar volume of mixture
    
    # Iterate to calculate solubility (simplified to one iteration)
    Q3 = Q2
    # Calculate W from polynomial regression
    W_calc = lin_reg.intercept_[0] + (spmix2 * lin_reg.coef_[0][0]) + ((spmix2**2) * lin_reg.coef_[0][1])
    
    # Calculate solubility using EHSA equation
    LX_calc = LXi - A2 * ((sp**2) + (spmix2**2) - (2 * W_calc))
    X_calc = 10**-LX_calc
    
    # Print EHSA model equation
    print(f"EHSA model equation: W = {lin_reg.intercept_[0]:.4f} + {lin_reg.coef_[0][0]:.4f} * S + {lin_reg.coef_[0][1]:.4f} * S²")
    
    return (X_calc, spmix, spmix2)

def plot_solubility_comparison(drug_name, experimental_data, ja_data, ehsa_data, solvent_system):
    """
    Creates a plot comparing experimental solubility data with predictions from
    Jouyban-Acree and EHSA models.
    
    Parameters:
    -----------
    drug_name : str
        Name of the drug
    experimental_data : tuple
        (x, y) for experimental data points
    ja_data : tuple
        (x, y) for Jouyban-Acree model predictions
    ehsa_data : tuple
        (x, y) for EHSA model predictions
    solvent_system : str
        Description of the solvent system (e.g., "Water-Ethanol")
    """
    plt.figure(figsize=(10, 6))
    
    # Plot experimental data
    plt.plot(
        experimental_data[0], experimental_data[1],
        "o", color='midnightblue', label="Experimental data"
    )
    
    # Plot EHSA model predictions
    plt.plot(
        ehsa_data[0], ehsa_data[1],
        color='orange', label="EHSA model"
    )
    
    # Plot Jouyban-Acree model predictions
    plt.plot(
        ja_data[0], ja_data[1],
        color='lightseagreen', label="Jouyban-Acree model"
    )
    
    plt.title(f"{drug_name} Solubility in {solvent_system} Mixture", fontsize=14)
    plt.xlabel("Solubility Parameter (cal/cm³)^0.5")
    plt.ylabel("Molar Solubility (mole fraction)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def main():
    """
    Example usage with Diclofenac sodium in Water-Ethanol mixture
    """
    print("Jouyban-Acree Model Demonstration: Diclofenac sodium in Water-Ethanol")
    print("-" * 70)
    
    # Drug properties - Diclofenac sodium
    drug_properties = {
        'name': 'Diclofenac sodium',
        'melting_point': 454.2,  # K
        'heat_fusion': 9416.826,  # cal/mol
        'solubility_param': 14.59,  # (cal/cm³)^0.5
        'molar_volume': 443.08,  # cm³/mol
        'solubility': {
            'water': 0.067,  # g/L
            'ethanol': 0.737,  # g/L
            'pg': 2.060  # g/L (propylene glycol)
        }
    }
    
    # Experimental solubility data for Water-Ethanol mixture
    Cm_WE = np.array([0.067, 0.093, 0.292, 0.480, 0.527, 0.655, 0.737]).reshape(-1, 1)
    
    # Get properties
    Cw = drug_properties['solubility']['water']
    Ce = drug_properties['solubility']['ethanol']
    dw = DENSITY['water']
    de = DENSITY['ethanol']
    mw_w = MOL_WEIGHT['water']
    mw_e = MOL_WEIGHT['ethanol']
    sp_w = SOL_PARAM['water']
    sp_e = SOL_PARAM['ethanol']
    
    # Calculate mole fractions from experimental data
    Cm_WE_mf, _, _ = mole_fraction(Cm_WE, dw, de, mw_w, mw_e)
    
    # Apply Jouyban-Acree model
    Sm_WE = jouyban_acree(Cm_WE, Cw, Ce, dw, de)
    Sm_WE_mf, _, _ = mole_fraction(Sm_WE, dw, de, mw_w, mw_e)
    
    # Apply EHSA model
    H_WE_mf, spmix_WE, spmix2_WE = ehsa(
        Cm_WE, 
        drug_properties['melting_point'], 
        drug_properties['heat_fusion'],
        drug_properties['solubility_param'],
        drug_properties['molar_volume'],
        sp_w, sp_e, dw, de, mw_w, mw_e
    )
    
    # Create plot
    plt = plot_solubility_comparison(
        drug_properties['name'],
        (spmix_WE, Cm_WE_mf),
        (spmix2_WE, Sm_WE_mf),
        (spmix2_WE, H_WE_mf),
        "Water-Ethanol"
    )
    
    print("\nPlot displayed. Close the plot window to exit the program.")
    plt.show()

if __name__ == "__main__":
    main()