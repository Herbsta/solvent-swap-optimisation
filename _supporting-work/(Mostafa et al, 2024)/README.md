# Cosolvency-Models
# Jouyban-Acree Model
The semi-empirical model that is generally used to predict physicochemical properties specially a solubility in mixed solvent. This model is expressed as following equation for calculating solubility in binary solvent mixture at different temperatures:
![image](https://github.com/mostafafazel/Solubility-Models/assets/72621118/528c2fee-504a-45d8-b3d2-8d0d8b9e5f4f)

Here C_(1,T)^Sat  ,C_(2,T)^Sat are saturated molar solubility in pure solvent 1, 2 and C_(m,T)^Sat is saturated molar solubility in mixture of solvents at the temperature T; w represented the mass fraction of each solvent. The Ji is the Jouyban-Acree model constant which calculated by regressing 
![image](https://github.com/mostafafazel/Solubility-Models/assets/72621118/477a7023-0950-4b80-9df7-15074a53864b)
by no intercept least square regression. In addition this model can be extended for predicting the saturated solubility in ternary solvent mixture which is expressed as:
![image](https://github.com/mostafafazel/Solubility-Models/assets/72621118/4f012799-b376-466e-8eac-95c529ba31bf)



# Extended Hildebrand solubility approach

The EHSA is predictive model based on thermodynamic properties which is derived form of the Hildebrand model for regular solution by using a solute-solvent interaction. The EHS model is expressed as follows:
![image](https://github.com/mostafafazel/Solubility-Models/assets/72621118/38f57460-1019-443b-8953-f04267289f63)

Where X_s^i represent an ideal mole fraction solubility of solute, V_s expresses the solute molar volume, R is universal gas constant, T is absolute temperature and δ_m,δ_s are the solubility parameter of mixture of solvent and solute respectively. φ_m shows the volume fraction of solvent mixture which is calculated as follows:

![image](https://github.com/mostafafazel/Solubility-Models/assets/72621118/76a88612-99bb-4420-a1ac-fef2576c90bc)


Where X_m is experimental solubility in solvent mixture and V_m shows the molar volume of solvent mixture.
The ideal mole fraction solubility is calculated as following equation:
![image](https://github.com/mostafafazel/Solubility-Models/assets/72621118/6c42c587-7070-4914-a547-351c63b90e00)
Here 〖∆H〗_f represents fusion enthalpy of solute and T_0  ,T are melting point of solute and absolute temperature respectively.
The solubility parameter of solvent mixture can be given by solubility parameter of pure solvent and volume fraction of each them:
![image](https://github.com/mostafafazel/Solubility-Models/assets/72621118/39ee904d-4593-449b-bf7c-c6303a3e3bd7)


