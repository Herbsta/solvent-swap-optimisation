DROP TABLE IF EXISTS combined_solubility;

CREATE TABLE combined_solubility AS
-- Dual solvent API data
SELECT 
   CAST(compound_name_id AS INTEGER) as compound_id,
   solvent_1_id as solvent_1,
   solvent_1_weight_fraction,
   solvent_1_mol_fraction,
   NULL as solvent_1_mol_g_fraction,
   solvent_2_id as solvent_2,
   temperature_k as temperature,
   solubility_mol_mol,
   solubility_g_g,
   NULL as solubility_mol_g,
   solvent_ratio,
   comments as comment,
   reference
FROM api_dual_solvent_data

UNION ALL

-- Single solvent API data
SELECT 
   CAST(compound_name_id AS INTEGER) as compound_id,
   solvent_id as solvent_1,
   1.0 as solvent_1_weight_fraction,
   1.0 as solvent_1_mol_fraction,
   NULL as solvent_1_mol_g_fraction,
   NULL as solvent_2,
   temperature_k as temperature,
   solubility_mol_mol,
   solubility_g_g,
   NULL as solubility_mol_g,
   NULL as solvent_ratio,
   comments as comment,
   reference
FROM api_single_solvent_data

UNION ALL

-- Dual solvent PubChem data
SELECT 
   CAST(pub_chem_id AS INTEGER) as compound_id,
   solvent_1_id as solvent_1,
   solvent_1_weight_fraction,
   solvent_1_mol_fraction,
   solvent_1_mol_g as solvent_1_mol_g_fraction,
   solvent_2_id as solvent_2,
   temperature_k as temperature,
   solubility_mol_mol,
   solubility_g_g,
   solubility_mol_g,
   solvent_ratio,
   comment,
   reference
FROM pubchem_dual_solvent_data

UNION ALL

-- Single solvent PubChem data
SELECT 
   CAST(pub_chem_id AS INTEGER) as compound_id,
   solvent_id as solvent_1,
   1.0 as solvent_1_weight_fraction,
   1.0 as solvent_1_mol_fraction,
   NULL as solvent_1_mol_g_fraction,
   NULL as solvent_2,
   temperature_k as temperature,
   solubility_mol_mol,
   solubility_g_g,
   solubility_mol_g,
   NULL as solvent_ratio,
   comment,
   reference
FROM pubchem_single_solvent_data

UNION ALL

-- Bao solubility
SELECT 
   CAST(compound_name_id AS INTEGER) as compound_id,
   solvent_1_id as solvent_1,
   solvent_1_weight_fraction,
   NULL as solvent_1_mol_g_fraction,
   solvent_1_mol_fraction,
   solvent_2_id as solvent_2,
   "temperature_(k)" as temperature,
   "solubility_(mol/mol)" as solubility_mol_mol,
   NULL as solubility_g_g,
   NULL as solubility_mol_g,
   NULL as solvent_ratio,
   NULL as comment,
   doi as reference
FROM bao_solubility;

DROP TABLE IF EXISTS api_dual_solvent_data;
DROP TABLE IF EXISTS api_single_solvent_data;

DROP TABLE IF EXISTS pubchem_single_solvent_data;
DROP TABLE IF EXISTS pubchem_dual_solvent_data;

DROP TABLE IF EXISTS bao_solubility;
