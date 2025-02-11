ALTER TABLE api_dual_solvent_data ADD COLUMN solvent_1_id INTEGER;
ALTER TABLE api_dual_solvent_data ADD COLUMN solvent_2_id INTEGER;
ALTER TABLE api_dual_solvent_data ADD COLUMN compound_name_id INTEGER;

UPDATE api_dual_solvent_data 
SET solvent_1_id = (
    SELECT pubchem_id 
    FROM solvents 
    WHERE lower(original_names) LIKE '%' || lower(solvent_1) || '%'
    LIMIT 1
);

UPDATE api_dual_solvent_data 
SET solvent_2_id = (
    SELECT pubchem_id 
    FROM solvents 
    WHERE lower(original_names) LIKE '%' || lower(solvent_2) || '%'
    LIMIT 1
);

UPDATE api_dual_solvent_data 
SET compound_name_id = (
    SELECT pubchem_id 
    FROM compounds 
    WHERE lower(original_names) LIKE '%' || lower(compound_name) || '%'
    LIMIT 1
);

-- For api_single_solvent_data

ALTER TABLE api_single_solvent_data ADD COLUMN solvent_id INTEGER;
ALTER TABLE api_single_solvent_data ADD COLUMN compound_name_id INTEGER;

UPDATE api_single_solvent_data
SET solvent_id = (
    SELECT pubchem_id 
    FROM solvents 
    WHERE lower(original_names) LIKE '%' || lower(solvent) || '%'
    LIMIT 1
);

UPDATE api_single_solvent_data
SET compound_name_id = (
    SELECT pubchem_id 
    FROM compounds 
    WHERE lower(original_names) LIKE '%' || lower(compound_name) || '%'
    LIMIT 1
);



-- For pubchem_dual_solvent_data

ALTER TABLE pubchem_dual_solvent_data ADD COLUMN solvent_1_id INTEGER;
ALTER TABLE pubchem_dual_solvent_data ADD COLUMN solvent_2_id INTEGER;


UPDATE pubchem_dual_solvent_data
SET solvent_1_id = (
    SELECT pubchem_id 
    FROM solvents 
    WHERE lower(original_names) LIKE '%' || lower(solvent_1) || '%'
    LIMIT 1
);

UPDATE pubchem_dual_solvent_data
SET solvent_2_id = (
    SELECT pubchem_id 
    FROM solvents 
    WHERE lower(original_names) LIKE '%' || lower(solvent_2) || '%'
    LIMIT 1
);

-- For pubchem_single_solvent_data

ALTER TABLE pubchem_single_solvent_data ADD COLUMN solvent_id INTEGER;

UPDATE pubchem_single_solvent_data
SET solvent_id = (
    SELECT pubchem_id 
    FROM solvents 
    WHERE lower(original_names) LIKE '%' || lower(solvent) || '%'
    LIMIT 1
);

-- for bao solubility 

ALTER TABLE bao_solubility ADD COLUMN solvent_1_id INTEGER;
ALTER TABLE bao_solubility ADD COLUMN solvent_2_id INTEGER;
ALTER TABLE bao_solubility ADD COLUMN compound_name_id INTEGER;

UPDATE bao_solubility 
SET solvent_1_id = (
    SELECT pubchem_id 
    FROM solvents 
    WHERE lower(original_names) LIKE '%' || lower(solvent_1) || '%'
    LIMIT 1
);

UPDATE bao_solubility 
SET solvent_2_id = (
    SELECT pubchem_id 
    FROM solvents 
    WHERE lower(original_names) LIKE '%' || lower(solvent_2) || '%'
    LIMIT 1
);

UPDATE bao_solubility 
SET compound_name_id = (
    SELECT pubchem_id 
    FROM compounds 
    WHERE lower(original_names) LIKE '%' || lower(drug) || '%'
    LIMIT 1
);