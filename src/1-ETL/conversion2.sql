-- Create a working table from the imported data
DROP TABLE IF EXISTS working_table;

CREATE TABLE working_table AS
SELECT
    pub_chem_id,
    solubility_g_l,
    saturation,
    temperature_k,
    solvent,
    CASE
    WHEN comment LIKE '%Ratio of solvents:%'
    THEN CASE 
         -- Handle the case with comma separator (e.g., "Ratio of solvents: 0.2051:0.7949 w/w, Solubility...")
         WHEN comment LIKE '%Ratio of solvents:%' AND comment LIKE '%, %'
         THEN trim(substr(
            comment,
            instr(comment, 'Ratio of solvents:') + 18,
            instr(substr(comment, instr(comment, 'Ratio of solvents:') + 18), ',') - 1
         ))
         -- Handle the simple percentage case (e.g., "Ratio of solvents: 20% v/v EtOH")
         ELSE trim(substr(
            comment,
            instr(comment, 'Ratio of solvents:') + 18
         ))
         END
    ELSE ratio_of_solvents
    END as solvent_ratio,
    CASE
    WHEN comment LIKE '%Ratio of solvents:%' AND comment LIKE '%, %'
    THEN trim(substr(comment, 
        instr(comment, ',') + 1))
    WHEN comment LIKE '%Ratio of solvents:%'
    THEN NULL  -- or '' depending on your preference for the simple percentage case
    ELSE comment
    END as comment,
    reference
FROM imported_data;


-- Add columns for standardized solubility values
ALTER TABLE working_table ADD COLUMN solubility_mol_mol REAL;
ALTER TABLE working_table ADD COLUMN solubility_mol_g REAL;
ALTER TABLE working_table ADD COLUMN solubility_g_g REAL;
ALTER TABLE working_table ADD COLUMN solubility_vol_vol REAL;
ALTER TABLE working_table ADD COLUMN solubility_mol_litre REAL;

ALTER TABLE working_table ADD COLUMN found INTEGER;

UPDATE working_table
SET found = 1
WHERE solubility_g_l IS NOT NULL;

UPDATE working_table
SET 
    solubility_mol_mol = CASE 
        WHEN comment LIKE '%mol-percent%' 
            THEN CAST(SUBSTR(comment, 
                            INSTR(comment, ':') + 2, 
                            INSTR(comment, ' mol-percent') - INSTR(comment, ':') - 2) AS REAL) / 100
        WHEN comment LIKE '%molpercent%'
            THEN CAST(SUBSTR(comment, 
                        INSTR(comment, ':') + 2, 
                        INSTR(comment, ' molpercent') - INSTR(comment, ':') - 2) AS REAL) / 100
        WHEN comment LIKE '%mol/1000mol%'
            THEN CAST(SUBSTR(comment,
                        INSTR(comment, ':') + 2,
                        INSTR(comment, ' mol/1000mol') - INSTR(comment, ':') - 2) AS REAL) / 1000
        WHEN comment LIKE '%mol/mol%'
            THEN CAST(SUBSTR(comment,
                        INSTR(comment, ':') + 2,
                        INSTR(comment, ' mol/mol') - INSTR(comment, ':') - 2) AS REAL)
    END,
    found = 1
WHERE (comment LIKE '%Solubilty:%' OR comment LIKE '%Solubility:%')
    AND (comment LIKE '%molpercent%' OR comment LIKE '%mol-percent%' 
         OR comment LIKE '%mol/1000mol%' OR comment LIKE '%mol/mol%');

UPDATE working_table
SET 
solubility_mol_g = CASE
    WHEN comment LIKE '%mol/kg%'
    THEN CAST(SUBSTR(comment,
        INSTR(comment, ':') + 2,
        INSTR(comment, ' mol/kg') - INSTR(comment, ':') - 2) AS REAL) / 1000
    END,
found = 1
WHERE (comment LIKE '%Solubilty:%' OR comment LIKE '%Solubility:%')
AND comment LIKE '%mol/kg%';

-- gram per gram
UPDATE working_table
SET
solubility_g_g = CASE
WHEN comment LIKE '%weight percent%'
THEN CAST(SUBSTR(comment,
INSTR(comment, ':') + 2,
INSTR(comment, ' weight percent') - INSTR(comment, ':') - 2) AS REAL) / 100
WHEN comment LIKE '%weight-percent%'
THEN CAST(SUBSTR(comment,
INSTR(comment, ':') + 2,
INSTR(comment, ' weight-percent') - INSTR(comment, ':') - 2) AS REAL) / 100
WHEN comment LIKE '%g/kg%'
THEN CAST(SUBSTR(comment,
INSTR(comment, ':') + 2,
INSTR(comment, ' g/kg') - INSTR(comment, ':') - 2) AS REAL) / 1000
WHEN comment LIKE '%mg/g%'
THEN CAST(SUBSTR(comment,
INSTR(comment, ':') + 2,
INSTR(comment, ' mg/g') - INSTR(comment, ':') - 2) AS REAL) / 1000
WHEN comment LIKE '%g solvent dissolves%'
THEN CAST(SUBSTR(comment,
INSTR(comment, 'dissolves') + 10,
INSTR(comment, ' g Substance') - INSTR(comment, 'dissolves') - 10) AS REAL) / 
CAST(SUBSTR(comment, 0, 
INSTR(comment, ' g solvent')) AS REAL)
WHEN comment LIKE '%p(g/100g solution)%'
THEN CAST(SUBSTR(comment,
INSTR(comment, ':') + 2,
INSTR(comment, ' p(g/100g solution)') - INSTR(comment, ':') - 2) AS REAL) / 100
WHEN comment LIKE '%g/100g%' AND comment NOT LIKE '%solution%'
THEN CAST(SUBSTR(comment,
INSTR(comment, ':') + 2,
INSTR(comment, ' g/100g') - INSTR(comment, ':') - 2) AS REAL) / 100
WHEN comment LIKE '%part(s) of substance.dissolves in:100 parts of solvent%'
THEN CAST(SUBSTR(comment, 0, 
INSTR(comment, ' part(s)')) AS REAL) / 100
WHEN comment LIKE '%part(s) of substance.dissolves in:%'
THEN CASE
-- Handle range case (e.g., 35-40)
WHEN SUBSTR(comment, INSTR(comment, 'in:') + 3,
INSTR(comment, ' parts of solvent') - INSTR(comment, 'in:') - 3) LIKE '%-%'
THEN CAST(SUBSTR(comment, 1, INSTR(comment, ' part(s)') - 1) AS REAL) /
((CAST(SUBSTR(SUBSTR(comment, INSTR(comment, 'in:') + 3), 1,
INSTR(SUBSTR(comment, INSTR(comment, 'in:') + 3), '-') - 1) AS REAL) +
CAST(SUBSTR(SUBSTR(comment, INSTR(comment, 'in:') + 3,
INSTR(comment, ' parts of solvent') - INSTR(comment, 'in:') - 3),
INSTR(SUBSTR(comment, INSTR(comment, 'in:') + 3), '-') + 1) AS REAL)) / 2)
-- Handle normal case
ELSE CAST(SUBSTR(comment, 1, INSTR(comment, ' part(s)') - 1) AS REAL) /
CAST(SUBSTR(comment, INSTR(comment, 'in:') + 3,
INSTR(comment, ' parts of solvent') - INSTR(comment, 'in:') - 3) AS REAL)
END
END,
found = 1
WHERE (comment LIKE '%Solubilty:%' OR comment LIKE '%Solubility:%' OR comment LIKE '%g solvent dissolves%' OR comment LIKE '%part(s) of substance.dissolves in:%')
AND (comment LIKE '%weight percent%'
OR comment LIKE '%weight-percent%'
OR comment LIKE '%g/kg%'
OR comment LIKE '%mg/g%'
OR comment LIKE '%g solvent dissolves%'
OR comment LIKE '%p(g/100g solution)%'
OR (comment LIKE '%g/100g%' AND comment NOT LIKE '%solution%')
OR comment LIKE '%part(s) of substance.dissolves in:%');

UPDATE working_table
SET
solubility_mol_litre = CAST(SUBSTR(comment,
    INSTR(comment, ':') + 2,
    INSTR(comment, ' mol/l') - INSTR(comment, ':') - 2) AS REAL),
found = 1
WHERE (comment LIKE '%Solubilty:%' OR comment LIKE '%Solubility:%')
AND comment LIKE '%mol/l%';

UPDATE working_table
SET
solubility_mol_litre = CAST(SUBSTR(comment,
    INSTR(comment, ':') + 2,
    INSTR(comment, ' mol/l') - INSTR(comment, ':') - 2) AS REAL),
found = 1
WHERE (comment LIKE '%Solubilty:%' OR comment LIKE '%Solubility:%')
AND comment LIKE '%mol/l%';

UPDATE working_table
SET 
solubility_vol_vol = CASE
    WHEN comment LIKE '%vol-percent%'
    THEN CAST(SUBSTR(comment,
        INSTR(comment, ':') + 2,
        INSTR(comment, ' vol-percent') - INSTR(comment, ':') - 2) AS REAL) / 100
    WHEN comment LIKE '%volpercent%'
    THEN CAST(SUBSTR(comment,
        INSTR(comment, ':') + 2,
        INSTR(comment, ' volpercent') - INSTR(comment, ':') - 2) AS REAL) / 100
    WHEN comment LIKE '%vol/vol%'
    THEN CAST(SUBSTR(comment,
        INSTR(comment, ':') + 2,
        INSTR(comment, ' vol/vol') - INSTR(comment, ':') - 2) AS REAL)
END,
found = 1
WHERE (comment LIKE '%Solubilty:%' OR comment LIKE '%Solubility:%')
AND (comment LIKE '%volpercent%' OR comment LIKE '%vol-percent%' 
    OR comment LIKE '%vol/vol%');


-- Remove rows where no standardization was possible
 DELETE FROM working_table WHERE found IS NULL;

-- Create single solvent table
DROP TABLE IF EXISTS single_solvent_data;

CREATE TABLE single_solvent_data AS
SELECT 
    pub_chem_id,
    solubility_mol_mol,
    solubility_g_l,
    solubility_mol_g,
    solubility_g_g,
    solubility_vol_vol,
    solubility_mol_litre,
    saturation,
    temperature_k,
    solvent,
    comment,
    reference
FROM working_table
WHERE (solvent_ratio IS NULL OR solvent_ratio = '');

-- Create dual solvent table
DROP TABLE IF EXISTS dual_solvent_data;

CREATE TABLE dual_solvent_data AS
SELECT 
    pub_chem_id,
    solubility_mol_mol,
    solubility_g_l,
    solubility_mol_g,
    solubility_g_g,
    solubility_vol_vol,
    solubility_mol_litre,
    saturation,
    temperature_k,
    SUBSTR(solvent, 1, INSTR(solvent, ', ') - 1) as solvent_1,
    SUBSTR(solvent, INSTR(solvent, ', ') + 2) as solvent_2,
    solvent_ratio,
    comment,
    reference
FROM working_table
WHERE solvent LIKE '%, %'
AND (solvent_ratio IS NOT NULL AND solvent_ratio != '');


ALTER TABLE dual_solvent_data ADD COLUMN solvent_1_mol_fraction REAL;
ALTER TABLE dual_solvent_data ADD COLUMN solvent_1_weight_fraction REAL;
ALTER TABLE dual_solvent_data ADD COLUMN solvent_1_vol_fraction REAL;

UPDATE dual_solvent_data
SET solvent_1_mol_fraction =
CASE
-- Handle 'molpercent' format in parentheses
WHEN solvent_ratio LIKE '%(% molpercent)%' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            INSTR(solvent_ratio, '(') + 1,
            INSTR(solvent_ratio, ' molpercent)') - INSTR(solvent_ratio, '(') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'xxxx:xxxx mol/mol' format
WHEN solvent_ratio LIKE '%:%_mol/mol' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ':') - 1
        ) AS FLOAT
    ) / (
        CAST(
            SUBSTR(
                solvent_ratio,
                1,
                INSTR(solvent_ratio, ':') - 1
            ) AS FLOAT
        ) +
        CAST(
            SUBSTR(
                solvent_ratio,
                INSTR(solvent_ratio, ':') + 1,
                INSTR(solvent_ratio, ' mol/mol') - INSTR(solvent_ratio, ':') - 1
            ) AS FLOAT
        )
    )
-- Handle 'xx:xx mol percent' format
WHEN solvent_ratio LIKE '%:%_mol percent' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ':') - 1
        ) AS FLOAT
    ) / 100
-- Handle '(x.xxxx mol)' format
WHEN solvent_ratio LIKE '%(% mol)%' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            INSTR(solvent_ratio, '(') + 1,
            INSTR(solvent_ratio, ' mol)') - INSTR(solvent_ratio, '(') - 1
        ) AS FLOAT
    )
-- Handle 'x.xxx/x.xxx mol/mol' format
WHEN solvent_ratio LIKE '%/%_mol/mol' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, '/') - 1
        ) AS FLOAT
    )
-- Handle 'xx.xx mol percent dioxane' format
WHEN solvent_ratio LIKE '% mol percent dioxane' THEN
    (100 - CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ' mol percent') - 1
        ) AS FLOAT
    )) / 100
END
WHERE solvent_ratio LIKE '%(% molpercent)%'
OR solvent_ratio LIKE '%:%_mol/mol'
OR solvent_ratio LIKE '%:%_mol percent'
OR solvent_ratio LIKE '%(% mol)%'
OR solvent_ratio LIKE '%/%_mol/mol'
OR solvent_ratio LIKE '% mol percent dioxane';

UPDATE dual_solvent_data
SET solvent_1_weight_fraction =
CASE
-- Handle 'weight percent' format
WHEN solvent_ratio LIKE '%(% weight percent)%' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            INSTR(solvent_ratio, '(') + 1,
            INSTR(solvent_ratio, ' weight percent)') - INSTR(solvent_ratio, '(') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'xx:xx w/w' format
WHEN solvent_ratio LIKE '%:%_w/w' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ':') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'solvent1 (X g); solvent2 (Y g)' format
WHEN solvent_ratio LIKE '%(% g);%(% g)' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            INSTR(solvent_ratio, '(') + 1,
            INSTR(solvent_ratio, ' g)') - INSTR(solvent_ratio, '(') - 1
        ) AS FLOAT
    ) / (
        CAST(
            SUBSTR(
                solvent_ratio,
                INSTR(solvent_ratio, '(') + 1,
                INSTR(solvent_ratio, ' g)') - INSTR(solvent_ratio, '(') - 1
            ) AS FLOAT
        ) +
        CAST(
            SUBSTR(
                solvent_ratio,
                INSTR(solvent_ratio, ';') + INSTR(SUBSTR(solvent_ratio, INSTR(solvent_ratio, ';')), '(') + 1,
                INSTR(SUBSTR(solvent_ratio, INSTR(solvent_ratio, ';')), ' g)') - INSTR(SUBSTR(solvent_ratio, INSTR(solvent_ratio, ';')), '(') - 1
            ) AS FLOAT
        )
    )
-- Handle 'xx:xx percentWt' format
WHEN solvent_ratio LIKE '%:%_percentWt' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ':') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'xx:xx wtpercent' format
WHEN solvent_ratio LIKE '%:%_wtpercent' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ':') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'xx wt percent : xx wt percent' format
WHEN solvent_ratio LIKE '%_wt percent :%_wt percent' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ' wt percent') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'xxpercent wt' format
WHEN solvent_ratio LIKE '%percent wt' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, 'percent') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'xx/xxpercent Wt' format
WHEN solvent_ratio LIKE '%/%percent Wt' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, '/') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'xx:xx wt' format
WHEN solvent_ratio LIKE '%:%_wt' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ':') - 1
        ) AS FLOAT
    ) / 100
END
WHERE solvent_ratio LIKE '%(% weight percent)%'
OR solvent_ratio LIKE '%:%_w/w'
OR solvent_ratio LIKE '%(% g);%(% g)'
OR solvent_ratio LIKE '%:%_percentWt'
OR solvent_ratio LIKE '%:%_wtpercent'
OR solvent_ratio LIKE '%_wt percent :%_wt percent'
OR solvent_ratio LIKE '%percent wt'
OR solvent_ratio LIKE '%/%percent Wt'
OR solvent_ratio LIKE '%:%_wt';

UPDATE dual_solvent_data
SET solvent_1_vol_fraction = 
CASE
-- Handle '(xx vol percent)' format
WHEN solvent_ratio LIKE '%(% vol percent)%' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            INSTR(solvent_ratio, '(') + 1,
            INSTR(solvent_ratio, ' vol percent)') - INSTR(solvent_ratio, '(') - 1
        ) AS FLOAT
    ) / 100
-- Handle 'xx:xx v/v' format
WHEN solvent_ratio LIKE '%:%_v/v' THEN
    CAST(
        SUBSTR(
            solvent_ratio,
            1,
            INSTR(solvent_ratio, ':') - 1
        ) AS FLOAT
    ) / 100
END
WHERE solvent_ratio LIKE '%(% vol percent)%'
OR solvent_ratio LIKE '%:%_v/v';

-- Special case where H20 is notes as solvent 1
ALTER TABLE dual_solvent_data ADD COLUMN solvent_1_mol_g REAL;

UPDATE dual_solvent_data 
SET 
    solvent_1 = solvent_2,
    solvent_2 = solvent_1,
    solvent_1_mol_g = CAST(
        REPLACE(SUBSTR(solvent_ratio, 1, INSTR(solvent_ratio, ' mol/kg')), ' mol/kg', '') AS DECIMAL
    ) / 1000.0
WHERE solvent_ratio LIKE '% mol/kg %';

DELETE FROM dual_solvent_data
WHERE solvent_1_mol_fraction IS NULL
    AND solvent_1_weight_fraction IS NULL
    AND solvent_1_vol_fraction IS NULL
	AND solvent_1_mol_g IS NULL;


-- Delete the current table
 DROP TABLE working_table;
 DROP TABLE imported_data;
