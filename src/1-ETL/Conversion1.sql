DROP TABLE IF EXISTS working_table;

-- Copy the solobuility data into a working table
CREATE TABLE working_table AS
SELECT
	compound_name, 
	saturation, 
	CASE 
	    WHEN CAST(temperature AS FLOAT) < 200 
	    THEN CAST(temperature AS FLOAT) + 273 
	    ELSE CAST(temperature AS FLOAT) 
	END as temperature_k,
	solvent,
	CASE 
        WHEN comments LIKE '%Ratio of solvents:%' 
        THEN trim(substr(
            substr(comments, 1, instr(comments, char(10))-1),
            instr(comments, 'Ratio of solvents:') + 18
        ))
        ELSE solvent_ratio 
    END as solvent_ratio,
	CASE 
        WHEN comments LIKE '%Ratio of solvents:%' 
        THEN trim(substr(comments, instr(comments, char(10))+1))
        ELSE comments 
    END as comments,
	reference 
FROM solubility_data;

DELETE FROM working_table
WHERE temperature_k IS NULL;

DELETE FROM working_table
WHERE comments IS NULL;

-- Add a new column to the table
ALTER TABLE working_table ADD COLUMN solubility_mol_mol REAL;
ALTER TABLE working_table ADD COLUMN solubility_g_g REAL;
ALTER TABLE working_table ADD COLUMN found INTEGER;

-- Update the new column with the extracted number from the comments column
-- Moles per mol
UPDATE working_table
SET 
    solubility_mol_mol = CASE 
        WHEN comments LIKE '%mol-percent%' 
            THEN CAST(SUBSTR(comments, 
                            INSTR(comments, ':') + 2, 
                            INSTR(comments, ' mol-percent') - INSTR(comments, ':') - 2) AS REAL) / 100
        WHEN comments LIKE '%molpercent%'
            THEN CAST(SUBSTR(comments, 
                        INSTR(comments, ':') + 2, 
                        INSTR(comments, ' molpercent') - INSTR(comments, ':') - 2) AS REAL) / 100
        WHEN comments LIKE '%mol/1000mol%'
            THEN CAST(SUBSTR(comments,
                        INSTR(comments, ':') + 2,
                        INSTR(comments, ' mol/1000mol') - INSTR(comments, ':') - 2) AS REAL) / 1000
        WHEN comments LIKE '%mol/mol%'
            THEN CAST(SUBSTR(comments,
                        INSTR(comments, ':') + 2,
                        INSTR(comments, ' mol/mol') - INSTR(comments, ':') - 2) AS REAL)
    END,
    found = 1
WHERE (comments LIKE '%Solubilty:%' OR comments LIKE '%Solubility:%')
    AND (comments LIKE '%molpercent%' OR comments LIKE '%mol-percent%' 
         OR comments LIKE '%mol/1000mol%' OR comments LIKE '%mol/mol%');

-- gram per gram
UPDATE working_table
SET 
    solubility_g_g = CASE 
        WHEN comments LIKE '%weight percent%' 
            THEN CAST(SUBSTR(comments, 
                            INSTR(comments, ':') + 2, 
                            INSTR(comments, ' weight percent') - INSTR(comments, ':') - 2) AS REAL) / 100
        WHEN comments LIKE '%weight-percent%' 
            THEN CAST(SUBSTR(comments, 
                            INSTR(comments, ':') + 2, 
                            INSTR(comments, ' weight-percent') - INSTR(comments, ':') - 2) AS REAL) / 100
        WHEN comments LIKE '%g/kg%'
            THEN CAST(SUBSTR(comments, 
                        INSTR(comments, ':') + 2, 
                        INSTR(comments, ' g/kg') - INSTR(comments, ':') - 2) AS REAL) / 1000
        WHEN comments LIKE '%mg/g%'
            THEN CAST(SUBSTR(comments, 
                        INSTR(comments, ':') + 2, 
                        INSTR(comments, ' mg/g') - INSTR(comments, ':') - 2) AS REAL) / 1000
        WHEN comments LIKE '%g solvent dissolves%'
            THEN CAST(SUBSTR(comments, 
                        INSTR(comments, 'dissolves') + 10, 
                        INSTR(comments, ' g Substance') - INSTR(comments, 'dissolves') - 10) AS REAL) / 100
        WHEN comments LIKE '%p(g/100g solution)%'
            THEN CAST(SUBSTR(comments,
                        INSTR(comments, ':') + 2,
                        INSTR(comments, ' p(g/100g solution)') - INSTR(comments, ':') - 2) AS REAL) / 100
        WHEN comments LIKE '%g/100g%' AND comments NOT LIKE '%solution%'
            THEN CAST(SUBSTR(comments,
                        INSTR(comments, ':') + 2,
                        INSTR(comments, ' g/100g') - INSTR(comments, ':') - 2) AS REAL) / 100
        WHEN comments LIKE '%part(s) of substance.dissolves in:100%'
            THEN CAST(SUBSTR(comments, 
                        1,
                        INSTR(comments, ' part(s)') - 1) AS REAL) / 100
        WHEN comments LIKE '%part(s) of substance.dissolves in:%' 
            THEN CASE 
                -- Handle range case (e.g., 35-40)
                WHEN SUBSTR(comments, INSTR(comments, 'in:') + 3, 
                          INSTR(comments, ' parts of solvent') - INSTR(comments, 'in:') - 3) LIKE '%-%'
                THEN CAST(SUBSTR(comments, 1, INSTR(comments, ' part(s)') - 1) AS REAL) / 
                     ((CAST(SUBSTR(SUBSTR(comments, INSTR(comments, 'in:') + 3), 1, 
                           INSTR(SUBSTR(comments, INSTR(comments, 'in:') + 3), '-') - 1) AS REAL) +
                       CAST(SUBSTR(SUBSTR(comments, INSTR(comments, 'in:') + 3, 
                           INSTR(comments, ' parts of solvent') - INSTR(comments, 'in:') - 3),
                           INSTR(SUBSTR(comments, INSTR(comments, 'in:') + 3), '-') + 1) AS REAL)) / 2)
                -- Handle normal case
                ELSE CAST(SUBSTR(comments, 1, INSTR(comments, ' part(s)') - 1) AS REAL) /
                     CAST(SUBSTR(comments, INSTR(comments, 'in:') + 3, 
                          INSTR(comments, ' parts of solvent') - INSTR(comments, 'in:') - 3) AS REAL)
            END
    END,
    found = 1
WHERE (comments LIKE 'Solubility:%' OR comments LIKE '%dissolves%')
    AND (comments LIKE '%weight percent%' 
         OR comments LIKE '%weight-percent%'
         OR comments LIKE '%g/kg%'
         OR comments LIKE '%mg/g%'
         OR comments LIKE '%g solvent dissolves%'
         OR comments LIKE '%p(g/100g solution)%'
         OR (comments LIKE '%g/100g%' AND comments NOT LIKE '%solution%')
         OR comments LIKE '%part(s) of substance.dissolves in:%');

DELETE FROM working_table
WHERE found IS NULL;

DROP TABLE IF EXISTS single_solvent_data;

-- Table for data with single solvent
CREATE TABLE single_solvent_data AS
SELECT
	compound_name, 
	saturation,
	solubility_mol_mol,
	solubility_g_g,
	temperature_k,
	solvent,
	comments,
	reference 
FROM working_table
WHERE (solvent_ratio IS NULL OR solvent_ratio = '');

-- Table for data with two solvents
DROP TABLE IF EXISTS dual_solvent_data;

CREATE TABLE dual_solvent_data AS
SELECT 
    compound_name, 
	solubility_mol_mol,
	solubility_g_g,
    saturation, 
    temperature_k,
    substr(solvent, 1, instr(solvent, char(10))-1) as solvent_1,
    substr(solvent, instr(solvent, char(10))+1) as solvent_2,
    solvent_ratio,
    comments,
    reference
FROM working_table
WHERE comments IS NOT NULL
AND ((solvent_ratio IS NOT NULL AND solvent_ratio != '')
OR comments LIKE '%Ratio%');

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
    END
WHERE solvent_ratio LIKE '%(% molpercent)%'
   OR solvent_ratio LIKE '%:%_mol/mol'
   OR solvent_ratio LIKE '%:%_mol percent'
   OR solvent_ratio LIKE '%(% mol)%';

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
    END
WHERE solvent_ratio LIKE '%(% weight percent)%'
   OR solvent_ratio LIKE '%:%_w/w';

UPDATE dual_solvent_data
SET solvent_1_vol_fraction = CAST(
    SUBSTR(
        solvent_ratio,
        INSTR(solvent_ratio, '(') + 1,
        INSTR(solvent_ratio, ' vol percent)') - INSTR(solvent_ratio, '(') - 1
    ) AS FLOAT
) / 100
WHERE solvent_ratio LIKE '%(% vol percent)%';

DELETE FROM dual_solvent_data
WHERE solvent_1_mol_fraction IS NULL
    AND solvent_1_weight_fraction IS NULL
    AND solvent_1_vol_fraction IS NULL;

DROP TABLE solubility_data;

-- Delete the current table
DROP TABLE working_table;

