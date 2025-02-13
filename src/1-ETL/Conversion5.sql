UPDATE combined_solubility
SET solvent_1_weight_fraction = 
    ROUND(
        solvent_1_mol_g_fraction * s1.molecular_weight,
        4
    )
FROM solvents s1
WHERE s1.pubchem_id = combined_solubility.solvent_1
AND combined_solubility.solvent_1_weight_fraction IS NULL 
AND combined_solubility.solvent_1_mol_g_fraction IS NOT NULL;

-- Fill in missing weight fractions where we have mole fractions
UPDATE combined_solubility
SET solvent_1_weight_fraction = 
    ROUND(
        (solvent_1_mol_fraction * s1.molecular_weight) / 
        (solvent_1_mol_fraction * s1.molecular_weight + 
         (1 - solvent_1_mol_fraction) * s2.molecular_weight),
        4
    )
FROM solvents s1, solvents s2
WHERE s1.pubchem_id = combined_solubility.solvent_1
AND s2.pubchem_id = combined_solubility.solvent_2
AND combined_solubility.solvent_1_weight_fraction IS NULL 
AND combined_solubility.solvent_1_mol_fraction IS NOT NULL;

-- Fill in missing mole fractions where we have weight fractions
UPDATE combined_solubility
SET solvent_1_mol_fraction = 
    ROUND(
        (solvent_1_weight_fraction * s2.molecular_weight) / 
        (solvent_1_weight_fraction * s2.molecular_weight + 
         (1 - solvent_1_weight_fraction) * s1.molecular_weight),
        4
    )
FROM solvents s1, solvents s2
WHERE s1.pubchem_id = combined_solubility.solvent_1
AND s2.pubchem_id = combined_solubility.solvent_2
AND combined_solubility.solvent_1_mol_fraction IS NULL 
AND combined_solubility.solvent_1_weight_fraction IS NOT NULL;

UPDATE combined_solubility
SET solubility_g_g = (
    SELECT
    ROUND(
        combined_solubility.solubility_mol_g * c.molecular_weight,
        4
    )
    FROM compounds c
    WHERE c.pubchem_id = combined_solubility.compound_id
)
WHERE solubility_g_g IS NULL
AND solubility_mol_g IS NOT NULL;

-- Convert solubility from mol/mol to g/g handling both single and binary solvent systems
UPDATE combined_solubility
SET solubility_g_g = (
    SELECT 
        CASE 
            -- Single solvent system (solvent_2 is NULL)
            WHEN combined_solubility.solvent_2 IS NULL THEN
                ROUND(
                    combined_solubility.solubility_mol_mol * c.molecular_weight / s1.molecular_weight,
                    4
                )
            -- Binary solvent system
            ELSE
                ROUND(
                    combined_solubility.solubility_mol_mol * c.molecular_weight / 
                    (
                        combined_solubility.solvent_1_mol_fraction * s1.molecular_weight + 
                        (1 - combined_solubility.solvent_1_mol_fraction) * s2.molecular_weight
                    ),
                    4
                )
        END
    FROM compounds c, solvents s1
    LEFT JOIN solvents s2 ON s2.pubchem_id = combined_solubility.solvent_2
    WHERE c.pubchem_id = combined_solubility.compound_id
    AND s1.pubchem_id = combined_solubility.solvent_1
)
WHERE solubility_g_g IS NULL 
AND solubility_mol_mol IS NOT NULL
AND (
    -- Ensure we have mol fraction for binary systems
    (solvent_2 IS NOT NULL AND solvent_1_mol_fraction IS NOT NULL) OR
    -- For single solvent, mol fraction should be 1 or NULL
    (solvent_2 IS NULL AND (solvent_1_mol_fraction IS NULL OR solvent_1_mol_fraction = 1))
);

UPDATE combined_solubility
SET solubility_mol_mol = (
    SELECT
    CASE
        -- Single solvent system (solvent_2 is NULL)
        WHEN combined_solubility.solvent_2 IS NULL THEN
            ROUND(
                combined_solubility.solubility_g_g * s1.molecular_weight / c.molecular_weight,
                4
            )
        -- Binary solvent system
        ELSE
            ROUND(
                combined_solubility.solubility_g_g * 
                (
                    combined_solubility.solvent_1_mol_fraction * s1.molecular_weight +
                    (1 - combined_solubility.solvent_1_mol_fraction) * s2.molecular_weight
                ) / c.molecular_weight,
                4
            )
    END
    FROM compounds c, solvents s1
    LEFT JOIN solvents s2 ON s2.pubchem_id = combined_solubility.solvent_2
    WHERE c.pubchem_id = combined_solubility.compound_id
    AND s1.pubchem_id = combined_solubility.solvent_1
)
WHERE solubility_mol_mol IS NULL
AND solubility_g_g IS NOT NULL
AND (
    -- Ensure we have mol fraction for binary systems
    (solvent_2 IS NOT NULL AND solvent_1_mol_fraction IS NOT NULL) OR
    -- For single solvent, mol fraction should be 1 or NULL
    (solvent_2 IS NULL AND (solvent_1_mol_fraction IS NULL OR solvent_1_mol_fraction = 1))
);

-- Step 1: Create a new table without the solvent_1_mol_g_fraction and solubility_mol_g columns
CREATE TABLE solubility AS
SELECT 
    compound_id, 
    solvent_1, 
    solvent_1_weight_fraction, 
    solvent_1_mol_fraction, 
    solvent_2, 
    temperature, 
    solubility_mol_mol, 
    solubility_g_g, 
    -- Combine solvent_ratio and comment into one string under comment
    COALESCE(solvent_ratio, '') || ' | ' || COALESCE(comment, '') AS comment, 
    reference
FROM 
    combined_solubility;

-- Step 2: Drop the old table
DROP TABLE combined_solubility;
