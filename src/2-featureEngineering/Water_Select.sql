-- Select statement for all permuatations that contain H20 as a solvent
SELECT 
    compound_id AS api,
    solubility_g_g,
    CASE 
        WHEN solvent_1 = 962 THEN solvent_2
        ELSE solvent_1
    END AS solvent_1,
    CASE 
        WHEN solvent_1 = 962 THEN 1 - solvent_1_weight_fraction
        ELSE solvent_1_weight_fraction
    END AS solvent_1_weight_fraction,
    temperature
FROM
    solubility s
WHERE 
    (solvent_2 = 962 OR solvent_1 = 962)
    AND solvent_2 IS NOT NULL
    AND (solvent_1_weight_fraction <> 1);
