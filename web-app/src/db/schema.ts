import { sqliteTable, AnySQLiteColumn, text, real, integer, numeric } from "drizzle-orm/sqlite-core"
  import { sql } from "drizzle-orm"

export const baoSolubility = sqliteTable("bao_solubility", {
	webOfScienceIndex: text("web_of_science_index"),
	drug: text(),
	solvent1: text("solvent_1"),
	solvent1WeightFraction: real("solvent_1_weight_fraction"),
	solvent1MolFraction: real("solvent_1_mol_fraction"),
	solvent2: text("solvent_2"),
	"temperature_(k)": real("temperature_(k)"),
	"solubility_(mol/mol)": real("solubility_(mol/mol)"),
	doi: text(),
});

export const baoDrugs = sqliteTable("bao_drugs", {
	drug: text(),
	"drugs@fda": text("drugs@fda"),
	cas: text(),
	smiles: text(),
	canonicalSmiles: text("canonical_smiles"),
	"casMp_(c)": real("cas_mp_(c)"),
	"chemicalBookMp_(c)": real("chemical_book_mp_(c)"),
	"chemspiderMp_(c)": real("chemspider_mp_(c)"),
	"scbtMp_(c)": real("scbt_mp_(c)"),
	"sigmaMp_(c)": real("sigma_mp_(c)"),
	"fisherMp_(c)": real("fisher_mp_(c)"),
	"drugbankMp_(c)": real("drugbank_mp_(c)"),
	"pubchemMp_(c)": real("pubchem_mp_(c)"),
	"lktLabsMp_©": real("lkt_labs_mp_©"),
	"wikiMp_(c)": real("wiki_mp_(c)"),
	"pesticidePropertiesDatabaseMp_(c)": real("pesticide_properties_database_mp_(c)"),
	"akScientific_(c)": real("ak_scientific_(c)"),
	"moltusResearchLab_(c)": real("moltus_research_lab_(c)"),
	"chemsrc_(c)": real("chemsrc_(c)"),
	"tci_(c)": real("tci_(c)"),
	"chembk_(c)": real("chembk_(c)"),
	"guidechem_(c)": real("guidechem_(c)"),
	"echemi_(c)": real("echemi_(c)"),
	"ebclink_(c)": real("ebclink_(c)"),
	"collectedMeltingTemp_©": real("collected_melting_temp_©"),
	"meltingTempStd_(c)": real("melting_temp_std_(c)"),
	"meltingTempCv_(c)": real("melting_temp_cv_(c)"),
	"collectedMeltingTemp_(k)": real("collected_melting_temp_(k)"),
	"predictedMeltingTemp_(c)": integer("predicted_melting_temp_(c)"),
	"predictedMeltingTemp_(k)": real("predicted_melting_temp_(k)"),
	ae: real(),
	"ae^2": real("ae^2"),
	"unnamed:32": real("unnamed:_32"),
	"unnamed:33": text("unnamed:_33"),
	"unnamed:34": real("unnamed:_34"),
});

export const baoSolvents = sqliteTable("bao_solvents", {
	solvent: text(),
	cas: text(),
	isomericSmiles: text("isomeric_smiles"),
	smiles: text(),
	"casMp_(c)": real("cas_mp_(c)"),
	"scbtMp_(c)": real("scbt_mp_(c)"),
	"sigmaMp_(c)": real("sigma_mp_(c)"),
	"fisherMp_(c)": real("fisher_mp_(c)"),
	"wikiMp_(c)": real("wiki_mp_(c)"),
	"collectedMeltingTemp_(c)": real("collected_melting_temp_(c)"),
	"meltingTempStd_(c)": real("melting_temp_std_(c)"),
	"meltingTempCv_(c)": real("melting_temp_cv_(c)"),
	"collectedMeltingTemp_(k)": real("collected_melting_temp_(k)"),
	"publicDatabase_(uWashington/uOfMassachusettsAmherst/louisianaStateU)": real("public_database_(u_washington/u_of_massachusetts_amherst/louisiana_state_u)"),
	stenutz: real(),
	literature: real(),
	"unnamed:16": text("unnamed:_16"),
	dielectricConstant: real("dielectric_constant"),
});

export const compounds = sqliteTable("compounds", {
	pubchemId: integer("pubchem_id").primaryKey(),
	canonicalSmiles: text("canonical_smiles"),
	molecularWeight: real("molecular_weight"),
	molecularName: text("molecular_name"),
	originalNames: text("original_names"),
});

export const solvents = sqliteTable("solvents", {
	pubchemId: integer("pubchem_id").primaryKey(),
	canonicalSmiles: text("canonical_smiles"),
	molecularWeight: real("molecular_weight"),
	molecularName: text("molecular_name"),
	originalNames: text("original_names"),
});

export const solubility = sqliteTable("solubility", {
	compoundId: integer("compound_id"),
	solvent1: integer("solvent_1"),
	solvent1WeightFraction: real("solvent_1_weight_fraction"),
	solvent1MolFraction: real("solvent_1_mol_fraction"),
	solvent2: integer("solvent_2"),
	temperature: numeric(),
	solubilityMolMol: real("solubility_mol_mol"),
	solubilityGG: real("solubility_g_g"),
	comment: numeric(),
	reference: text(),
});

