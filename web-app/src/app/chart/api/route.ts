// app/api/solubility/route.ts
import { NextResponse } from "next/server";
import { db } from "@/db";
import { sql } from "drizzle-orm";
import { compounds, solvents, solubility, baoSolubility } from "@/db/schema";

export const dynamic = "force-static";

export async function GET() {
  try {
    const results1 = await db
      .select({
        compoundId: solubility.compoundId,
        solvent1: solubility.solvent1,
        solvent2: solubility.solvent2,
        weightFractions: sql`GROUP_CONCAT(${solubility.solvent1WeightFraction})`,
        solubilityValues: sql`GROUP_CONCAT(${solubility.solubilityMolMol})`,
        temperatureValues: sql`GROUP_CONCAT(${solubility.temperature})`,
      })
      .from(solubility)
      .groupBy(solubility.compoundId, solubility.solvent1, solubility.solvent2)
      .orderBy(solubility.compoundId, solubility.solvent1, solubility.solvent2)

    const results2 = await db
      .select({
        compoundId: baoSolubility.drug,
        solvent1: baoSolubility.solvent1,
        solvent2: baoSolubility.solvent2,
        weightFractions: sql`GROUP_CONCAT(${baoSolubility.solvent1WeightFraction})`,
        solubilityValues: sql`GROUP_CONCAT(${baoSolubility["solubility_(mol/mol)"]})`,
        temperatureValues: sql`GROUP_CONCAT(${baoSolubility["temperature_(k)"]})`,
      })
      .from(baoSolubility)
      .groupBy(baoSolubility.drug, baoSolubility.solvent1, baoSolubility.solvent2)
      .orderBy(baoSolubility.drug, baoSolubility.solvent1, baoSolubility.solvent2)

    const processedResults = (results1.concat(results2)).map(row => ({
      ...row,
      weightFractions: row.weightFractions?.split(',').map(Number),
      solubilityValues: row.solubilityValues?.split(',').map(Number),
      temperatureValues: row.temperatureValues?.split(',').map(Number),
    }));

    return NextResponse.json(processedResults);
  } catch (error) {
    console.error("Error fetching solubility data:", error);
    return NextResponse.json(
      { error: "Failed to fetch solubility data" },
      { status: 500 }
    );
  }
}