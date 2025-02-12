import { defineConfig } from "drizzle-kit";

export default defineConfig({
    out: "./drizzle",
    dialect: 'sqlite',
    dbCredentials: {
      url: "./src/db/MasterDatabase.db",
    },
    verbose: true,
    strict: true,
  });
