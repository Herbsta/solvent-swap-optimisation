import { drizzle } from 'drizzle-orm/better-sqlite3';

export const db = drizzle('./src/db/MasterDatabase.db');