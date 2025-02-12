// app/api/download/route.js
import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

export const dynamic = 'force-static'

const DB_PATH = path.join(process.cwd(), 'src/db/MasterDatabase.db')

export async function GET() {
  try {
    // Check if database file exists
    try {
      await fs.access(DB_PATH)
    } catch {
      return NextResponse.json(
        { error: 'Database file not found' },
        { status: 404 }
      )
    }

    const fileBuffer = await fs.readFile(DB_PATH)

    const response = new NextResponse(fileBuffer)

    response.headers.set(
      'Content-Disposition',
      `attachment; filename="database.db"`
    )
    response.headers.set('Content-Type', 'application/x-sqlite3')

    return response
  } catch (error) {
    console.error('Download error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}