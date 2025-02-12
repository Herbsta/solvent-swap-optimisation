import fs from 'fs';
import { NextResponse } from 'next/server';

export const dynamic = 'force-static'

export async function GET() {

  const filePath = './src/db/MasterDatabase.db';

  try {
    const fileContents = fs.readFileSync(filePath);
    return new NextResponse(fileContents, {
      status: 200,
      headers: new Headers({
        'Content-Disposition': `attachment; filename="MasterDatabase.db"`,
        'Content-Type': 'application/octet-stream',
      }),
    });
  } catch (error) {
    console.error('Error serving file:', error);
    return NextResponse.json({ error: 'File not found' });
  }
}