'use client';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import DBDownloadButton from "./db-download";

export default function Home() {

  return (
    <Card>
      <CardHeader>
        <CardTitle>ETL Process</CardTitle>
      </CardHeader>
      <CardContent>
        <DBDownloadButton />
      </CardContent>
    </Card>
  );
}
