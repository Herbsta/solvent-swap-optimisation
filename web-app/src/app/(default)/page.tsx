import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Home() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>ETL Process</CardTitle>
      </CardHeader>
      <CardContent>
        <a href="/download/api" className={buttonVariants({ variant: "link" })}>
          Download database.db
        </a>
      </CardContent>
    </Card>
  );
}
