import * as React from "react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { SocialsFooter } from "@/components/layout/socials-footer";
import { Link } from "@/components/custom/link";

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <main className="container mx-auto flex min-h-screen flex-col gap-4 p-4 sm:p-16">
        <div className="flex flex-col gap-4 sm:gap-8 w-full max-w-7xl mx-auto relative min-h-full h-full rounded-lg border border-border/50 bg-background/50 p-4 backdrop-blur-[2px] sm:p-8">
          <div className="grid gap-1">
            <h1 className="text-3xl font-semibold text-foreground">
              Solvent Swapping Optimisation
            </h1>
            <h2 className="text-lg text-muted-foreground">
              A Loughborough university dissertation to utilise Variational Autoencoders
              to aide with the prediction of API solubility within binary solvent systems
            </h2>
            <p className="text-muted-foreground">
              Code available on{" "}
              <Link href="https://github.com/Herbsta/Solvent_Swap_Optimisation">
                GitHub
              </Link>
              .
            </p>
          </div>
          <Separator />
          {children}
          <Badge
            variant="outline"
            className="absolute -top-2.5 left-4 bg-background sm:left-8"
          >
            Work in progress
          </Badge>
        </div>
        <SocialsFooter />
      </main>
    </>
  );
}
