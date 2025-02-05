import { ModeToggle } from "@/components/theme/toggle-mode";
import { Button } from "@/components/ui/button";
import { Github, Globe, Linkedin } from "lucide-react";
import { Link } from "../custom/link";

export function SocialsFooter() {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-center items-center gap-2 p-1">
        <ModeToggle className="[&>svg]:h-4 [&>svg]:w-4" />
        <Button variant="ghost" size="sm" className="w-9 px-0" asChild>
          <Link href="https://github.com/Herbsta/Solvent_Swap_Optimisation">
            <Github className="h-4 w-4" />
          </Link>
        </Button>
        <Button variant="ghost" size="sm" className="w-9 px-0" asChild>
          <Link href="https://www.linkedin.com/in/herbiebingham/">
            <Linkedin className="h-4 w-4" />
          </Link>
        </Button>
        <Button variant="ghost" size="sm" className="w-9 px-0" asChild>
          <Link href="https://herbiebingham.co.uk/">
            <Globe className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      <p className="text-muted-foreground text-center text-sm">
        Powered by <Link href="https://openstatus.dev">Herbie Bingham</Link> with Loughborough University
      </p>
    </div>
  );
}
