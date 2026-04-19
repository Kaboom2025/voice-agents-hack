import type { Config } from "tailwindcss";
import typography from "@tailwindcss/typography";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0b1220",      // primary text (slate-950-ish)
        mute: "#64748b",     // slate-500
        edge: "#e2e8f0",     // slate-200 hairline
        accent: "#f97316",   // orange-500
      },
      boxShadow: {
        glass: "0 1px 2px rgba(15,23,42,0.04), 0 12px 32px -16px rgba(15,23,42,0.12)",
      },
      backdropBlur: {
        xs: "2px",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [typography],
} satisfies Config;
