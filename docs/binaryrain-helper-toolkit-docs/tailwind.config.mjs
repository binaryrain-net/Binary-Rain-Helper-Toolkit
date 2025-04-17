import colors from "tailwindcss/colors";
import starlightPlugin from "@astrojs/starlight-tailwind";

// Generated color palettes
const accent = {
  200: "#aad7a0",
  600: "#165800",
  900: "#0d3e00",
  950: "#072d00",
};
const gray = {
  100: "#f3f7f5",
  200: "#e8f0eb",
  300: "#bbc5bf",
  400: "#7d9186",
  500: "#4b5d53",
  700: "#2b3d33",
  800: "#1a2b22",
  900: "#131a16",
};

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: { accent, gray },
      fontFamily: {
        // Your preferred text font. Starlight uses a system font stack by default.
        sans: ['"Atkinson Hyperlegible"'],
        // Your preferred code font. Starlight uses system monospace fonts by default.
        mono: ['"IBM Plex Mono"'],
      },
    },
  },
  plugins: [starlightPlugin()],
};
