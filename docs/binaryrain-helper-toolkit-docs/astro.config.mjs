// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

import tailwindcss from "@tailwindcss/vite";

// https://astro.build/config
export default defineConfig({
  site: "https://binaryrain-net.github.io",
  base: "Binary-Rain-Helper-Toolkit",
  integrations: [
    starlight({
      title: "Binary Rain Helper Toolkit",
      description:
        "A collection of python toolkits aimed to simplify common functions.",
      favicon: "/src/assets/binaryrain.png",
      logo: {
        light: "/src/assets/binaryrain.png",
        dark: "/src/assets/binaryrain.png",
      },
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/binaryrain-net/Binary-Rain-Helper-Toolkit",
        },
        {
          icon: "seti:python",
          label: "PyPI",
          href: "https://pypi.org/user/Binary-Rain/",
        },
      ],
      sidebar: [
        {
          label: "Overview",
          slug: "overview",
        },
        {
          label: "Toolkits",
          autogenerate: { directory: "toolkits" },
        },
        {
          label: "Template Repositories",
          autogenerate: { directory: "templates" },
        },
      ],
      tableOfContents: {
        maxHeadingLevel: 3,
        minHeadingLevel: 2,
      },
      customCss: ["./src/styles/global.css"],
    }),
  ],

  vite: {
    plugins: [tailwindcss()],
  },
});
