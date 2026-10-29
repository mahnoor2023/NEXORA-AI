import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({ baseDirectory: dirname(fileURLToPath(import.meta.url)) });

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      "next-env.d.ts",
      "backend/.venv/**",
      "**/.pytest_cache/**",
      "**/__pycache__/**",
      "**/.cache/**",
      "runtime/**",
      "backend/runtime/**",
      ".uv-cache/**",
      ".pip-cache/**",
      ".npm-cache/**",
    ],
  },
];

export default eslintConfig;
