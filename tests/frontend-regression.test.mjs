import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import test from "node:test";
import ts from "typescript";
import { fileURLToPath } from "node:url";

const projectRoot = fileURLToPath(new URL("../", import.meta.url));
const sourceRoot = join(projectRoot, "src");

function sourceFiles(directory) {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.(tsx?|jsx?)$/.test(name) ? [path] : [];
  });
}

function isFunction(node) {
  return ts.isArrowFunction(node) || ts.isFunctionExpression(node) || ts.isFunctionDeclaration(node);
}

test("useEffect callbacks cannot implicitly return promises or non-function cleanup values", () => {
  const effects = [];
  for (const path of sourceFiles(sourceRoot)) {
    const text = readFileSync(path, "utf8");
    const ast = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, path.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
    const visit = (node) => {
      if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === "useEffect") effects.push({ path, node });
      ts.forEachChild(node, visit);
    };
    visit(ast);
  }
  assert.ok(effects.length > 0, "expected React effects to audit");
  for (const { path, node } of effects) {
    const callback = node.arguments[0];
    assert.ok(ts.isArrowFunction(callback), `${relative(projectRoot, path)}: useEffect must receive an explicit arrow callback`);
    assert.ok(!callback.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.AsyncKeyword), `${relative(projectRoot, path)}: useEffect callback must not be async`);
    assert.ok(ts.isBlock(callback.body), `${relative(projectRoot, path)}: useEffect callback must use a block so async/setter results are not returned implicitly`);
    const returns = [];
    const inspect = (child) => {
      if (child !== callback.body && isFunction(child)) return;
      if (ts.isReturnStatement(child)) returns.push(child);
      ts.forEachChild(child, inspect);
    };
    inspect(callback.body);
    for (const statement of returns) {
      if (!statement.expression) continue;
      assert.ok(ts.isArrowFunction(statement.expression) || ts.isFunctionExpression(statement.expression), `${relative(projectRoot, path)}: effect cleanup must be a function, never a Promise, setter result, or subscription object`);
    }
  }
});

test("dataset routes preserve a selected dataset and expose reopen actions", () => {
  const detailRoute = readFileSync(join(sourceRoot, "app/datasets/[datasetId]/page.tsx"), "utf8");
  const dashboard = readFileSync(join(sourceRoot, "components/dashboard.tsx"), "utf8");
  const pages = readFileSync(join(sourceRoot, "components/data-pages.tsx"), "utf8");
  const nav = readFileSync(join(sourceRoot, "components/app-nav.tsx"), "utf8");
  assert.match(detailRoute, /Dashboard\s+datasetId=\{datasetId\}/);
  assert.match(dashboard, /reopenDataset\(datasetId\)/);
  assert.match(dashboard, /downloadDatasetSource\(datasetId,\s*analysis\.filename\)/);
  assert.match(dashboard, /\/datasets\/\$\{encodeURIComponent\(result\.dataset_id\)\}/);
  assert.match(pages, /href=\{`\/datasets\/\$\{item\.id\}`\}/);
  assert.match(pages, /href=\{`\/datasets\/\$\{x\.dataset_id\}/);
  assert.match(nav, /href:\s*"\/new-analysis"/);
});

test("AI chat identifies deterministic Pandas analysis and keeps LLM status separate", () => {
  const chat = readFileSync(join(sourceRoot, "components/ai-analyst-chat.tsx"), "utf8");
  const types = readFileSync(join(sourceRoot, "types/dataset.ts"), "utf8");
  const api = readFileSync(join(sourceRoot, "lib/api.ts"), "utf8");
  assert.match(chat, /Pandas Data Analysis Assistant/);
  assert.match(chat, /No generative AI is used/);
  assert.match(chat, /not been tested or used by this assistant/);
  assert.match(chat, /askAnalyst\(datasetId, prompt\)/);
  assert.match(api, /body\.append\("dataset_id", datasetId\)/);
  assert.match(types, /provider: "deterministic_pandas"/);
  assert.doesNotMatch(chat, /third-party cloud AI provider/);
});

test("dashboard reports runtime storage mode and opens chat against selected dataset", () => {
  const dashboard = readFileSync(join(sourceRoot, "components/dashboard.tsx"), "utf8");
  assert.match(dashboard, /\/runtime\/status/);
  assert.match(dashboard, /storageStatus\.dataset_storage/);
  assert.match(dashboard, /datasetId=\{datasetId!\}/);
  assert.doesNotMatch(dashboard, /Local data workspace/);
});
