import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { access } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(here, "..");

async function read(relativePath) {
  return readFile(path.join(siteRoot, relativePath), "utf8");
}

test("ships a dependency-free static portfolio entry point", async () => {
  const [html, css, packageJson, vercelJson] = await Promise.all([
    read("index.html"),
    read("styles.css"),
    read("package.json"),
    read("vercel.json"),
  ]);

  assert.match(html, /<main\b/);
  assert.match(html, /styles\.css/);
  assert.match(html, /app\.js/);
  assert.doesNotMatch(html, /<(?:script|link)[^>]+https?:\/\//i);
  assert.doesNotMatch(css, /@import\s+url\(["']?https?:\/\//i);
  assert.deepEqual(JSON.parse(packageJson).dependencies ?? {}, {});
  assert.equal(JSON.parse(vercelJson).cleanUrls, true);
});

test("includes the verified result figures and evidence-qualified metrics", async () => {
  const html = await read("index.html");

  await Promise.all([
    access(path.join(siteRoot, "assets", "rre_comparison.png")),
    access(path.join(siteRoot, "assets", "clustering_comparison.png")),
  ]);

  for (const value of ["0.407", "0.185", "0.357", "0.276", "0.364", "0.557", "0.670", "0.178", "0.304", "0.280", "0.155", "0.196"]) {
    assert.match(html, new RegExp(value.replace(".", "\\.")));
  }
  assert.match(html, /historical experiment/i);
  assert.match(html, /asymmetric/i);
});

test("provides equivalent English and Chinese content plus persistent language behavior", async () => {
  const [html, app] = await Promise.all([read("index.html"), read("app.js")]);

  for (const section of ["problem", "method", "evidence", "reproducibility", "limitations", "collaboration"]) {
    assert.match(html, new RegExp(`id="${section}"`));
  }
  assert.match(html, /data-en=/);
  assert.match(html, /data-zh=/);
  assert.match(html, /aria-label="Language"/);
  assert.match(app, /localStorage/);
  assert.match(app, /document\.documentElement\.lang/);
  assert.match(app, /meta\[name="description"\]/);
  assert.match(app, /document\.title/);
});

test("links to the report and GitHub without prohibited personal or institutional details", async () => {
  const html = await read("index.html");
  const prohibited = [
    /university/i,
    /course/i,
    /assignment/i,
    /grade/i,
    /student id/i,
    /@[\w.-]+\.[a-z]{2,}/i,
    /C:\\/,
  ];

  assert.match(html, /href="\.\.\/docs\/robust_nmf_technical_report\.pdf"/);
  assert.match(html, /href="https:\/\/github\.com\/666junyichen\/robust-nmf-face-reconstruction"/);
  for (const pattern of prohibited) {
    assert.doesNotMatch(html, pattern);
  }
});

test("includes responsive and reduced-motion accessibility safeguards", async () => {
  const css = await read("styles.css");

  assert.match(css, /@media\s*\([^)]*max-width/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(css, /:focus-visible/);
});
