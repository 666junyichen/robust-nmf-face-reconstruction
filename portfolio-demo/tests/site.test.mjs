import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { access } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";

const here = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(here, "..");

async function read(relativePath) {
  return readFile(path.join(siteRoot, relativePath), "utf8");
}

function element(dataset = {}, attributes = {}) {
  const listeners = {};
  return {
    dataset,
    textContent: "",
    attributes: { ...attributes },
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    },
    getAttribute(name) {
      return this.attributes[name] ?? null;
    },
    addEventListener(name, handler) {
      listeners[name] = handler;
    },
    click() {
      listeners.click?.();
    },
  };
}

async function runApp({ savedLanguage = null, getThrows = false, setThrows = false } = {}) {
  const app = await read("app.js");
  const heading = element({ en: "Evidence", zh: "实验依据" });
  const labelled = element(
    { enAriaLabel: "Project summary", zhAriaLabel: "项目摘要" },
    { "aria-label": "Project summary" },
  );
  const image = element(
    { enAlt: "RRE comparison", zhAlt: "RRE 对比" },
    { alt: "RRE comparison" },
  );
  const englishButton = element({ language: "en" }, { "aria-pressed": "true" });
  const chineseButton = element({ language: "zh" }, { "aria-pressed": "false" });
  const metadata = new Map(
    [
      ['meta[name="description"]', "description"],
      ['meta[property="og:title"]', "og:title"],
      ['meta[property="og:description"]', "og:description"],
      ['meta[property="og:locale"]', "og:locale"],
      ['meta[name="twitter:title"]', "twitter:title"],
      ['meta[name="twitter:description"]', "twitter:description"],
    ].map(([selector, name]) => [selector, element({}, { content: name })]),
  );
  const values = new Map();
  if (savedLanguage !== null) values.set("robust-nmf-language", savedLanguage);

  const document = {
    documentElement: { lang: "en" },
    title: "",
    querySelectorAll(selector) {
      if (selector === "[data-en][data-zh]") return [heading];
      if (selector === "[data-language]") return [englishButton, chineseButton];
      if (selector === "[data-en-aria-label][data-zh-aria-label]") return [labelled];
      if (selector === "[data-en-alt][data-zh-alt]") return [image];
      return [];
    },
    querySelector(selector) {
      return metadata.get(selector) ?? null;
    },
  };
  const localStorage = {
    getItem(key) {
      if (getThrows) throw new Error("blocked get");
      return values.get(key) ?? null;
    },
    setItem(key, value) {
      if (setThrows) throw new Error("blocked set");
      values.set(key, value);
    },
  };
  const window = {
    matchMedia: () => ({ matches: true }),
  };

  vm.runInNewContext(app, { document, localStorage, window });
  return { document, heading, labelled, image, englishButton, chineseButton, metadata, values };
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
  assert.doesNotMatch(html, /<script[^>]+https?:\/\//i);
  assert.doesNotMatch(html, /<link[^>]+rel="stylesheet"[^>]+https?:\/\//i);
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

test("executes default English state and persists it", async () => {
  const state = await runApp();

  assert.equal(state.document.documentElement.lang, "en");
  assert.equal(state.document.title, "Robust NMF — Reconstruction Dossier");
  assert.equal(state.heading.textContent, "Evidence");
  assert.equal(state.englishButton.getAttribute("aria-pressed"), "true");
  assert.equal(state.chineseButton.getAttribute("aria-pressed"), "false");
  assert.equal(state.labelled.getAttribute("aria-label"), "Project summary");
  assert.equal(state.image.getAttribute("alt"), "RRE comparison");
  assert.equal(state.values.get("robust-nmf-language"), "en");
  assert.equal(
    state.metadata.get('meta[property="og:locale"]').getAttribute("content"),
    "en_AU",
  );
});

test("switches to Chinese and synchronizes metadata and accessible names", async () => {
  const state = await runApp();
  state.chineseButton.click();

  assert.equal(state.document.documentElement.lang, "zh-CN");
  assert.equal(state.document.title, "鲁棒 NMF — 重建研究档案");
  assert.equal(state.heading.textContent, "实验依据");
  assert.equal(state.englishButton.getAttribute("aria-pressed"), "false");
  assert.equal(state.chineseButton.getAttribute("aria-pressed"), "true");
  assert.equal(state.labelled.getAttribute("aria-label"), "项目摘要");
  assert.equal(state.image.getAttribute("alt"), "RRE 对比");
  assert.equal(state.values.get("robust-nmf-language"), "zh");
  assert.equal(
    state.metadata.get('meta[property="og:title"]').getAttribute("content"),
    "鲁棒 NMF — 重建研究档案",
  );
  assert.equal(
    state.metadata.get('meta[name="twitter:description"]').getAttribute("content"),
    "一份双语重建研究档案，记录四人团队完成的鲁棒 NMF 人脸重建项目。",
  );
});

test("normalizes an invalid saved preference to English", async () => {
  const state = await runApp({ savedLanguage: "fr" });

  assert.equal(state.document.documentElement.lang, "en");
  assert.equal(state.heading.textContent, "Evidence");
  assert.equal(state.values.get("robust-nmf-language"), "en");
});

test("keeps working when localStorage get or set is blocked", async () => {
  const getBlocked = await runApp({ getThrows: true });
  const setBlocked = await runApp({ setThrows: true });

  assert.equal(getBlocked.document.documentElement.lang, "en");
  assert.equal(setBlocked.document.documentElement.lang, "en");
  setBlocked.chineseButton.click();
  assert.equal(setBlocked.document.documentElement.lang, "zh-CN");
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

  assert.match(html, /href="assets\/robust_nmf_technical_report\.pdf"/);
  assert.match(html, /href="https:\/\/github\.com\/666junyichen\/robust-nmf-face-reconstruction"/);
  assert.match(html, /MIT License covers only newly organized project source code and configuration/);
  assert.match(html, /report and figures[\s\S]*not MIT-licensed/);
  for (const pattern of prohibited) {
    assert.doesNotMatch(html, pattern);
  }
});

test("all local links and sources resolve inside the deployed site root", async () => {
  const html = await read("index.html");
  const references = [...html.matchAll(/\b(?:href|src)="([^"]+)"/g)].map((match) => match[1]);

  for (const reference of references) {
    if (
      reference.startsWith("#") ||
      reference.startsWith("https://") ||
      reference.startsWith("http://")
    ) {
      continue;
    }
    const resolved = path.resolve(siteRoot, reference);
    assert.ok(
      resolved === siteRoot || resolved.startsWith(`${siteRoot}${path.sep}`),
      `${reference} escapes the deployed site root`,
    );
    await access(resolved);
  }
});

test("declares bilingual metadata and accessible names for labelled media and controls", async () => {
  const html = await read("index.html");
  const labelledTags = [...html.matchAll(/<[^>]+\baria-label="[^"]+"[^>]*>/g)].map(
    (match) => match[0],
  );
  const imageTags = [...html.matchAll(/<img\b[^>]*>/g)].map((match) => match[0]);

  assert.match(html, /rel="canonical" href="https:\/\/robust-nmf-face-reconstruction\.vercel\.app\/"/);
  assert.match(html, /property="og:title"/);
  assert.match(html, /name="twitter:card"/);
  for (const tag of labelledTags) {
    assert.match(tag, /data-en-aria-label=/);
    assert.match(tag, /data-zh-aria-label=/);
  }
  for (const tag of imageTags) {
    assert.match(tag, /data-en-alt=/);
    assert.match(tag, /data-zh-alt=/);
    assert.match(tag, /\bwidth="\d+"/);
    assert.match(tag, /\bheight="\d+"/);
  }
});

test("configures security headers for every Vercel route", async () => {
  const config = JSON.parse(await read("vercel.json"));
  const catchAll = config.headers?.find((entry) => entry.source === "/(.*)");
  const headers = Object.fromEntries(
    (catchAll?.headers ?? []).map(({ key, value }) => [key.toLowerCase(), value]),
  );

  assert.match(headers["content-security-policy"] ?? "", /default-src 'self'/);
  assert.match(headers["content-security-policy"] ?? "", /img-src 'self' data:/);
  assert.match(headers["content-security-policy"] ?? "", /style-src 'self'/);
  assert.match(headers["content-security-policy"] ?? "", /script-src 'self'/);
  assert.match(headers["content-security-policy"] ?? "", /object-src 'none'/);
  assert.match(headers["content-security-policy"] ?? "", /base-uri 'self'/);
  assert.match(headers["content-security-policy"] ?? "", /frame-ancestors 'none'/);
  assert.equal(headers["x-content-type-options"], "nosniff");
  assert.equal(headers["referrer-policy"], "strict-origin-when-cross-origin");
  assert.ok(headers["permissions-policy"]);
});

test("includes responsive and reduced-motion accessibility safeguards", async () => {
  const css = await read("styles.css");

  assert.match(css, /@media\s*\([^)]*max-width/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(css, /:focus-visible/);
});
