// Ask every edge-case question through the real UI and record what a user sees:
// the verdict card text, the badge, console errors, how many graph nodes are
// shown, the time to the answer, and a screenshot per question.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const BASE = process.env.TARGET_URL || 'http://localhost:8768';
const OUT = process.env.OUT_DIR;                       // screenshots + results.jsonl
const ONLY = (process.env.ONLY || '').split(',').filter(Boolean);
const questions = JSON.parse(fs.readFileSync(process.env.QUESTIONS, 'utf8'))
  .filter(q => !ONLY.length || ONLY.includes(q.id));

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const results = fs.createWriteStream(path.join(OUT, 'results.jsonl'));
  const browser = await chromium.launch({ headless: process.env.PW_HEADLESS === 'true' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    let errors = [];
    page.on('console', m => { if (m.type() === 'error' && !m.text().includes('favicon')) errors.push(m.text()); });
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto(BASE);
    await page.getByText('Live', { exact: true }).waitFor({ timeout: 30000 });
    for (const { id, q } of questions) {
      errors = [];
      const box = page.getByPlaceholder('Describe the incident…');
      const t0 = Date.now();
      const answered = page.waitForResponse(r => r.url().endsWith('/api/ask'), { timeout: 130000 });
      await box.fill(q);
      await box.press('Enter');
      let status = 0, api = null;
      try { const resp = await answered; status = resp.status(); api = await resp.json(); } catch (e) { errors.push('no answer: ' + e.message); }
      const seconds = (Date.now() - t0) / 1000;
      // The step player replays each step (1.5 s each), then lands on the verdict.
      await page.locator('.metarow').waitFor({ timeout: 15000 }).catch(() => errors.push('no badge row'));
      await page.waitForFunction(() => typeof current !== 'undefined' && current && pos === current.out.steps.length,
                                 null, { timeout: 30000 }).catch(() => {});
      await page.waitForTimeout(600);
      const card = await page.locator('[data-pane="answer"]').innerText().catch(() => '');
      // What the card shows, read from the page (textContent, so collapsed parts count).
      const dom = await page.evaluate(() => {
        const t = sel => document.querySelector(sel)?.textContent.replace(/\s+/g, ' ').trim() || null;
        return {
          reasonline: t('.reasonline'),
          headline: t('.verdict .headline'),
          kind: t('.verdict .kind'),
          chain: [...document.querySelectorAll('.verdict .vrow')].map(r => r.textContent.replace(/\s+/g, ' ').trim()),
          pages: [...document.querySelectorAll('.verdict .page')].map(r => r.textContent.replace(/\s+/g, ' ').trim()),
          acts: [...document.querySelectorAll('.verdict [data-act]')].map(r => r.textContent.replace(/\s+/g, ' ').trim()),
          ladder: [...document.querySelectorAll('.rs')].map(r => r.textContent.replace(/\s+/g, ' ').trim()),
          notice: t('.verdict .notice'),
          full: t('details.fold .body'),
          badge: t('.metarow'),
          tabs: [...document.querySelectorAll('[data-tab]')].map(b => b.textContent.replace(/\s+/g, ' ').trim()),
        };
      });
      const graph = await page.evaluate(() => ({
        shown: cy.nodes().filter(n => n.visible()).length,
        services: cy.nodes('[kind = "service"]').length,
        lit: cy.nodes().filter(n => n.visible() && !n.hasClass('faded')).length,
        zoom: Math.round(cy.zoom() * 100) / 100 }));
      await page.screenshot({ path: path.join(OUT, `${id}.png`) });
      const row = { id, q, status, seconds, errors, graph, card, dom, api };
      results.write(JSON.stringify(row) + '\n');
      console.log(`${errors.length ? 'ERR ' : 'OK  '} ${id.padEnd(4)} ${String(status).padEnd(4)} ${seconds.toFixed(1)}s `
                  + `shown=${graph.shown}/${graph.services} lit=${graph.lit} ${errors.join(' | ').slice(0, 200)}`);
    }
  } finally {
    results.end();
    await browser.close();
  }
})();
