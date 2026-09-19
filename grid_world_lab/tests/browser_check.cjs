// Optional UI integration check: npm install --no-save playwright; npx playwright install chromium
// node tests/browser_check.cjs runs/sparse-demo/report.html runs/sparse-demo/report-preview.png
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
  const payload = await page.locator('#experiment-data').textContent().then(JSON.parse);
  const count = payload.samples.length;
  assert.equal(await page.locator('#true-map svg').count(), 1);
  assert.equal(await page.locator('#recon-map svg').count(), 1);
  assert.equal(await page.locator('#selection-badge').textContent(), `${count} selected`);
  await page.locator('#sample-count').fill('0');
  assert.equal(await page.locator('#selection-badge').textContent(), '0 selected');
  await page.locator('#sample-subset').fill('1, 3-4');
  assert.equal(await page.locator('#selection-badge').textContent(), '3 selected');
  await page.locator('#sample-subset').fill(payload.samples[0].id);
  assert.equal(await page.locator('#selection-badge').textContent(), '1 selected');
  await page.locator('#sample-subset').fill('999999');
  assert.match(await page.locator('#subset-status').textContent(), /out of range/);
  await page.locator('#clear-subset').click();
  await page.locator('#cohort').selectOption('unseen');
  assert.equal(await page.locator('#selection-badge').textContent(), `${payload.samples.filter(s=>s.cohort==='unseen').length} selected`);
  await page.locator('#show-illegal').uncheck();
  await page.locator('#min-confidence').fill('0.9');
  await page.locator('#frequency-nodes').click();
  assert.match(await page.locator('#frequency-head').textContent(), /Post-action probe visits/);
  const downloaded = page.waitForEvent('download');
  await page.locator('#export-events').click();
  const download = await downloaded;
  assert.equal(download.suggestedFilename(), 'selected-events.csv');
  await page.locator('#cohort').selectOption('all');
  await page.locator('#show-illegal').check();
  await page.locator('#min-confidence').fill('0');
  await page.locator('#frequency-edges').click();
  await page.screenshot({path: path.resolve(process.argv[3] || 'report-preview.png'), fullPage: true});
  await page.setViewportSize({width: 420, height: 900});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth), true, 'Mobile viewport should not overflow horizontally');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({samples: count, browserErrors: errors, checks: 'maps, zero/all/subset/IDs, cohort, filters, frequency, CSV, responsive layout'}));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
