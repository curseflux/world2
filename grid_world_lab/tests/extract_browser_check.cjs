// Optional: NODE_PATH must include Playwright; pass an existing extraction.html.
const assert = require('node:assert/strict');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1350, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
  assert.equal(await page.locator('#truth svg').count(), 1);
  assert.equal(await page.locator('#inferred svg').count(), 1);
  assert.ok(await page.locator('#rows tr').count() > 0);
  await page.locator('#rows tr').first().click();
  assert.ok(await page.locator('#branches tr').count() > 0);
  await page.locator('#direction').selectOption('S');
  await page.locator('#matched').check();
  await page.locator('#kind').selectOption('generated_invalid');
  assert.equal(await page.locator('#rows tr').count(), 0);
  await page.locator('#matched').uncheck();
  await page.locator('#kind').selectOption('reference');
  await page.locator('#source').fill('999999');
  await page.locator('#source').dispatchEvent('change');
  assert.equal(await page.locator('#rows tr').count(), 0);
  await page.locator('#source').fill('');
  await page.locator('#source').dispatchEvent('change');
  await page.setViewportSize({width: 430, height: 900});
  if (process.argv[3]) await page.screenshot({path: process.argv[3], fullPage: true});
  assert.deepEqual(errors, []);
  await browser.close();
  console.log('Extraction viewer controls passed; no browser errors.');
})().catch(error => {console.error(error); process.exit(1);});
