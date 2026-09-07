import { expect, test, type Page } from "@playwright/test";

const chartCard = (page: Page, title: RegExp) =>
  page
    .locator("div")
    .filter({ has: page.getByRole("heading", { name: title }) })
    .first();

test.describe("Charts — Recharts SVG rendering", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");
  });

  test("page contains exactly three chart SVGs", async ({ page }) => {
    const svgs = page.locator("svg.recharts-surface");
    await expect(svgs).toHaveCount(3);
  });

  test("bitcoin price chart renders inside its card", async ({ page }) => {
    const card = chartCard(page, /bitcoin closing price/i);
    await expect(card.locator("svg.recharts-surface").first()).toBeVisible();
    await expect(card.getByText(/\$\d+k/).first()).toBeVisible();
  });

  test("sentiment chart legend lists all three model variants", async ({
    page,
  }) => {
    const card = chartCard(page, /daily sentiment.*three models/i);
    await expect(card.getByText(/A.*baseline VADER/i)).toBeVisible();
    await expect(card.getByText(/B.*crypto lexicon.*winner/i)).toBeVisible();
    await expect(card.getByText(/C.*intensity-weighted/i)).toBeVisible();
  });

  test("tweet-volume chart renders bar rectangles", async ({ page }) => {
    const card = chartCard(page, /tweet volume per day/i);
    const bars = card.locator("svg.recharts-surface path");
    await expect(bars.first()).toBeVisible();
    expect(await bars.count()).toBeGreaterThan(5);
  });
});
