import { expect, test } from "@playwright/test";

test.describe("Homepage — landing content", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("renders the page title and personal-project framing", async ({ page }) => {
    await expect(page).toHaveTitle(/bitcoin|hamdaan/i);
    // The dissertation badge was removed in Nov 2026; the opening paragraph
    // now leads with the "ongoing personal project" framing instead.
    await expect(
      page.getByText(/ongoing personal project/i)
    ).toBeVisible();
  });

  test("renders the main heading", async ({ page }) => {
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: /machine learning for bitcoin price prediction/i,
      })
    ).toBeVisible();
  });

  test("shows the four dissertation metric tiles with correct values", async ({
    page,
  }) => {
    await expect(page.getByText("0.787", { exact: true })).toBeVisible();
    await expect(page.getByText("$1,300", { exact: true })).toBeVisible();
    await expect(page.getByText("2.16%", { exact: true })).toBeVisible();
    await expect(page.getByText("Dataset", { exact: true })).toBeVisible();
    await expect(page.getByText("tweets", { exact: true })).toBeVisible();
  });

  test("repo link points to the public GitHub repo", async ({ page }) => {
    const repoLink = page.getByRole("link", { name: /repo/i }).first();
    await expect(repoLink).toBeVisible();
    await expect(repoLink).toHaveAttribute(
      "href",
      /github\.com\/HamdaanCode\/BTC-Sentiment-Live/
    );
    await expect(repoLink).toHaveAttribute("target", "_blank");
  });

  test("Phase 2 anchor link jumps to the Phase 2 section", async ({ page }) => {
    const anchor = page.getByRole("link", { name: /phase 2 progress/i });
    await expect(anchor).toHaveAttribute("href", "#phase-2");

    await anchor.click();
    await expect(page).toHaveURL(/#phase-2$/);
    await expect(
      page.getByRole("heading", { name: /taking it live in 2026/i })
    ).toBeInViewport();
  });

  test("Research design block renders both experiments", async ({ page }) => {
    await expect(
      page.getByRole("heading", { name: /research design/i })
    ).toBeVisible();
    await expect(page.getByText(/Experiment A/i)).toBeVisible();
    await expect(page.getByText(/time regime generalisation/i)).toBeVisible();
    await expect(page.getByText(/Experiment B/i)).toBeVisible();
    await expect(page.getByText(/live 2026 Reddit pipeline/i)).toBeVisible();
  });

  test("Today's signals block renders four live metric tiles", async ({
    page,
  }) => {
    await expect(
      page.getByRole("heading", { name: /today.*signals/i })
    ).toBeVisible();
    // Metric labels — values come from the daily ingest JSON so we don't
    // assert specific numbers (they change daily), only that each tile is
    // present with its label.
    await expect(page.getByText("BTC close", { exact: true })).toBeVisible();
    await expect(page.getByText("r/Bitcoin", { exact: true })).toBeVisible();
    await expect(page.getByText("Posts scored", { exact: true })).toBeVisible();
    await expect(page.getByText("BTC rows", { exact: true })).toBeVisible();
  });

  test("L3 row shows Reddit as a shipped second sentiment source", async ({ page }) => {
    await expect(
      page.getByText(/Reddit added as a second live sentiment source/i)
    ).toBeVisible();
  });
});
