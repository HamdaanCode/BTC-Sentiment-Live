import { expect, test } from "@playwright/test";

test.describe("Homepage — landing content", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("renders the page title and dissertation badge", async ({ page }) => {
    await expect(page).toHaveTitle(/bitcoin|dissertation|hamdaan/i);
    await expect(
      page.getByText(/BSc Dissertation.*Heriot-Watt.*First Class/i)
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
});
