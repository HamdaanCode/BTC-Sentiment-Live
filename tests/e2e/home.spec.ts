import { expect, test } from "@playwright/test";

test.describe("Homepage — landing content", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("renders the page title and intro", async ({ page }) => {
    await expect(page).toHaveTitle(/bitcoin|hamdaan/i);
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: /bitcoin-sentiment-live/i,
      })
    ).toBeVisible();
    await expect(
      page.getByText(/does what people say about Bitcoin/i)
    ).toBeVisible();
  });

  test("shows the four original dissertation metric tiles", async ({ page }) => {
    await expect(page.getByText("0.787", { exact: true })).toBeVisible();
    await expect(page.getByText("$1,300", { exact: true })).toBeVisible();
    await expect(page.getByText("2.16%", { exact: true })).toBeVisible();
    await expect(page.getByText("dataset", { exact: true })).toBeVisible();
    await expect(page.getByText("tweets", { exact: true })).toBeVisible();
  });

  test("repo link points to the public GitHub repo", async ({ page }) => {
    const repoLink = page.getByRole("link", { name: /github/i }).first();
    await expect(repoLink).toBeVisible();
    await expect(repoLink).toHaveAttribute(
      "href",
      /github\.com\/HamdaanCode\/BTC-Sentiment-Live/
    );
    await expect(repoLink).toHaveAttribute("target", "_blank");
  });

  test("today's signals anchor jumps to the live block", async ({ page }) => {
    const anchor = page.getByRole("link", { name: /today.*signals/i });
    await expect(anchor).toHaveAttribute("href", "#live");

    await anchor.click();
    await expect(page).toHaveURL(/#live$/);
    await expect(
      page.getByText(/today.*signals/i).first()
    ).toBeInViewport();
  });

  test("today's signals block renders four live metric tiles", async ({
    page,
  }) => {
    await expect(page.getByText(/today.*signals/i).first()).toBeVisible();
    await expect(page.getByText("btc close", { exact: true })).toBeVisible();
    await expect(page.getByText("r/bitcoin", { exact: true })).toBeVisible();
    await expect(page.getByText("posts scored", { exact: true })).toBeVisible();
    await expect(page.getByText("btc rows", { exact: true })).toBeVisible();
  });

  test("renders the three forum thread entries (original + two replies)", async ({
    page,
  }) => {
    await expect(page.getByText(/original post/i)).toBeVisible();
    await expect(page.getByText(/reply #1/i)).toBeVisible();
    await expect(page.getByText(/reply #2/i)).toBeVisible();
  });

  test("H1 2022 bear-market reply shows the real numbers", async ({ page }) => {
    await expect(
      page.getByRole("heading", { name: /does it still work in a bear market/i })
    ).toBeVisible();
    await expect(page.getByText("0.018", { exact: true })).toBeVisible();
    await expect(page.getByText("$3,670", { exact: true })).toBeVisible();
    await expect(page.getByText("60.4%", { exact: true })).toBeVisible();
  });

  test("coming-soon footer teases next work without naming it", async ({ page }) => {
    await expect(page.getByText(/more coming soon/i)).toBeVisible();
  });
});
