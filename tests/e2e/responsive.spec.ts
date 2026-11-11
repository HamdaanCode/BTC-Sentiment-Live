import { expect, test } from "@playwright/test";

test.describe("Responsive layout", () => {
  test("no horizontal scroll on mobile viewport (375px)", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/");

    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));

    expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth + 1);
  });

  test("Today's signals block is reachable on mobile", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/");

    await page.getByRole("link", { name: /today.*signals/i }).click();
    await expect(
      page.getByText(/today.*signals/i).first()
    ).toBeInViewport();
  });
});
