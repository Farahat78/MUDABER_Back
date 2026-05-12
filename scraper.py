import sys
import random
import time
import csv
import os
import re
from datetime import datetime, timezone, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Force UTF-8 output on Windows to avoid UnicodeEncodeError on cp1252 terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Save to data/scraped/ ──
scraped_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "scraped"
)
os.makedirs(scraped_dir, exist_ok=True)
# Egypt timezone (UTC+3)
egypt_tz = timezone(timedelta(hours=3))
OUTPUT_FILE = os.path.join(
    scraped_dir,
    f"carrefour_products-{datetime.now(egypt_tz).strftime('%Y-%m-%d')}.csv"
)

# ── Updated selector: every product page link uses /p/ in the href ──
PRODUCT_LINK_SEL = "a[href*='/p/']"


# ─── Helpers ─────────────────────────────────────────────────────────────────
def human_sleep(a=0.8, b=2.5):
    time.sleep(random.uniform(a, b))


def week_of_month(date):
    return (date.day - 1) // 7 + 1


def random_mouse_move(page):
    """Simulate a random human-like mouse movement."""
    try:
        page.mouse.move(
            random.randint(100, 900),
            random.randint(100, 700),
        )
    except Exception:
        pass


# ─── Browser launch ──────────────────────────────────────────────────────────
def launch_browser(p):
    browser = p.firefox.launch(
        headless=True,
        args=[
            "--start-maximized",
        ],
    )

    context = browser.new_context(
        viewport=None,
        locale="ar-EG",
        timezone_id="Africa/Cairo",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        extra_http_headers={
            "Accept-Language": "ar-EG,ar;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
        },
    )

    page = context.new_page()

    return browser, context, page


# ─── Load all products (infinite scroll) ──────────────────────────────────────
def load_all_products(page):
    """Scroll down to trigger infinite loading until all products are loaded."""
    prev_count = 0
    no_change_streak = 0

    for _ in range(150):  # High limit to ensure we get everything
        # Scroll to the very bottom to trigger the infinite load
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        random_mouse_move(page)
        human_sleep(2.0, 4.0)

        # Scroll up slightly then back down to help trigger lazy loading if stuck
        page.mouse.wheel(0, -500)
        human_sleep(0.5, 1.0)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        human_sleep(1.0, 2.0)

        # Check if new products loaded
        try:
            hrefs = {
                lnk.get_attribute("href") or ""
                for lnk in page.query_selector_all(PRODUCT_LINK_SEL)
                if lnk.get_attribute("href")
            }
            current_count = len(hrefs)

            if current_count == prev_count:
                no_change_streak += 1
                if no_change_streak >= 5: # Wait for ~5 attempts without new products before giving up
                    break
            else:
                no_change_streak = 0
                prev_count = current_count
        except Exception:
            break


# ─── Extract products from current page ──────────────────────────────────────
def extract_products(page, main_category, sub_category):
    """
    Extract products using stable data-testids and CSS selectors.
    Targets [data-testid="product_card"] to ensure we don't mix up data between cards.
    """
    today    = datetime.today()
    date_str = today.strftime("%Y-%m-%d")
    data     = []

    try:
        # Wait for at least one product card to be present
        page.wait_for_selector('[data-testid="product_card"]', timeout=5000)
        cards = page.query_selector_all('[data-testid="product_card"]')
    except Exception:
        # Fallback to old link-based detection if data-testid is missing
        cards = []

    if not cards:
        # Fallback: try to find anything that looks like a product link
        all_links = page.query_selector_all(PRODUCT_LINK_SEL)
        # (Rest of the old grouping logic would go here, but let's prioritize the new stable one)
        # For now, if cards are empty, we return empty so the main loop can retry or fail gracefully
        return data

    for card in cards:
        try:
            # 1. Name: Usually an <a> tag with product title
            # Target links with /p/ inside the card
            name_el = card.query_selector("a[href*='/p/']")
            product_name = ""
            if name_el:
                product_name = name_el.inner_text().strip()
            
            if not product_name or len(product_name) < 3:
                continue

            # 2. Price: Target the price container specifically
            # Current price usually has large numbers. 
            # We look for the container that has 'EGP' or 'ج.م'
            price = ""
            # Selector for the main price container (contains whole, dot, decimals)
            price_container = card.query_selector("div.flex.flex-col.justify-center.items-start")
            
            if price_container:
                price_text = price_container.inner_text()
                # Carrefour Egypt format: "34\n.\n99\nEGP"
                # We want to find the first sequence of [digits] [.] [digits]
                # Or just [digits] [digits]
                num_parts = re.findall(r"\d+", price_text)
                if len(num_parts) >= 2:
                    # Take first two as whole and decimal
                    price = f"{num_parts[0]}.{num_parts[1]}"
                elif len(num_parts) == 1:
                    price = num_parts[0]

            # 3. Discount & Badge
            discount = ""
            badge_el = card.query_selector("div[class*='badge'], span[class*='discount']")
            if badge_el:
                discount = badge_el.inner_text().strip()

            # 4. Image
            image_url = ""
            img_el = card.query_selector("img")
            if img_el:
                image_url = img_el.get_attribute("src") or ""

            # Check if name contains weight info we should append
            # (Carrefour often puts weight in a sibling element .text-xs.text-gray-500)
            weight_el = card.query_selector(".text-xs.text-gray-500")
            if weight_el:
                w_text = weight_el.inner_text().strip()
                if w_text and w_text not in product_name:
                    product_name = f"{product_name} ({w_text})"

            data.append([
                main_category,
                sub_category,
                product_name,
                price,
                discount,
                image_url,
                week_of_month(today),
                "Carrefour Egypt",
                date_str,
            ])

        except Exception:
            continue

    return data


# ─── Get subcategory buttons ─────────────────────────────────────────────────
def get_sub_buttons(page):
    """
    Detect subcategory filter buttons on the page.
    On the updated Carrefour Egypt site, subcategory buttons have circular icons
    and specific classes ('text-primary' and 'flex-col').
    Returns (list_of_button_elements, list_of_button_texts) excluding the first 'All'/'الجميع' button.
    """
    try:
        # The exact selector for the subcategory buttons
        all_btns = page.query_selector_all("button[class*='text-primary'][class*='flex-col']")
        valid = []
        texts = []

        for btn in all_btns:
            try:
                if not btn.is_visible():
                    continue
                txt = btn.inner_text().strip()
                if not txt:
                    continue

                # The user explicitly wants to skip "الجميع" (All)
                # Also skip delivery buttons and the Electronics floating button
                txt_lower = txt.lower()
                if "الجميع" in txt_lower or "all" == txt_lower:
                    continue
                if "الكترونيات" in txt_lower:
                    continue
                if "توصيل" in txt_lower or "delivery" in txt_lower:
                    continue

                valid.append(btn)
                texts.append(txt)
            except Exception:
                continue

        # Subcategory rows typically have multiple buttons
        if len(valid) > 0:
            return valid, texts

    except Exception:
        pass

    return [], []


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    categories = {
        "Fresh Foods":            "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1600000",
        "Supermarket":            "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1700000",
        "Vegetables & Fruits":    "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1660000",
        "Dairy products & eggs":  "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1630000",
        "Beverages":              "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1500000",
        "Frozen Foods":           "https://www.carrefouregypt.com/mafegy/ar/c/FEGY6000000",
        "Organic & Health Foods": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1200000",
        "Bakery":                 "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1610000",
        "Cleaning Tools":         "https://www.carrefouregypt.com/mafegy/ar/c/NFEGY3000000",
        "Baby Products":          "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1000000",
        "School Supplies":        "https://www.carrefouregypt.com/mafegy/ar/c/NFEGY1300000",
    }

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "main_category", "sub_category", "product_name",
            "price", "discount", "image_url",
            "week_of_month", "source", "date",
        ])

        with sync_playwright() as p:
            for main_cat, url in categories.items():
                print(f"\n{'='*55}")
                print(f"[CAT] {main_cat}")
                print(f"{'='*55}")

                browser, context, page = launch_browser(p)

                try:
                    # Load the category page
                    page.goto(url, timeout=90000, wait_until="domcontentloaded")
                    human_sleep(6, 10)
                    random_mouse_move(page)

                    # Detect subcategory buttons
                    sub_buttons, sub_names = get_sub_buttons(page)
                    print(f"   [SUB] Subcategories ({len(sub_names)}): {sub_names}")

                    # ── No subcategories: scrape the whole category page ──
                    if not sub_names:
                        print(f"   [WARN] No subcategories - scraping main category directly")
                        try:
                            page.wait_for_selector(PRODUCT_LINK_SEL, timeout=35000)
                        except PlaywrightTimeout:
                            print(f"   [ERR] No products found for {main_cat}")
                            context.close()
                            browser.close()
                            human_sleep(6, 9)
                            continue

                        load_all_products(page)
                        products = extract_products(page, main_cat, "All")
                        writer.writerows(products)
                        f.flush()
                        print(f"   [OK] {main_cat} / All: {len(products)} products")
                        context.close()
                        browser.close()
                        human_sleep(8, 13)
                        continue

                    # ── Scrape each subcategory ──
                    completed   = 0
                    retry_count = 0
                    MAX_RETRIES = 3

                    while completed < len(sub_names):
                        if retry_count >= MAX_RETRIES:
                            print(f"   [SKIP] '{sub_names[completed]}' after {MAX_RETRIES} retries")
                            completed   += 1
                            retry_count  = 0
                            continue

                        sub_name = sub_names[completed]

                        try:
                            # Reload category page and reselect subcategory
                            page.goto(url, timeout=90000, wait_until="domcontentloaded")
                            human_sleep(4, 7)
                            random_mouse_move(page)

                            buttons, names = get_sub_buttons(page)

                            if not buttons or completed >= len(buttons):
                                print(f"   [WARN] Button for '{sub_name}' not found, skipping")
                                completed   += 1
                                retry_count  = 0
                                continue

                            target_btn = buttons[completed]
                            target_btn.scroll_into_view_if_needed()
                            human_sleep(1, 2)
                            target_btn.click()
                            human_sleep(3, 5)
                            random_mouse_move(page)

                            # Wait for products to appear
                            try:
                                page.wait_for_selector(PRODUCT_LINK_SEL, timeout=35000)
                            except PlaywrightTimeout:
                                print(f"   [TIMEOUT] '{sub_name}' retry {retry_count+1}/{MAX_RETRIES}")
                                retry_count += 1
                                human_sleep(6, 10)
                                continue

                            load_all_products(page)
                            products = extract_products(page, main_cat, sub_name)
                            writer.writerows(products)
                            f.flush()

                            print(f"   [OK] {main_cat} / {sub_name}: {len(products)} products")
                            completed   += 1
                            retry_count  = 0
                            human_sleep(3, 6)

                        except PlaywrightTimeout:
                            print(f"   [TIMEOUT] '{sub_name}' retry {retry_count+1}/{MAX_RETRIES}")
                            retry_count += 1
                            human_sleep(8, 13)

                        except Exception as e:
                            print(f"   [ERR] '{sub_name}': {e}")
                            retry_count += 1
                            human_sleep(4, 7)

                except Exception as e:
                    print(f"   [FATAL] {main_cat}: {e}")

                finally:
                    try:
                        context.close()
                        browser.close()
                    except Exception:
                        pass

                human_sleep(10, 16)  # pause between categories

    print(f"\n[DONE] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
