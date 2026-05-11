import sys
import random
import time
import csv
import os
import re
from datetime import datetime, timezone, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
try:
    from playwright_stealth import stealth_sync
except ImportError:
    stealth_sync = None

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
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--disable-extensions",
        ],
    )

    context = browser.new_context(
        viewport=None,
        locale="ar-EG",
        timezone_id="Africa/Cairo",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        extra_http_headers={
            "Accept-Language": "ar-EG,ar;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
        },
    )

    page = context.new_page()

    if stealth_sync:
        stealth_sync(page)
    else:
        # Fallback if stealth is not installed
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

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
    Each product has 2 <a> tags sharing the same href:
      - one wraps the product image
      - one wraps the product name text
    We group them by href to avoid duplicates, then extract:
      - name  : from the link that contains visible text
      - price : from the card container innerText (regex)
      - image : from the link that contains an <img>
      - discount: from the card innerText (regex)
    """
    today    = datetime.today()
    date_str = today.strftime("%Y-%m-%d")
    data     = []

    try:
        all_links = page.query_selector_all(PRODUCT_LINK_SEL)
    except Exception:
        return data

    # Group links by href: href -> {name_link, img_link, card_el}
    products_map = {}
    for link in all_links:
        try:
            href = link.get_attribute("href") or ""
            if not href or "/p/" not in href:
                continue

            link_text = link.inner_text().strip()
            has_text  = bool(link_text and len(link_text) > 3)
            has_img   = link.query_selector("img") is not None

            if href not in products_map:
                products_map[href] = {"name_link": None, "img_link": None, "card_el": None}

            if has_text and not products_map[href]["name_link"]:
                products_map[href]["name_link"] = link
            if has_img and not products_map[href]["img_link"]:
                products_map[href]["img_link"] = link

            # Walk up the DOM to find the card container
            if products_map[href]["card_el"] is None:
                el = link
                for _ in range(5):
                    parent = el.evaluate_handle("el => el.parentElement")
                    try:
                        children = parent.evaluate("el => el.children.length")
                        tag      = parent.evaluate("el => el.tagName")
                        if tag == "LI" or children >= 3:
                            products_map[href]["card_el"] = parent
                            break
                    except Exception:
                        break
                    el = parent

        except Exception:
            continue

    # Extract data for each unique product
    for href, info in products_map.items():
        try:
            name_link = info["name_link"]
            img_link  = info["img_link"]
            card_el   = info["card_el"]

            # Name
            product_name = ""
            if name_link:
                product_name = name_link.inner_text().strip()
            if not product_name:
                continue

            # Price: parse from full card innerText
            price = ""
            if card_el:
                try:
                    card_text = card_el.evaluate("el => el.innerText")
                    matches = re.findall(r"\b\d+(?:[.,]\d{1,2})?\b", card_text)
                    for m in matches:
                        val = float(m.replace(",", "."))
                        if val > 1.0:
                            price = m.replace(",", ".")
                            break
                except Exception:
                    pass

            # Discount
            discount = ""
            if card_el:
                try:
                    card_text2 = card_el.evaluate("el => el.innerText")
                    disc_match = re.search(
                        r"\u062e\u0635\u0645\s*\d+%?|\d+%\s*\u062e\u0635\u0645|Save\s*\d+%?",
                        card_text2,
                    )
                    if disc_match:
                        discount = disc_match.group(0).strip()
                except Exception:
                    pass

            # Image
            image_url = ""
            if img_link:
                try:
                    img_el = img_link.query_selector("img")
                    if img_el:
                        image_url = img_el.get_attribute("src") or ""
                except Exception:
                    pass

            if discount:
                product_name = f"{product_name} ({discount})"

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
        "Vegetables & Fruits":    "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1660000",
        "Dairy products & eggs":  "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1630000",
        "Supermarket":            "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1700000",
        "Beverages":              "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1500000",
        "Baby Products":          "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1000000",
        "Frozen Foods":           "https://www.carrefouregypt.com/mafegy/ar/c/FEGY6000000",
        "Organic & Health Foods": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1200000",
        "Bakery":                 "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1610000",
        "Cleaning Tools":         "https://www.carrefouregypt.com/mafegy/ar/c/NFEGY3000000",
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
