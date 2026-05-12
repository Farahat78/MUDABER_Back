import sys
import random
import time
import csv
import os
import re
from datetime import datetime, timezone, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ── Fix encoding ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ── Output path ──
scraped_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "scraped"
)
os.makedirs(scraped_dir, exist_ok=True)

egypt_tz = timezone(timedelta(hours=3))
OUTPUT_FILE = os.path.join(
    scraped_dir,
    f"carrefour_products-{datetime.now(egypt_tz).strftime('%Y-%m-%d')}.csv"
)

PRODUCT_LINK_SEL = "a[href*='/p/']"


# ── Utils ──
def human_sleep(a=0.8, b=2.5):
    time.sleep(random.uniform(a, b))


def week_of_month(date):
    return (date.day - 1) // 7 + 1


def random_mouse_move(page):
    try:
        page.mouse.move(
            random.randint(100, 900),
            random.randint(100, 700),
        )
    except Exception:
        pass


# ── SAFE NAVIGATION (IMPORTANT FIX) ──
def safe_goto(page, url, retries=3):
    for i in range(retries):
        try:
            page.goto(url, timeout=90000, wait_until="domcontentloaded")
            return True
        except Exception as e:
            print(f"[RETRY {i+1}] goto failed: {e}")
            time.sleep(5)
    return False


# ── Browser ──
def launch_browser(p):
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ],
    )

    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        locale="ar-EG",
        timezone_id="Africa/Cairo",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        extra_http_headers={
            "Accept-Language": "ar-EG,ar;q=0.9,en-US;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        },
    )

    page = context.new_page()
    return browser, context, page


# ── Load products ──
def load_all_products(page):
    prev = 0
    streak = 0

    for _ in range(120):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        human_sleep(2, 4)

        try:
            links = page.query_selector_all(PRODUCT_LINK_SEL)
            hrefs = {
                l.get_attribute("href")
                for l in links
                if l.get_attribute("href")
            }

            curr = len(hrefs)

            if curr == prev:
                streak += 1
                if streak >= 5:
                    break
            else:
                streak = 0
                prev = curr

        except Exception:
            break


# ── Extract products ──
def extract_products(page, main_category, sub_category):
    today = datetime.today()
    date_str = today.strftime("%Y-%m-%d")

    data = []

    try:
        page.wait_for_selector(PRODUCT_LINK_SEL, timeout=15000)
        links = page.query_selector_all(PRODUCT_LINK_SEL)
    except Exception:
        return []

    products_map = {}

    for l in links:
        try:
            href = l.get_attribute("href")
            if not href:
                continue

            if href.startswith("/"):
                href = "https://www.carrefouregypt.com" + href

            products_map.setdefault(href, []).append(l)
        except:
            continue

    for href, items in products_map.items():
        try:
            link = items[0]

            name = link.inner_text().strip()
            if not name:
                continue

            parent_text = ""
            try:
                parent = link.evaluate_handle("el => el.parentElement")
                if parent:
                    parent_text = parent.evaluate("el => el.innerText")
            except:
                pass

            price = ""
            m = re.search(r"(\d+)\s*[\n.]+\s*(\d+)\s*EGP", parent_text)
            if m:
                price = f"{m.group(1)}.{m.group(2)}"
            else:
                m = re.search(r"(\d+(?:[.,]\d+)?)\s*EGP", parent_text)
                if m:
                    price = m.group(1)

            discount = ""
            d = re.search(r"\d+%", parent_text)
            if d:
                discount = d.group(0)

            img = ""
            img_el = link.query_selector("img")
            if img_el:
                img = img_el.get_attribute("src") or ""

            data.append([
                main_category,
                sub_category,
                name,
                price,
                discount,
                img,
                week_of_month(today),
                "Carrefour Egypt",
                date_str
            ])

        except:
            continue

    return data


# ── Sub categories ──
def get_sub_buttons(page):
    try:
        btns = page.query_selector_all("button[class*='text-primary'][class*='flex-col']")
        valid, names = [], []

        for b in btns:
            try:
                if not b.is_visible():
                    continue

                t = b.inner_text().strip()
                if not t:
                    continue

                if "الجميع" in t or "all" in t.lower():
                    continue

                valid.append(b)
                names.append(t)

            except:
                continue

        return valid, names

    except:
        return [], []


# ── MAIN ──
def main():
    categories = {
        "Fresh Foods": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1600000",
        "Supermarket": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1700000",
        "Vegetables & Fruits": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1660000",
        "Dairy products & eggs": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1630000",
        "Beverages": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1500000",
        "Frozen Foods": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY6000000",
        "Organic & Health Foods": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1200000",
        "Bakery": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1610000",
        "Cleaning Tools": "https://www.carrefouregypt.com/mafegy/ar/c/NFEGY3000000",
        "Baby Products": "https://www.carrefouregypt.com/mafegy/ar/c/FEGY1000000",
        "School Supplies": "https://www.carrefouregypt.com/mafegy/ar/c/NFEGY1300000",
    }

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "main_category", "sub_category", "product_name",
            "price", "discount", "image_url",
            "week_of_month", "source", "date"
        ])

        with sync_playwright() as p:
            for cat, url in categories.items():
                print("\n" + "="*60)
                print("[CAT]", cat)
                print("="*60)

                browser, context, page = launch_browser(p)

                try:
                    if not safe_goto(page, url):
                        print("[SKIP] failed navigation")
                        continue

                    human_sleep(5, 8)

                    subs, names = get_sub_buttons(page)
                    print("[SUB]", names)

                    if not names:
                        try:
                            page.wait_for_selector(PRODUCT_LINK_SEL, timeout=30000)
                        except:
                            print("[WARN] no products")
                            continue

                        load_all_products(page)
                        data = extract_products(page, cat, "All")
                        writer.writerows(data)
                        continue

                    for i, sub in enumerate(names):
                        try:
                            if not safe_goto(page, url):
                                continue

                            subs, names = get_sub_buttons(page)
                            if i >= len(subs):
                                continue

                            btn = subs[i]
                            btn.click()
                            human_sleep(3, 5)

                            try:
                                page.wait_for_selector(PRODUCT_LINK_SEL, timeout=30000)
                            except:
                                continue

                            load_all_products(page)
                            data = extract_products(page, cat, sub)
                            writer.writerows(data)

                            print(f"[OK] {cat} / {sub}: {len(data)}")

                        except Exception as e:
                            print("[ERR]", e)

                finally:
                    try:
                        context.close()
                        browser.close()
                    except:
                        pass

    print("\n[DONE]", OUTPUT_FILE)


if __name__ == "__main__":
    main()
