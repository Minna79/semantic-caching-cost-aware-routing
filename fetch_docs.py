import time
import requests
from pathlib import Path

OUT = Path("docs")
OUT.mkdir(exist_ok=True)

EN = ["Qanat", "Antikythera mechanism", "Voynich manuscript", "Tardigrade",
      "Svalbard Global Seed Vault", "Great Molasses Flood", "Krakatoa",
      "Tunguska event", "Rosetta Stone", "Library of Alexandria",
      "Panama Canal", "Transistor", "Enigma machine", "Ada Lovelace",
      "Kepler's laws of planetary motion"]

FA = ["حافظ", "سعدی", "تخت جمشید", "دریاچه ارومیه", "زلزله بم", "قنات",
      "نوروز", "فردوسی", "شاهنامه", "ابن سینا", "خوارزمی", "دماوند",
      "کویر لوت", "پاسارگاد", "چغازنبیل"]

# Wikimedia asks for a descriptive UA with a contact (repo URL is fine)
HEADERS = {"User-Agent": "rag-eval-project/1.0 (https://github.com/Minna79/semantic-caching-cost-aware-routing)"}

session = requests.Session()
session.headers.update(HEADERS)

def fetch(lang, title, retries=5):
    for attempt in range(retries):
        try:
            r = session.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={"action": "query", "prop": "extracts", "explaintext": 1,
                        "titles": title, "redirects": 1, "format": "json"},
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"  network error on {lang}:{title}: {e}")
            time.sleep(2 ** attempt)
            continue

        if r.status_code != 200:
            wait = int(r.headers.get("Retry-After", 2 ** attempt))
            print(f"  HTTP {r.status_code} on {lang}:{title}, retry in {wait}s")
            time.sleep(wait)
            continue

        try:
            pages = r.json()["query"]["pages"]
        except (ValueError, KeyError):
            print(f"  bad response on {lang}:{title}: {r.text[:150]!r}")
            time.sleep(2 ** attempt)
            continue

        page = next(iter(pages.values()))
        return page.get("extract", "")
    return None  # gave up

for lang, titles in [("en", EN), ("fa", FA)]:
    for i, t in enumerate(titles):
        path = OUT / f"{lang}_{i:02d}.txt"
        if path.exists():  # resume: don't refetch what you already have
            print(f"have {lang}:{t}")
            continue

        text = fetch(lang, t)
        if text is None:
            print(f"FAIL {lang}:{t} (gave up after retries)")
            continue
        if len(text) < 500:
            print(f"skip {lang}:{t} (not found or too short)")
            continue

        path.write_text(f"{t}\n\n{text}", encoding="utf-8")
        print(f"ok   {lang}:{t} ({len(text)} chars)")
        time.sleep(1)  # be polite between requests