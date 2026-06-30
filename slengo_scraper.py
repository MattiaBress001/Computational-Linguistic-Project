"""
slengo_scraper.py
Scraper completo per https://slengo.it
Fase 1: raccoglie tutti i termini dall'indice /browse
Fase 2: scarica definizioni ed esempi per ciascun termine

Dipendenze: pip install requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
import csv
import json
import time
import random
import re
import sys
from urllib.parse import unquote

# ── Configurazione ────────────────────────────────────────────────────────────
DELAY_MIN = 1.0      # pausa minima tra richieste (secondi)
DELAY_MAX = 2.5      # pausa massima
OUTPUT_CSV  = "slengo_dictionary.csv"
OUTPUT_JSON = "slengo_dictionary.json"
CHECKPOINT  = "slengo_checkpoint.json"  # riprende da qui se interrotto
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)


def get_all_terms() -> list[dict]:
    """
    Legge /browse e restituisce lista di {"term": str, "url": str}.
    La pagina contiene tutti i termini come <a href="/define/..."> in un
    unico blocco — non c'è paginazione da gestire.
    """
    print("→ Recupero indice termini da /browse …")
    r = session.get("https://slengo.it/browse", timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    terms = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/define/") and href != "/define":
            slug = href[len("/define/"):]
            if slug and slug not in seen:
                seen.add(slug)
                terms.append({
                    "term": unquote(slug).replace("-", " "),
                    "slug": slug,
                    "url":  f"https://slengo.it{href}",
                })

    print(f"   Trovati {len(terms)} termini unici.")
    return terms


def scrape_definition(url: str) -> dict:
    """
    Scarica la pagina di un termine e ne estrae:
      - term         : parola/espressione
      - variants     : varianti (es. "anche attaccare il pippone")
      - definitions  : lista di definizioni numerate
      - examples     : lista di esempi
      - related      : termini correlati (Cfr.)
    """
    r = session.get(url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    result = {
        "term": "",
        "variants": "",
        "definitions": [],
        "examples": [],
        "related": [],
    }

    # ── Titolo / varianti ─────────────────────────────────────────────────────
    # La struttura è:  <h1> termine </h1>  seguito da testo italico per varianti
    h1 = soup.find("h1")
    if h1:
        result["term"] = h1.get_text(" ", strip=True)
        # testo dopo h1 nella stessa riga (varianti in <em>)
        em = h1.find_next("em")
        if em and em.parent == h1.parent:
            result["variants"] = em.get_text(strip=True)

    # ── Definizioni ───────────────────────────────────────────────────────────
    # Ogni definizione è in un <p> o <div> contenente il testo lungo,
    # preceduto da un numero (1, 2, …).  Il modo più robusto è leggere
    # i meta tag og:description che contengono la definizione pulita.
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        # rimuove il prefisso "Significato di TERMINE: "
        raw = og_desc["content"]
        raw = re.sub(r"^Significato di .+?:\s*", "", raw, flags=re.DOTALL)
        result["definitions"] = [raw.strip()]

    # Se non c'è og:description, fallback al corpo della pagina
    if not result["definitions"]:
        # cerca il contenitore principale della definizione
        for sel in ["div.word-body", "div.definition", "article", "main"]:
            block = soup.select_one(sel)
            if block:
                text = block.get_text("\n", strip=True)
                if len(text) > 30:
                    result["definitions"] = [text[:2000]]
                    break

    # ── Esempi ────────────────────────────────────────────────────────────────
    # Gli esempi sono in <li> o <p> all'interno di una sezione "Esempi"
    esempi_header = soup.find(lambda t: t.name in ("h2", "h3") and
                              "esempi" in t.get_text(strip=True).lower())
    if esempi_header:
        container = esempi_header.find_next_sibling()
        while container:
            if container.name in ("h2", "h3"):
                break
            items = container.find_all("li") if container.name == "ul" else [container]
            for item in items:
                text = item.get_text("\n", strip=True)
                if text and len(text) > 5:
                    result["examples"].append(text)
            container = container.find_next_sibling()

    # ── Termini correlati (Cfr.) ──────────────────────────────────────────────
    cfr_pattern = re.compile(r"cfr\.?", re.IGNORECASE)
    for tag in soup.find_all(string=cfr_pattern):
        parent = tag.parent
        if parent:
            links = parent.find_all("a", href=lambda h: h and "/define/" in h)
            result["related"] = [a.get_text(strip=True) for a in links]
            break

    return result


def load_checkpoint() -> set:
    """Carica la lista degli slug già processati."""
    try:
        with open(CHECKPOINT, encoding="utf-8") as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(list(done), f)


def main():
    terms = get_all_terms()
    done  = load_checkpoint()

    # Carica risultati già salvati (se si riprende dopo interruzione)
    existing = []
    try:
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    results  = existing
    total    = len(terms)
    skipped  = 0

    print(f"→ Inizia lo scraping di {total} termini "
          f"({len(done)} già completati)…\n")

    for idx, t in enumerate(terms, 1):
        slug = t["slug"]
        if slug in done:
            skipped += 1
            continue

        try:
            data = scrape_definition(t["url"])
            entry = {
                "term":        t["term"],
                "slug":        slug,
                "url":         t["url"],
                "variants":    data["variants"],
                "definitions": data["definitions"],
                "examples":    data["examples"],
                "related":     data["related"],
            }
            results.append(entry)
            done.add(slug)

            defs_ok = "✓" if data["definitions"] else "✗"
            ex_n    = len(data["examples"])
            print(f"  [{idx:>5}/{total}] {defs_ok} def  {ex_n} esempi  {t['term']}")

        except requests.HTTPError as e:
            print(f"  [{idx:>5}/{total}] HTTP {e.response.status_code}  {t['term']}")
        except Exception as e:
            print(f"  [{idx:>5}/{total}] ERRORE: {e}  {t['term']}")

        # Salva checkpoint ogni 50 termini
        if (idx - skipped) % 50 == 0:
            save_checkpoint(done)
            _write_outputs(results)
            print(f"   💾  Checkpoint salvato ({len(done)} completati)")

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    # Salvataggio finale
    save_checkpoint(done)
    _write_outputs(results)
    print(f"\n✅ Completato! {len(results)} termini salvati in {OUTPUT_JSON} e {OUTPUT_CSV}")


def _write_outputs(results: list[dict]):
    # JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV (definitions e examples come stringhe separate da " | ")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "term", "slug", "url", "variants",
            "definitions", "examples", "related"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                **r,
                "definitions": " | ".join(r["definitions"]),
                "examples":    " | ".join(r["examples"]),
                "related":     ", ".join(r["related"]),
            })


if __name__ == "__main__":
    main()
