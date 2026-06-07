import re
import json
import time
import logging
import os
from typing import List, Dict
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL       = "http://localhost:11434/api/generate"
MODEL            = "gemma2:2b"
NUM_ENTRIES      = 3
MAX_RETRIES      = 3
RETRY_DELAY      = 1.5
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE      = os.path.join(SCRIPT_DIR, "parole_generate.json")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------------------------

def load_existing_words() -> List[Dict]:
    """Carica le parole già generate dal file JSON. Ritorna lista vuota se non esiste."""
    if not os.path.exists(OUTPUT_FILE):
        return []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            log.warning("parole_generate.json corrotto, si riparte da zero.")
            return []


def save_words(entries: List[Dict]) -> None:
    """Sovrascrive il file JSON con la lista aggiornata."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    log.info("✓ parole_generate.json aggiornato (%d parole totali).", len(entries))


def append_new_entries(new_entries: List[Dict]) -> List[Dict]:
    """Aggiunge le nuove entry a quelle esistenti, evitando duplicati, e salva."""
    existing     = load_existing_words()
    known_words  = {e["parola"].lower() for e in existing}

    added = []
    for entry in new_entries:
        if entry["parola"].lower() not in known_words:
            existing.append(entry)
            known_words.add(entry["parola"].lower())
            added.append(entry)
        else:
            log.warning("Duplicato ignorato: '%s'", entry["parola"])

    if added:
        save_words(existing)

    return added

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(n: int) -> str:
    existing     = load_existing_words()
    known_words  = [e["parola"] for e in existing]

    if known_words:
        blacklist = ", ".join(f'"{w}"' for w in known_words)
        avoid_section = (
            f"\nParole già generate (NON ripetere nessuna di queste):\n{blacklist}\n"
        )
    else:
        avoid_section = ""

    return f"""Sei un generatore creativo di slang italiano contemporaneo.

Genera esattamente {n} voci slang NUOVE e ORIGINALI usate da giovani italiani oggi.
{avoid_section}
Rispondi SOLO con JSON valido, senza testo aggiuntivo, senza markdown, senza code block.

{{
  "entries": [
    {{
      "parola": "stringa",
      "definizione": "spiegazione chiara e diretta",
      "contesto_di_utilizzo": ["esempio di frase 1", "esempio di frase 2"]
    }}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def call_model(prompt: str, timeout: int = 180) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"]

# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("Nessun oggetto JSON trovato nell'output.")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            return text[start : i + 1]

    raise ValueError("JSON non bilanciato nell'output del modello.")

# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------

def parse_entries(raw: str) -> List[Dict]:
    cleaned     = extract_json(raw)
    data        = json.loads(cleaned)
    entries_raw = data.get("entries")

    if not isinstance(entries_raw, list) or len(entries_raw) == 0:
        raise ValueError("Il campo 'entries' è assente o vuoto.")

    entries = []
    for i, item in enumerate(entries_raw):
        parola       = item.get("parola", "").strip()
        definizione  = item.get("definizione", "").strip()
        contesti_raw = item.get("contesto_di_utilizzo", [])

        if not parola or not definizione:
            log.warning("Entry %d ignorata: mancano campi obbligatori.", i)
            continue

        contesti = (
            [str(c).strip() for c in contesti_raw if str(c).strip()]
            if isinstance(contesti_raw, list)
            else [str(contesti_raw).strip()]
        )

        entries.append({
            "parola": parola,
            "definizione": definizione,
            "contesto_di_utilizzo": contesti,
            "timestamp": datetime.now().isoformat(),
        })

    return entries

# ---------------------------------------------------------------------------
# Generation with retry
# ---------------------------------------------------------------------------

def generate_slang(n: int = NUM_ENTRIES) -> List[Dict]:
    prompt = build_prompt(n)

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("Tentativo %d/%d …", attempt, MAX_RETRIES)
        try:
            raw     = call_model(prompt)
            entries = parse_entries(raw)

            if not entries:
                raise ValueError("Nessuna entry valida estratta.")

            log.info("✓ Estratte %d/%d entries.", len(entries), n)
            return entries

        except requests.RequestException as e:
            log.error("Errore di rete: %s", e)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            log.warning("Parsing fallito: %s", e)

        if attempt < MAX_RETRIES:
            log.info("Riprovo tra %.1fs …", RETRY_DELAY)
            time.sleep(RETRY_DELAY)

    log.error("Tutti i tentativi falliti.")
    return []

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_entries(entries: List[Dict]) -> None:
    if not entries:
        print("\nNessuna entry da mostrare.")
        return

    print(f"\n{'─' * 60}")
    for e in entries:
        print(f"  Parola      : {e['parola']}")
        print(f"  Definizione : {e['definizione']}")
        print(f"  Contesti    :")
        for ctx in e["contesto_di_utilizzo"]:
            print(f"    • {ctx}")
        print(f"  Timestamp   : {e['timestamp']}")
        print(f"{'─' * 60}")

# ---------------------------------------------------------------------------
# Entry point — loop interattivo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    total = load_existing_words()
    log.info("Parole già nel database: %d", len(total))

    print("\nQuante volte vuoi inviare il prompt? (INVIO = 1)")
    cmd = input("> ").strip()

    try:
        runs = int(cmd) if cmd else 1
    except ValueError:
        log.warning("Input non valido, uso 1.")
        runs = 1

    for i in range(runs):
        log.info("── Run %d/%d ──", i + 1, runs)

        new_entries = generate_slang(n=NUM_ENTRIES)

        if new_entries:
            added = append_new_entries(new_entries)
            log.info("Parole nuove aggiunte: %d", len(added))
            print_entries(added)
        else:
            print("Nessuna parola generata in questo giro.")

    log.info("Fatto! Totale parole salvate: %d", len(load_existing_words()))