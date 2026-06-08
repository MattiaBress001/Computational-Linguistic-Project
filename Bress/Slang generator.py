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
TIMEOUT          = 240

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Everything that you find inside the triple quotation marks ("""This""") is documentation
# If you later want to visualize what a precise function does, you can add to the main (if __name__ == "__main__":):
# help(function_name)
# This will print on console the documentation related to the function :D 

# ---------------------------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------------------------

def load_existing_words() -> List[Dict]:
    """Load the already generated slang terms from the JSON file. The function returns an empty list if this file doesn't exist.""" 
    if not os.path.exists(OUTPUT_FILE):
        return []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            log.warning("parole_generate.json is corrupt, let's restart from the beginning.")
            return []


def save_words(entries: List[Dict]) -> None:
    """Overwrite the file JSON with the alredy genereted slang terms with an updated list."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    log.info("The file parole_generate.json has been updated (%d total words).", len(entries))


def append_new_entries(new_entries: List[Dict]) -> List[Dict]:
    """Adds the new entries to the existing ones, avoiding duplicates, and saves."""
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
    existing    = load_existing_words()
    known_words = [e["parola"] for e in existing]

    # Semd only the last 30 words in order to not make the prompt enormus (small models tend to lose focus)
    recent = known_words[-30:] if len(known_words) > 30 else known_words

    if recent:
        blacklist = "\n".join(f"- {w}" for w in recent)
        avoid_section = f"\nParole vietate (non generare nessuna di queste né varianti):\n{blacklist}\n"
    else:
        avoid_section = ""

    return f"""
Sei un esperto di linguaggio giovanile italiano contemporaneo (Gen Z, social media, meme culture).

Genera esattamente {n} termini slang INVENTATI ma plausibili — non devono esistere già in italiano.

Ogni termine può nascere da:
- abbreviazioni (es. "troppof" = troppo forte)
- italianizzazione di inglese (es. "scrollare", "floppare")
- metafore da internet/meme/social
- storpiature fonetiche creative
{avoid_section}
Output: SOLO JSON valido, zero testo extra, zero markdown.

{{
  "entries": [
    {{
      "parola": "termine inventato",
      "definizione": "significato breve e diretto",
      "contesto_di_utilizzo": ["frase esempio 1", "frase esempio 2"]
    }}
  ]
}}
"""

# Quick tip: select the code and press Ctrl + / for commenting/uncommenting in bulk ;)

#     return f"""
# Sei un generatore creativo di slang italiano contemporaneo.

# Genera esattamente {n} voci slang NUOVE e ORIGINALI usate da giovani italiani oggi.
# {avoid_section}
# Rispondi SOLO con JSON valido, senza testo aggiuntivo, senza markdown, senza code block.

# {{
#   "entries": [
#     {{
#       "parola": "stringa",
#       "definizione": "spiegazione chiara e diretta",
#       "contesto_di_utilizzo": ["esempio di frase 1", "esempio di frase 2"]
#     }}
#   ]
# }}
# """

# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def call_model(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()["response"]

# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON objects were found in the output.")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            return text[start : i + 1]

    raise ValueError("Unbalanced JSON in the model output.")

# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------

def parse_entries(raw: str) -> List[Dict]:
    cleaned     = extract_json(raw)
    data        = json.loads(cleaned)
    entries_raw = data.get("entries")

    if not isinstance(entries_raw, list) or len(entries_raw) == 0:
        raise ValueError("The 'entries' field is missing or empty.")

    entries = []
    for i, item in enumerate(entries_raw):
        parola       = item.get("parola", "").strip()
        definizione  = item.get("definizione", "").strip()
        contesti_raw = item.get("contesto_di_utilizzo", [])

        if not parola or not definizione:
            log.warning("Entry %d ignored: required fields are missing.", i)
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
        log.info("Attempt %d/%d …", attempt, MAX_RETRIES)
        try:
            raw     = call_model(prompt)
            entries = parse_entries(raw)

            if not entries:
                raise ValueError("Nessuna entry valida estratta.")

            log.info("✓ %d/%d entries retrieved.", len(entries), n)
            return entries

        except requests.RequestException as e:
            log.error("Connection error: %s", e)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            log.warning("Parsing failed: %s", e)

        if attempt < MAX_RETRIES:
            log.info("I'll try again after %.1fs …", RETRY_DELAY)
            time.sleep(RETRY_DELAY)

    log.error("All attempts are failed.")
    return []

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_entries(entries: List[Dict]) -> None:
    if not entries:
        print("\nNo entry to show.")
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
    # help(load_existing_words)
    total = load_existing_words()
    log.info("Words already in the database: %d", len(total))

    print("\nHow many times do you want to send the prompt? (ENTER = 1)")
    cmd = input("> ").strip()

    try:
        runs = int(cmd) if cmd else 1
    except ValueError:
        log.warning("The input is invalid, I'm using 1.")
        runs = 1

    for i in range(runs):
        log.info("── Run %d/%d ──", i + 1, runs)

        new_entries = generate_slang(n=NUM_ENTRIES)

        if new_entries:
            added = append_new_entries(new_entries)
            log.info("New words added: %d", len(added))
            print_entries(added)
        else:
            print("No words generated in this round.")

    log.info("Done! Total words saved: %d", len(load_existing_words()))