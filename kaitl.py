import re
import json
import time
import logging
import os
from typing import List, Dict, Optional
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "gemma2:2b"
NUM_ENTRIES = 5
MAX_RETRIES = 3
RETRY_DELAY = 1.5   # seconds between retries

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(n: int) -> str:
    return f"""Sei un generatore creativo di slang italiano contemporaneo.

Genera esattamente {n} voci slang originali usate da giovani italiani oggi.

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
    """
    Pulisce l'output del modello ed estrae il primo oggetto JSON valido,
    anche se il modello aggiunge testo o code-fence prima/dopo.
    """
    # Rimuove eventuali code-fence markdown
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
    cleaned = extract_json(raw)
    data    = json.loads(cleaned)

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

        if isinstance(contesti_raw, list):
            contesti = [str(c).strip() for c in contesti_raw if str(c).strip()]
        else:
            contesti = [str(contesti_raw).strip()]

        entries.append(
            {
                "parola": parola,
                "definizione": definizione,
                "contesto_di_utilizzo": contesti,
                "timestamp": datetime.now().isoformat(),
            }
        )

    return entries


# ---------------------------------------------------------------------------
# Main loop with retry
# ---------------------------------------------------------------------------

def generate_slang(n: int = NUM_ENTRIES) -> List[Dict]:
    prompt = build_prompt(n)

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("Tentativo %d/%d …", attempt, MAX_RETRIES)
        try:
            raw = call_model(prompt)
            log.debug("Output grezzo:\n%s", raw)

            entries = parse_entries(raw)

            if len(entries) == 0:
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

    log.error("Tutti i tentativi falliti. Ritorno lista vuota.")
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


def save_to_json(entries: List[Dict], path: Optional[str] = None) -> str:
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename   = f"slang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path       = os.path.join(script_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    log.info("✓ Salvato in: %s", path)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    entries = generate_slang(n=NUM_ENTRIES)
    print_entries(entries)

    if entries:
        save_to_json(entries)