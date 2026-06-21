import os
import requests

#API_KEY = os.environ["GEMINI_API_KEY"]
#API_KEY = os.environ["GEMINI_API_KEY_1"]
API_KEY = os.environ["GEMINI_API_KEY_2"]

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [{
        "parts": [{"text": 'Sei un madrelingua italiano. Genera una nuova parola che non esiste in italiano per la seguente definizione:'
        '"Una persona che non vince mai, ma che non perde neanche, inutile, mediocre, senza infamia e senza lode, sostanzialmente innocua. "'
        'Rispondi solo con la parola.'}]
    }],
    "generationConfig": {
        "temperature": 1.0,
        "topK": 64,
        "topP": 0.95,
        "maxOutputTokens": 500,
        "thinkingConfig": {"thinkingBudget": 0}
    }
}

response = requests.post(url, json=payload)
data = response.json()
print(data)
print(data["candidates"][0]["content"]["parts"][0]["text"].strip())