"""Sicherheits-Prüfungen für das Moteur: API-Dokumentation aus, Schlüsselpflicht auf allen Routen, Schutz-Header, Längenlimit.
Aufruf: python3 test_hardening.py"""
import asyncio
import os
import sys

import harness as h

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = next(p for p in (os.path.join(HERE, "..", "api_server.py"), os.path.join(HERE, "api_server.py")) if os.path.exists(p))
failed = 0


def check(name, cond, detail=""):
    global failed
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond:
        failed += 1


def engine(name, docs=None):
    old = os.environ.pop("WHISPER_ENABLE_DOCS", None)
    if docs is not None:
        os.environ["WHISPER_ENABLE_DOCS"] = docs
    try:
        model = h.FakeModel()
        return h.load_engine(ENGINE_PATH, name, model=model, diarization=False), model
    finally:
        os.environ.pop("WHISPER_ENABLE_DOCS", None)
        if old is not None:
            os.environ["WHISPER_ENABLE_DOCS"] = old


async def call(mod, method, url, **kw):
    async with h.client(mod) as c:
        return await c.request(method, url, **kw)


os.environ["WHISPER_API_KEY"] = "geheim-key"
AUTH = {"Authorization": "Bearer geheim-key"}

# --- API-Dokumentation ---
mod, _ = engine("eng_h1")
for url in ("/docs", "/redoc", "/openapi.json"):
    r = asyncio.run(call(mod, "GET", url))
    check(f"Standard: {url} ist nicht erreichbar (404)", r.status_code == 404, r.status_code)
mod2, _ = engine("eng_h2", docs="true")
check("WHISPER_ENABLE_DOCS=true schaltet /docs und /openapi.json ein", asyncio.run(call(mod2, "GET", "/docs")).status_code == 200 and asyncio.run(call(mod2, "GET", "/openapi.json")).status_code == 200, "")

# --- Schlüsselpflicht ---
check("/health ist ohne Schlüssel erreichbar (für die Überwachung)", asyncio.run(call(mod, "GET", "/health")).status_code == 200, "")
files = {"file": ("t.wav", h.wav_bytes(2), "audio/wav")}
for method, url, kw in (("GET", "/v1/models", {}), ("POST", "/v1/jobs/abc/cancel", {}), ("POST", "/v1/audio/transcriptions", {"files": files}), ("POST", "/v1/audio/translations", {"files": files})):
    r = asyncio.run(call(mod, method, url, **kw))
    check(f"Ohne Schlüssel: {method} {url} -> 401", r.status_code == 401, r.status_code)
    r = asyncio.run(call(mod, method, url, headers={"Authorization": "Bearer falsch"}, **kw))
    check(f"Falscher Schlüssel: {method} {url} -> 401", r.status_code == 401, r.status_code)
check("Mit richtigem Schlüssel: /v1/models -> 200", asyncio.run(call(mod, "GET", "/v1/models", headers=AUTH)).status_code == 200, "")
unprotected = []
for route in mod.app.routes:
    path = getattr(route, "path", "")
    dependant = getattr(route, "dependant", None)
    if dependant is None or path in ("/health",):
        continue
    if not any(d.call is mod._verify_api_key for d in dependant.dependencies):
        unprotected.append(path)
check("Jede Route außer /health hat die Schlüsselprüfung (auch künftig neue)", not unprotected, unprotected)

# --- Schutz-Header ---
r = asyncio.run(call(mod, "GET", "/v1/models", headers=AUTH))
check("Antworten tragen nosniff, no-store und no-referrer", r.headers.get("x-content-type-options") == "nosniff" and r.headers.get("cache-control") == "no-store" and r.headers.get("referrer-policy") == "no-referrer", dict(r.headers))
r = asyncio.run(call(mod, "POST", "/v1/audio/transcriptions", headers=AUTH, files=files, data={"stream": "true"}))
check("Der Live-Stream behält seinen eigenen Cache-Header (no-cache), bekommt aber nosniff", r.headers.get("cache-control") == "no-cache" and r.headers.get("x-content-type-options") == "nosniff", dict(r.headers))

# --- Längenlimit ---
mod3, model3 = engine("eng_h3")
mod3._max_duration_min = 1
long_wav, short_wav = h.wav_bytes(90), h.wav_bytes(30)
r = asyncio.run(call(mod3, "POST", "/v1/audio/transcriptions", headers=AUTH, files={"file": ("l.wav", long_wav, "audio/wav")}))
check("Zu lange Aufnahme (90 s bei Limit 1 Minute) wird mit 413 abgelehnt", r.status_code == 413, (r.status_code, r.text[:120]))
check("... mit verständlichem Text", "too long" in r.text and "1 minutes" in r.text, r.text[:160])
check("... und Whisper wurde gar nicht erst aufgerufen", model3.calls == [], len(model3.calls))
async def stream_with_key(mod, wav):
    async with h.client(mod) as c:
        async with c.stream("POST", "/v1/audio/transcriptions", headers=AUTH, files={"file": ("t.wav", wav, "audio/wav")}, data={"stream": "true"}) as r:
            await r.aread()
            return r.status_code
check("Auch im Live-Ablauf (stream=true) kommt 413, noch bevor der Stream beginnt", asyncio.run(stream_with_key(mod3, long_wav)) == 413, "")
r = asyncio.run(call(mod3, "POST", "/v1/audio/transcriptions", headers=AUTH, files={"file": ("s.wav", short_wav, "audio/wav")}))
check("Eine kurze Aufnahme (30 s) geht durch", r.status_code == 200, (r.status_code, r.text[:100]))
mod3._max_duration_min = 0
r = asyncio.run(call(mod3, "POST", "/v1/audio/transcriptions", headers=AUTH, files={"file": ("l.wav", long_wav, "audio/wav")}))
check("Mit Limit 0 (unbegrenzt) geht auch die lange Aufnahme durch", r.status_code == 200, r.status_code)
# zweite Sicherung: der Kopf behauptet nichts, aber die dekodierte Länge ist zu groß
mod4, model4 = engine("eng_h4")
mod4._max_duration_min = 1
mod4._probe_duration = lambda path, cap: None
r = asyncio.run(call(mod4, "POST", "/v1/audio/transcriptions", headers=AUTH, files={"file": ("l.wav", long_wav, "audio/wav")}))
check("Zweite Sicherung: auch wenn der Dateikopf keine Dauer verrät, wird die zu lange Aufnahme nach dem Dekodieren gestoppt", r.status_code >= 400 and model4.calls == [], (r.status_code, len(model4.calls)))
import tempfile
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(h.wav_bytes(5)); path = f.name
d = mod3._probe_duration(path, 3600)
check("_probe_duration: 5-Sekunden-Datei ergibt etwa 5 s", d is not None and abs(d - 5) < 0.2, d)
with open(path, "wb") as f:
    f.write(b"das ist kein audio" * 50)
check("_probe_duration: Unlesbares ergibt None (der normale Weg meldet den Fehler)", mod3._probe_duration(path, 3600) is None, "")
os.unlink(path)

print("\n" + (f"{failed} Prüfung(en) FEHLGESCHLAGEN" if failed else "Alle Prüfungen bestanden"))
sys.exit(1 if failed else 0)
