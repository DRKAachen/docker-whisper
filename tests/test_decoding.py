"""Prüft, mit welchen Einstellungen das Moteur Whisper aufruft (Schutz vor Wiederholungsschleifen).
Aufruf: python3 test_decoding.py"""
import asyncio
import os
import sys

import harness as h

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = next(p for p in (os.path.join(HERE, "..", "api_server.py"), os.path.join(HERE, "api_server.py")) if os.path.exists(p))
AUDIO = h.wav_bytes(3)
FALLBACK = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
failed = 0


def check(name, cond, detail=""):
    global failed
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond:
        failed += 1


def engine(name, env=None):
    old = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT")
    os.environ.pop("WHISPER_CONDITION_ON_PREVIOUS_TEXT", None)
    if env is not None:
        os.environ["WHISPER_CONDITION_ON_PREVIOUS_TEXT"] = env
    try:
        model = h.FakeModel()
        return h.load_engine(ENGINE_PATH, name, model=model, diarization=False), model
    finally:
        os.environ.pop("WHISPER_CONDITION_ON_PREVIOUS_TEXT", None)
        if old is not None:
            os.environ["WHISPER_CONDITION_ON_PREVIOUS_TEXT"] = old


async def batch(mod, data=None):
    async with h.client(mod) as c:
        return await c.post("/v1/audio/transcriptions", files={"file": ("t.wav", AUDIO, "audio/wav")}, data=data or {})


async def stream(mod, data=None):
    async with h.client(mod) as c:
        async with c.stream("POST", "/v1/audio/transcriptions", files={"file": ("t.wav", AUDIO, "audio/wav")}, data={"stream": "true", **(data or {})}) as r:
            async for _ in r.aiter_lines():
                pass
            return r.status_code


# --- Hilfsfunktion ---
mod, _ = engine("eng_dec_unit")
o = mod._decode_options(0.0)
check("Temperatur 0 (Standard der Anfrage) ergibt die Ersatzfolge 0, 0.2 ... 1.0", o["temperature"] == FALLBACK, o)
check("Ausdrücklich angegebene Temperatur bleibt unverändert", mod._decode_options(0.4)["temperature"] == 0.4, "")
check("Standard: der Text davor ist KEINE Vorlage (condition_on_previous_text aus)", o["condition_on_previous_text"] is False, o)
check("Schwellen für Wiederholung und Unsicherheit sind gesetzt", o["compression_ratio_threshold"] == 2.4 and o["log_prob_threshold"] == -1.0 and o["no_speech_threshold"] == 0.6, o)
for val, want in [("true", True), ("TRUE", True), ("false", False), ("", False), ("ja", False)]:
    m, _ = engine("eng_dec_env", env=val)
    check(f"WHISPER_CONDITION_ON_PREVIOUS_TEXT={val!r} -> {want}", m._decode_options(0)["condition_on_previous_text"] is want, "")

# --- Aufruf im normalen Ablauf ---
mod, model = engine("eng_dec_batch")
r = asyncio.run(batch(mod))
kw = model.calls[0][1] if model.calls else {}
check("Normaler Ablauf: Antwort 200", r.status_code == 200, r.status_code)
check("Normaler Ablauf: Whisper bekommt die Ersatzfolge statt einer festen 0", kw.get("temperature") == FALLBACK, kw.get("temperature"))
check("Normaler Ablauf: condition_on_previous_text=False und Schwellen kommen an", kw.get("condition_on_previous_text") is False and kw.get("compression_ratio_threshold") == 2.4, kw)
mod, model = engine("eng_dec_batch2")
asyncio.run(batch(mod, {"temperature": "0.3"}))
check("Normaler Ablauf: angegebene Temperatur 0.3 wird verwendet", model.calls[0][1].get("temperature") == 0.3, model.calls[0][1].get("temperature"))

# --- Aufruf im Live-Ablauf (stream=true), wie ihn das Frontend benutzt ---
mod, model = engine("eng_dec_stream")
code = asyncio.run(stream(mod))
kw = model.calls[0][1] if model.calls else {}
check("Live-Ablauf: Antwort 200", code == 200, code)
check("Live-Ablauf: Whisper bekommt die Ersatzfolge und kein condition_on_previous_text", kw.get("temperature") == FALLBACK and kw.get("condition_on_previous_text") is False, kw)
mod, model = engine("eng_dec_stream2", env="true")
asyncio.run(stream(mod))
check("Live-Ablauf: mit WHISPER_CONDITION_ON_PREVIOUS_TEXT=true gilt wieder das frühere Verhalten", model.calls[0][1].get("condition_on_previous_text") is True, model.calls[0][1])

print("\n" + (f"{failed} Prüfung(en) FEHLGESCHLAGEN" if failed else "Alle Prüfungen bestanden"))
sys.exit(1 if failed else 0)
