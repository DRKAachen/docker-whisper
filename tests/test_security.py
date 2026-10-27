"""Sicherheits-Tests für das Moteur: Schlüsselprüfung in konstanter Zeit und sichere Dateiendung.
Aufruf: python3 test_security.py"""
import inspect
import os
import sys

from fastapi import HTTPException

import harness as h

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = next(p for p in (os.path.join(HERE, "..", "api_server.py"), os.path.join(HERE, "api_server.py")) if os.path.exists(p))
failed = 0


def check(name, cond, detail=""):
    global failed
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond:
        failed += 1


mod = h.load_engine(ENGINE_PATH, "engine_security")

# --- Dateiendung der temporären Datei ---
cases = {
    "a.wav": ".wav", "Aufnahme.MP3": ".mp3", "x": ".audio", "x.tar.gz": ".gz", "../../x.wav": ".wav",
    "evil.wa/../../etc": ".audio", "a.\x00wav": ".audio", "x.abcdefghijk": ".audio", "weird.w av": ".audio",
    "a.wä": ".audio", "": ".audio", None: ".audio", ".hidden": ".audio", "C:\\tmp\\x.m4a": ".m4a",
}
for name, want in cases.items():
    got = mod._safe_suffix(name)
    check(f"_safe_suffix({name!r}) = {want!r}", got == want, got)
check("Die Endung wird im Upload-Weg verwendet (nicht mehr die rohe Endung)", "_safe_suffix(original_name)" in inspect.getsource(mod), "")

# --- Schlüsselprüfung ---
os.environ["WHISPER_API_KEY"] = "geheim-123"


def verdict(header):
    try:
        mod._verify_api_key(header)
        return 200
    except HTTPException as e:
        return e.status_code


check("Ohne Kopfzeile: 401", verdict(None) == 401, verdict(None))
check("Falscher Schlüssel: 401", verdict("Bearer falsch") == 401, verdict("Bearer falsch"))
check("Zu kurzer und zu langer Schlüssel: 401", verdict("Bearer geheim-12") == 401 and verdict("Bearer geheim-1234") == 401, "")
check("Falsches Schema: 401", verdict("Basic geheim-123") == 401 and verdict("geheim-123") == 401, "")
check("Sonderzeichen im Schlüssel führen zu 401 statt zu einem Fehler", verdict("Bearer geheim-é\u00ff") == 401, verdict("Bearer geheim-é\u00ff"))
check("Richtiger Schlüssel: durchgelassen", verdict("Bearer geheim-123") == 200, verdict("Bearer geheim-123"))
check("Schreibweise 'bearer' wird akzeptiert", verdict("bearer geheim-123") == 200, "")
src = inspect.getsource(mod._verify_api_key)
check("Der Vergleich läuft in konstanter Zeit (hmac.compare_digest), nicht mit !=", "compare_digest" in src and "!= required" not in src, src[-200:])
os.environ["WHISPER_API_KEY"] = ""
check("Ohne gesetzten Schlüssel bleibt es offen (wie bisher, wird im Frontend durch REQUIRE_AUTH und die Netzwerkgrenze geschützt)", verdict(None) == 200, "")

print("\n" + (f"{failed} Prüfung(en) FEHLGESCHLAGEN" if failed else "Alle Prüfungen bestanden"))
sys.exit(1 if failed else 0)
