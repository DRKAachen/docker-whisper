"""Prüft die Prüfsummen der Sprecher-Modelle (diarizer.py), ohne sherpa-onnx und ohne Netz.
Aufruf: python3 test_integrity.py"""
import importlib.util
import os
import re
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = next(p for p in (os.path.join(HERE, "..", "diarizer.py"), os.path.join(HERE, "diarizer.py")) if os.path.exists(p))
failed = 0


def check(name, cond, detail=""):
    global failed
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond:
        failed += 1


sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))


def fresh():
    spec = importlib.util.spec_from_file_location("dz_integrity", PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


dz = fresh()
os.environ.pop("WHISPER_DIARIZE_SKIP_CHECKSUM", None)

needed = {dz._SEG_ARCHIVE, dz._SEG_MODEL_REL, *dz._EMB_MODELS.values()}
check("Für JEDES Modell (Archiv, Segmentierung, beide Einbettungen) ist eine Prüfsumme festgelegt", needed <= set(dz._SHA256), needed - set(dz._SHA256))
check("Alle Prüfsummen sind gültige SHA-256-Werte (64 Hexzeichen)", all(re.fullmatch(r"[0-9a-f]{64}", v) for v in dz._SHA256.values()), "")

with tempfile.TemporaryDirectory() as d:
    f = os.path.join(d, "modell.onnx")
    with open(f, "wb") as fh:
        fh.write(b"echtes modell")
    good = dz._sha256_of(f)
    check("_sha256_of: bekannter Wert", dz._sha256_of(f) == "a37951786df9e63237ed3c9843134b93d7d411486bea2756120c410f7d9e6df2", dz._sha256_of(f))
    dz._SHA256["test-modell"] = good
    try:
        dz._verify_file(f, "test-modell"); ok = True
    except RuntimeError:
        ok = False
    check("Passende Datei wird akzeptiert", ok and os.path.exists(f), "")
    with open(f, "wb") as fh:
        fh.write(b"echtes modell, aber ein Byte mehr")
    try:
        dz._verify_file(f, "test-modell"); rejected = False
    except RuntimeError as e:
        rejected = "Checksum mismatch" in str(e)
    check("Veränderte Datei wird abgelehnt", rejected, "")
    check("... und sofort gelöscht (sie kann nie geladen werden)", not os.path.exists(f), "")
    with open(f, "wb") as fh:
        fh.write(b"x")
    try:
        dz._verify_file(f, "unbekannter-schluessel"); unknown = False
    except RuntimeError as e:
        unknown = "No pinned checksum" in str(e)
    check("Datei ohne festgelegte Prüfsumme wird abgelehnt (nie 'stillschweigend ok')", unknown, "")
    os.environ["WHISPER_DIARIZE_SKIP_CHECKSUM"] = "true"
    try:
        dz._verify_file(f, "unbekannter-schluessel"); skipped = True
    except RuntimeError:
        skipped = False
    os.environ.pop("WHISPER_DIARIZE_SKIP_CHECKSUM")
    check("WHISPER_DIARIZE_SKIP_CHECKSUM=true ist die einzige Ausnahme (ausdrücklich, mit Warnung im Log)", skipped, "")

    # _ensure_models mit einem 'vergifteten' Zwischenspeicher: kein Netz nötig, weil die Dateien schon da sind
    seg = os.path.join(d, dz._SEG_MODEL_REL)
    os.makedirs(os.path.dirname(seg), exist_ok=True)
    with open(seg, "wb") as fh:
        fh.write(b"vergiftetes segmentierungsmodell")
    try:
        dz._ensure_models(d); poisoned_ok = True
    except RuntimeError as e:
        poisoned_ok = False
    check("Ein ausgetauschtes Modell im Zwischenspeicher wird beim Start erkannt und nicht geladen", not poisoned_ok, "")
    check("... und aus dem Zwischenspeicher entfernt (beim nächsten Start wird frisch und geprüft geladen)", not os.path.exists(seg), "")

models = os.environ.get("DIARIZER_MODELS")
if models and os.path.isdir(models):
    for key in dz._SHA256:
        p = os.path.join(models, key)
        if os.path.isfile(p):
            check(f"Echte Datei stimmt mit der Prüfsumme überein: {key}", dz._sha256_of(p) == dz._SHA256[key], "")

print("\n" + (f"{failed} Prüfung(en) FEHLGESCHLAGEN" if failed else "Alle Prüfungen bestanden"))
sys.exit(1 if failed else 0)
