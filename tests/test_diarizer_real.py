"""Prüft die echte Sprecher-Erkennung (diarizer.py mit sherpa-onnx): Fortschritt, Abbruch, Einstellungen, Aufräumen.
Braucht die Modelle und ein Beispiel-Audio mit mindestens zwei Stimmen. Ohne sie wird übersprungen.
Aufruf:  DIARIZER_MODELS=/pfad/zu/modellen DIARIZER_TEST_AUDIO=/pfad/zu/audio.wav python3 test_diarizer_real.py
(DIARIZER_MODELS enthält sherpa-onnx-pyannote-segmentation-3-0/model.onnx und das 3dspeaker-Modell.)"""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = next(p for p in (os.path.join(HERE, "..", "diarizer.py"), os.path.join(HERE, "diarizer.py")) if os.path.exists(p))
MODELS, AUDIO = os.environ.get("DIARIZER_MODELS"), os.environ.get("DIARIZER_TEST_AUDIO")
failed = 0


def check(name, cond, detail=""):
    global failed
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond:
        failed += 1


try:
    import sherpa_onnx  # noqa: F401
except ImportError:
    print("ÜBERSPRUNGEN: sherpa-onnx ist nicht installiert")
    sys.exit(0)
if not (MODELS and AUDIO and os.path.isdir(MODELS) and os.path.isfile(AUDIO)):
    print("ÜBERSPRUNGEN: DIARIZER_MODELS und DIARIZER_TEST_AUDIO setzen")
    sys.exit(0)


def fresh(name="dz_real"):
    spec = importlib.util.spec_from_file_location(name, PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Einstellungen ---
dz = fresh()
saved = {k: os.environ.get(k) for k in ("WHISPER_DIARIZE_THREADS", "WHISPER_DIARIZE_EMBEDDING")}
for val, ok in (("3", lambda n: n == 3), ("1", lambda n: n == 1), ("99", lambda n: n == 32), ("0", lambda n: 1 <= n <= 8), ("abc", lambda n: 1 <= n <= 8), ("", lambda n: 1 <= n <= 8), ("-2", lambda n: 1 <= n <= 8)):
    os.environ["WHISPER_DIARIZE_THREADS"] = val
    check(f"WHISPER_DIARIZE_THREADS={val!r} ergibt eine sinnvolle Zahl", ok(dz._num_threads()), dz._num_threads())
os.environ.pop("WHISPER_DIARIZE_THREADS")
check("Ohne Angabe: alle verfügbaren Kerne, höchstens 8", 1 <= dz._num_threads() <= 8, dz._num_threads())
for val, want in (("campplus", "campplus"), ("CAMPPLUS", "campplus"), ("eres2net", "eres2net"), ("", "eres2net"), ("quatsch", "eres2net")):
    os.environ["WHISPER_DIARIZE_EMBEDDING"] = val
    check(f"WHISPER_DIARIZE_EMBEDDING={val!r} -> {want}", dz._embedding_name() == want, dz._embedding_name())
os.environ.pop("WHISPER_DIARIZE_EMBEDDING")
check("Standard-Modell ist weiter ERes2Net (kein stiller Wechsel)", dz._embedding_name() == "eres2net", "")

# --- echte Erkennung ---
dz = fresh("dz_real2")
dz.load(cache_dir=MODELS)
t = time.time(); base = dz.diarize(AUDIO); t_full = time.time() - t
check("Erkennung ohne Fortschritt liefert Abschnitte mit mindestens zwei Sprechern", len(base) > 1 and len({s[2] for s in base}) >= 2, (len(base), len({s[2] for s in base})))

seen = []
turns = dz.diarize(AUDIO, progress=lambda f: seen.append(f) or False)
check("Fortschritt: beginnt bei 0, endet bei 1.0, steigt nie", seen[0] == 0.0 and seen[-1] == 1.0 and all(b >= a for a, b in zip(seen, seen[1:])), (seen[:2], seen[-2:]))
check("Fortschritt: viele Zwischenschritte (nicht nur Anfang und Ende)", len(seen) > 8, len(seen))
check("Fortschritt: Werte liegen immer zwischen 0 und 1", all(0.0 <= f <= 1.0 for f in seen), "")
check("Mit Fortschritt kommt dasselbe Ergebnis wie ohne", len(turns) == len(base) and all(abs(a[0]-b[0]) < 0.05 and a[2] == b[2] for a, b in zip(turns, base)), (len(turns), len(base)))

calls = []
def stop_at_half(f):
    calls.append(f)
    return f >= 0.5
t = time.time()
try:
    dz.diarize(AUDIO, progress=stop_at_half)
    aborted = False
except dz.DiarizationAborted:
    aborted = True
t_abort = time.time() - t
check("Abbruch: wirft DiarizationAborted", aborted, "")
check("Abbruch: die Berechnung hört wirklich auf (deutlich schneller als der volle Lauf)", t_abort < 0.8 * t_full, f"{t_abort:.1f}s gegen {t_full:.1f}s")
check("Abbruch: nach dem Abbruch kommt kein weiterer Fortschritt mehr", calls[-1] >= 0.5 and len([c for c in calls if c >= 0.5]) == 1, calls[-3:])
again = dz.diarize(AUDIO)
check("Nach einem Abbruch funktioniert die Erkennung weiter, mit demselben Ergebnis", len(again) == len(base) and all(abs(a[0]-b[0]) < 0.05 and a[2] == b[2] for a, b in zip(again, base)), "")
try:
    dz.diarize(AUDIO, progress=lambda f: True)
    sofort = False
except dz.DiarizationAborted:
    sofort = True
check("Sofortiger Abbruch (schon beim ersten Aufruf) funktioniert", sofort, "")

# --- Aufräumen: nur wenige, begrenzte Proben ---
dz2 = fresh("dz_real3")
embedded = []
dz2._embed_segment = lambda audio, sr, s, e, max_seconds=10.0: (embedded.append((s, e)) or __import__("numpy").array([1.0, 0.0, 0.0]))
many = [(float(i), float(i) + 2.0, "SPEAKER_00") for i in range(0, 400, 3)] + [(500.0, 501.0, "SPEAKER_01"), (900.0, 900.4, "SPEAKER_02")]
cleaned = dz2._merge_sliver_speakers_by_voice(many, None, 16000, max_turns_per_speaker=12)
check("Aufräumen: Mittelpunkte aus höchstens 12 Abschnitten je Sprecher (statt aus allen, hier 134)", len(embedded) <= 12 * 2 + 2, len(embedded))
check("Aufräumen: Splitter-Sprecher werden weiter einem Hauptsprecher zugeordnet", {s[2] for s in cleaned} <= {"SPEAKER_00", "SPEAKER_01"} or "SPEAKER_02" not in {s[2] for s in cleaned}, {s[2] for s in cleaned})
dz3 = fresh("dz_real4")
class FakeStream:
    def __init__(self): self.n = None
    def accept_waveform(self, sample_rate, waveform): self.n = len(waveform)
    def input_finished(self): pass
class FakeExtractor:
    last = None
    def create_stream(self): FakeExtractor.last = FakeStream(); return FakeExtractor.last
    def compute(self, s): return [0.0, 1.0]
dz3._get_embedding_extractor = lambda: FakeExtractor()
import numpy as np
dz3._embed_segment(np.zeros(16000 * 100, dtype="float32"), 16000, 10.0, 70.0)
check("Aufräumen: eine 60-s-Probe wird auf 10 s gekürzt", FakeExtractor.last.n == 16000 * 10, FakeExtractor.last.n)
dz3._embed_segment(np.zeros(16000 * 100, dtype="float32"), 16000, 10.0, 14.0)
check("Aufräumen: eine 4-s-Probe bleibt unverändert", FakeExtractor.last.n == 16000 * 4, FakeExtractor.last.n)

for k, v in saved.items():
    if v is not None:
        os.environ[k] = v
print("\n" + (f"{failed} Prüfung(en) FEHLGESCHLAGEN" if failed else "Alle Prüfungen bestanden"))
sys.exit(1 if failed else 0)
