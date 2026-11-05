"""Tests für den Gateway-Weg (WHISPER_BACKEND=gateway) mit einer falschen Passerelle.
Aufruf: python3 test_gateway.py        (optional: TEST_SPEECH_WAV=<Datei mit echter Sprache> für den Test der Sprachaktivitäts-Erkennung)"""
import asyncio, hashlib, io, json, logging, os, random, struct, sys, tempfile, threading, time, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import harness as h

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = next(p for p in (os.path.join(HERE, "..", "api_server.py"), os.path.join(HERE, "api_server.py")) if os.path.exists(p))
KEY = "geheimer-test-schluessel-123"
RATE = 16000
failed = 0


def check(name, cond, detail=""):
    global failed
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   -> {str(detail)[:300]}"))
    failed += (not cond)


# --------------------------------------------------------------------------- falsche Passerelle
def parse_multipart(body, ctype):
    boundary = ctype.split("boundary=")[1].encode()
    fields, file_bytes = {}, None
    for part in body.split(b"--" + boundary):
        if not part.startswith(b"\r\n") or b"\r\n\r\n" not in part:
            continue
        head, data = part[2:].split(b"\r\n\r\n", 1)
        data = data[:-2] if data.endswith(b"\r\n") else data
        name = head.decode().split('name="')[1].split('"')[0]
        if 'filename="' in head.decode():
            file_bytes = data
        else:
            fields.setdefault(name, []).append(data.decode())
    return fields, file_bytes


class FakeGateway:
    def __init__(self):
        self.records, self.lock, self.n = [], threading.Lock(), 0
        self.plan, self.always, self.latency = {}, None, (lambda n: 0.0)
        self.max_mb, self.language, self.error_text, self.redirect_to = 25, None, None, None
        self.words, self.active, self.max_active = False, 0, 0
        self.redirect_status = 307
        gw = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with gw.lock:
                    gw.n += 1
                    n = gw.n
                    gw.active += 1
                    gw.max_active = max(gw.max_active, gw.active)
                try:
                    time.sleep(gw.latency(n))
                    fields, file_bytes = parse_multipart(body, self.headers["Content-Type"])
                    wav = wave.open(io.BytesIO(file_bytes), "rb")
                    seconds = wav.getnframes() / wav.getframerate()
                    rec = {"n": n, "path": self.path, "auth": self.headers.get("Authorization"), "fields": fields, "size": len(file_bytes),
                           "seconds": seconds, "rate": wav.getframerate(), "channels": wav.getnchannels(), "sha": hashlib.sha1(file_bytes).hexdigest()[:8],
                           "pcm": np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("float32") / 32768.0}
                    with gw.lock:
                        gw.records.append(rec)
                    status = gw.always or gw.plan.get(n)
                    if gw.redirect_to:
                        self.send_response(gw.redirect_status); self.send_header("Location", gw.redirect_to); self.send_header("Content-Length", "0"); self.end_headers(); return
                    if status:
                        text = gw.error_text if gw.error_text is not None else "boom"
                        out = json.dumps({"error": {"message": text}}).encode()
                        self.send_response(status); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out); return
                    if self.headers.get("Authorization") != f"Bearer {KEY}":
                        out = b'{"error":{"message":"Authentication Error"}}'
                        self.send_response(401); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out); return
                    if len(file_bytes) > gw.max_mb * 1024 * 1024:
                        out = b'{"error":{"message":"Maximum file size exceeded"}}'
                        self.send_response(400); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out); return
                    segs = []
                    for k in range(max(1, int(seconds // 10) + (1 if seconds % 10 > 0.5 else 0))):
                        a, b = k * 10.0, min(seconds, k * 10.0 + 10.0)
                        segs.append({"id": k, "seek": 0, "start": a, "end": b, "text": f" C{rec['sha']}-{k}", "tokens": [1, 2], "temperature": 0.0,
                                     "avg_logprob": -0.1, "compression_ratio": 1.0, "no_speech_prob": None})
                    resp = {"text": " ".join(s["text"].strip() for s in segs), "language": (fields.get("language") or [self.language_for()])[0], "duration": seconds,
                            "segments": segs, "words": None}
                    if gw.words and "word" in fields.get("timestamp_granularities[]", []):
                        resp["words"] = [{"word": f"w{i}", "start": float(i), "end": float(i) + 0.9} for i in range(int(seconds))]
                    out = json.dumps(resp).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
                finally:
                    with gw.lock:
                        gw.active -= 1

            def language_for(self):
                return gw.language or "de"

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.shutdown()


# --------------------------------------------------------------------------- Hilfen
def make_wav(path, seconds, gaps=(), noise_only=False):
    t = np.arange(int(RATE * seconds)) / RATE
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))
    if noise_only:
        x = 0.0005 * np.random.RandomState(1).randn(len(t))
    for a, b in gaps:
        x[int(a * RATE):int(b * RATE)] = 0
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


ENV_KEYS = ["WHISPER_BACKEND", "GATEWAY_URL", "GATEWAY_API_KEY", "GATEWAY_MODEL", "GATEWAY_CHUNK_SECONDS", "GATEWAY_PARALLEL", "GATEWAY_TIMEOUT_S",
            "GATEWAY_RETRIES", "GATEWAY_RETRY_BACKOFF", "GATEWAY_FALLBACK", "GATEWAY_WORD_TIMESTAMPS", "GATEWAY_VAD", "GATEWAY_LANGUAGE", "GATEWAY_ALLOW_HTTP"]


def set_env(gw=None, **kw):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    base = {"WHISPER_BACKEND": "gateway", "GATEWAY_API_KEY": KEY, "GATEWAY_VAD": "false", "GATEWAY_RETRY_BACKOFF": "0"}
    if gw:
        base["GATEWAY_URL"] = gw.url
    base.update({k: str(v) for k, v in kw.items()})
    os.environ.update(base)


class LogCatcher(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []
    def emit(self, record):
        self.lines.append(record.getMessage())


catcher = LogCatcher()
logging.getLogger("whisper_server").addHandler(catcher)
logging.getLogger("whisper_server").setLevel(logging.DEBUG)

model = h.FakeModel()
mod = h.load_engine(ENGINE_PATH, "eng_gw", model=model, diarization=False)
tmp = tempfile.mkdtemp()


def run(path, cancel=None, kwargs=None):
    r = mod._TranscribeRun(path, dict(kwargs or {"beam_size": 5}), should_cancel=cancel)
    return list(r.iter_segments()), r


# --------------------------------------------------------------------------- Einstellungen
set_env(); os.environ.pop("WHISPER_BACKEND")
check("Ohne WHISPER_BACKEND bleibt es beim lokalen Weg", mod._gateway_config() is None, "")
set_env(GATEWAY_URL="https://litellm.example.org")
c = mod._gateway_config()
check("https-Adresse wird angenommen, der Pfad ergänzt", c and c["endpoint"] == "https://litellm.example.org/v1/audio/transcriptions", c)
set_env(GATEWAY_URL="https://litellm.example.org/v1/")
check("Adresse mit /v1 am Ende ergibt denselben Pfad", mod._gateway_config()["endpoint"] == "https://litellm.example.org/v1/audio/transcriptions", "")
set_env(GATEWAY_URL="https://litellm.example.org", GATEWAY_CHUNK_SECONDS=120, GATEWAY_PARALLEL=99)
c = mod._gateway_config()
check("Stückgröße wird auf höchstens 29 s begrenzt (ein Whisper-Fenster) und Parallelität auf 8", c["chunk_s"] == 29.0 and c["parallel"] == 8, c)
set_env(GATEWAY_URL="http://litellm.example.org")
check("http zu einem fremden Rechner wird abgelehnt (der Ton ginge unverschlüsselt)", mod._gateway_config() is None, "")
set_env(GATEWAY_URL="http://127.0.0.1:9")
check("http zum eigenen Rechner ist für Tests erlaubt", mod._gateway_config() is not None, "")
set_env(GATEWAY_URL="https://litellm.example.org"); os.environ.pop("GATEWAY_API_KEY")
check("Ohne Schlüssel wird der Gateway-Weg nicht benutzt", mod._gateway_config() is None, "")
set_env(GATEWAY_URL="ftp://litellm.example.org")
check("Andere Protokolle werden abgelehnt", mod._gateway_config() is None, "")

# --------------------------------------------------------------------------- Ablauf: Schnitt, Zeitstempel, Anfrage
gw = FakeGateway()
wav100 = make_wav(os.path.join(tmp, "a.wav"), 90, gaps=[(22, 23), (45, 46), (68, 69)])
set_env(gw, GATEWAY_LANGUAGE="de")
segs, r = run(wav100)
recs = sorted(gw.records, key=lambda x: x["n"])
check("90 s Audio werden in 4 Stücke zerlegt, jedes höchstens 28 s", len(recs) == 4 and all(x["seconds"] <= 28.001 for x in recs), [round(x["seconds"], 1) for x in recs])
check("Zusammen ergeben die Stücke genau die ganze Aufnahme (nichts fehlt, nichts doppelt)", abs(sum(x["seconds"] for x in recs) - 90) < 0.01, sum(x["seconds"] for x in recs))
check("Jede Anfrage trägt den Schlüssel als Bearer-Kopfzeile", all(x["auth"] == f"Bearer {KEY}" for x in recs), recs[0]["auth"])
check("Pfad, Modell und Antwortformat stimmen", all(x["path"] == "/v1/audio/transcriptions" and x["fields"]["model"] == ["whisper-large-v3-turbo"] and x["fields"]["response_format"] == ["verbose_json"] for x in recs), recs[0]["fields"])
check("Die Stücke sind gültiges WAV: 16 kHz, mono, unter 1,5 MB", all(x["rate"] == 16000 and x["channels"] == 1 and x["size"] < 1_500_000 for x in recs), [x["size"] for x in recs])
pcm_by_len = sorted(recs, key=lambda x: x["pcm"].shape[0])
# Schnitt in der Stille: Ende/Anfang der Stücke ist leise (Reihenfolge der Stücke rekonstruieren über Audio)
audio_full = np.frombuffer(wave.open(wav100).readframes(100 * RATE), dtype="<i2").astype("float32") / 32768.0
pos, cuts_quiet = 0, True
for x in sorted(recs, key=lambda x: x["n"]):
    pass
starts = []
p = 0
sequence = [x for x in recs]
# Anhand der Texte (Prüfsumme) die Reihenfolge der gelieferten Segmente bestimmen
texts = [s.text.strip() for s in segs]
check("Die Segmente kommen in der richtigen Reihenfolge, die Zeiten wachsen ständig", all(segs[i].end <= segs[i + 1].start + 1e-6 for i in range(len(segs) - 1)) and segs[0].start == 0.0, [(round(s.start, 1), round(s.end, 1)) for s in segs][:6])
check("Das Ende des letzten Segments liegt bei der Länge der Aufnahme (±0,1 s)", abs(segs[-1].end - 90) < 0.1, segs[-1].end)
ids = [s.id for s in segs]
check("Die Segment-Nummern laufen lückenlos von 0 aufwärts", ids == list(range(len(segs))), ids[:8])
cut_points = []
acc = 0.0
for x in recs:
    acc += x["seconds"]
    cut_points.append(acc)
cut_points = cut_points[:-1]
check("Geschnitten wird in den Pausen (22-23 s, 45-46 s, 68-69 s)", all(any(a - 0.05 <= cp <= b + 0.05 for a, b in [(22, 23), (45, 46), (68, 69)]) for cp in cut_points), cut_points)
check("Zeitstempel sind um den Start des Stücks verschoben (zweites Stück beginnt bei ca. 22-23 s)", 22 <= [s for s in segs if s.start > 21][0].start <= 23.05, [round(s.start, 2) for s in segs][:5])
check("Sprache aus GATEWAY_LANGUAGE wird bei jeder Anfrage mitgeschickt", all(x["fields"].get("language") == ["de"] for x in recs), [x["fields"].get("language") for x in recs])
check("Das Ergebnis meldet Sprache und Dauer", r.info.language == "de" and abs(r.info.duration - 90) < 0.01, (r.info.language, r.info.duration))
check("Das lokale Modell wurde nicht gebraucht", model.calls == [], len(model.calls))

# --------------------------------------------------------------------------- Sprache bestimmen (Abstimmung)
gw.records.clear()
set_env(gw)
run(wav100)
recs = sorted(gw.records, key=lambda x: x["n"])
langs = [x["fields"].get("language") for x in recs]
check("Ohne Vorgabe: die ersten drei Stücke ohne Sprache, danach die abgestimmte Sprache (de)", all(l is None for l in langs[:3]) and langs[3] == ["de"], langs)
gw.records.clear(); set_env(gw)
run(wav100, kwargs={"beam_size": 5, "language": "en"})
check("Eine vorgegebene Sprache (en) wird bei allen Stücken benutzt", all(x["fields"].get("language") == ["en"] for x in gw.records), [x["fields"].get("language") for x in gw.records])

# --------------------------------------------------------------------------- Parallelität und Reihenfolge
gw.records.clear(); gw.max_active = 0
rnd = random.Random(5)
gw.latency = lambda n: rnd.uniform(0.0, 0.25)
set_env(gw, GATEWAY_LANGUAGE="de", GATEWAY_PARALLEL=3)
wav_long = make_wav(os.path.join(tmp, "b.wav"), 300, gaps=[(a, a + 1) for a in range(22, 300, 23)])
t0 = time.time(); segs, r = run(wav_long); dt = time.time() - t0
gw.latency = lambda n: 0.0
check("Mehrere Anfragen laufen gleichzeitig (mindestens 2, höchstens 3)", 2 <= gw.max_active <= 3, gw.max_active)
check("Trotz zufälliger Antwortzeiten bleibt die Reihenfolge stimmig (Zeiten wachsen)", all(segs[i].start <= segs[i + 1].start + 1e-6 for i in range(len(segs) - 1)) and abs(segs[-1].end - 300) < 0.2, (len(segs), segs[-1].end))
audio300 = np.frombuffer(wave.open(wav_long).readframes(300 * RATE), dtype="<i2").astype("float32") / 32768.0
check("Jedes Stück wurde genau einmal geschickt, zusammen genau 300 s", len(gw.records) == len(mod._gw_spans(audio300, 28.0)) and abs(sum(x["seconds"] for x in gw.records) - 300) < 0.01, (len(gw.records), sum(x["seconds"] for x in gw.records)))

# --------------------------------------------------------------------------- Fehler, Wiederholung, Rückfall
wav40 = make_wav(os.path.join(tmp, "c.wav"), 40, gaps=[(22, 23)])
gw.records.clear(); gw.n = 0; gw.plan = {1: 500}; set_env(gw, GATEWAY_LANGUAGE="de", GATEWAY_RETRIES=2)
segs, _ = run(wav40)
check("Ein Fehler 500 wird wiederholt und dann gelingt es (alles über das Gateway)", len(gw.records) == 3 and model.calls == [] and all("C" in s.text for s in segs), (len(gw.records), len(model.calls)))
gw.records.clear(); gw.n = 0; gw.plan = {1: 429}
segs, _ = run(wav40)
check("Auch 429 (zu viele Anfragen) wird wiederholt", len(gw.records) == 3 and model.calls == [], len(gw.records))
gw.plan = {}; gw.records.clear(); gw.always = 401; gw.n = 0; model.calls.clear(); catcher.lines.clear()
set_env(gw)  # ohne Sprachvorgabe: die ersten Stücke laufen nacheinander
segs, _ = run(wav40)
check("401 wird nicht wiederholt (nur eine Anfrage), danach läuft der ganze Rest lokal", len(gw.records) == 1 and len(model.calls) == 2 and all("Satz" in s.text for s in segs), (len(gw.records), len(model.calls)))
gw.records.clear(); gw.n = 0; model.calls.clear(); catcher.lines.clear()
set_env(gw, GATEWAY_LANGUAGE="de")  # mit Sprachvorgabe laufen die Stücke gleichzeitig
segs, _ = run(wav40)
check("Gleichzeitige Stücke: jedes Stück wird höchstens einmal angefragt (kein Wiederholen bei 401), alle laufen am Ende lokal", len(gw.records) <= 2 and len(model.calls) == 2, (len(gw.records), len(model.calls)))
check("Der Fehler wird nur einmal gemeldet, nicht für jedes Stück", sum("Gateway failed" in l for l in catcher.lines) == 1, [l for l in catcher.lines if "failed" in l])
check("Die lokalen Segmente beginnen an der richtigen Stelle der Aufnahme (zweites Stück ab ca. 22 s)", max(s.start for s in segs) > 22, max(s.start for s in segs))
check("Der Rückfall wird im Protokoll gemeldet", any("Gateway failed" in l for l in catcher.lines), catcher.lines[-3:])
set_env(gw, GATEWAY_LANGUAGE="de", GATEWAY_FALLBACK="false"); gw.records.clear(); gw.n = 0
try:
    run(wav40); raised = None
except mod._GatewayError as e:
    raised = str(e)
check("Ohne Rückfall kommt ein Fehler mit verständlichem Text", raised is not None and "HTTP 401" in raised, raised)
gw.always = 400; gw.error_text = f"Bad key {KEY} for user"; gw.records.clear()
try:
    run(wav40); raised = None
except mod._GatewayError as e:
    raised = str(e)
check("Steht der Schlüssel in der Fehlerantwort, wird er unkenntlich gemacht", raised is not None and KEY not in raised and "***" in raised, raised)
gw.always = None; gw.error_text = None

# --------------------------------------------------------------------------- Weiterleitung wird abgelehnt
for status in (301, 302, 303, 307, 308):
    other = FakeGateway()
    gw.redirect_to, gw.redirect_status = other.url + "/geklaut", status
    gw.records.clear(); gw.n = 0
    try:
        run(wav40); raised = None
    except mod._GatewayError as e:
        raised = str(e)
    check(f"Weiterleitung {status} wird nicht befolgt: Fehler, und der Schlüssel erreicht keinen anderen Rechner", raised is not None and str(status) in raised and other.records == [] and other.n == 0, (raised, other.n))
    other.stop()
gw.redirect_to = None
set_env(gw, GATEWAY_LANGUAGE="de")

# --------------------------------------------------------------------------- Abbrechen
gw.records.clear(); gw.n = 0; calls = {"n": 0}
def cancel():
    calls["n"] += 1
    return len(gw.records) >= 2
gw.latency = lambda n: 0.05
try:
    run(wav_long, cancel=cancel); cancelled = False
except mod._Cancelled:
    cancelled = True
gw.latency = lambda n: 0.0
time.sleep(0.4)
check("Abbrechen stoppt den Ablauf (_Cancelled) und schickt nicht mehr alle 13 Stücke", cancelled and len(gw.records) < 13, (cancelled, len(gw.records)))

# --------------------------------------------------------------------------- Protokoll: keine Geheimnisse, kein Text
set_env(gw, GATEWAY_LANGUAGE="de"); catcher.lines.clear(); gw.records.clear(); gw.n = 0
run(wav40)
alllines = "\n".join(catcher.lines)
check("Der Schlüssel steht in keiner Protokollzeile", KEY not in alllines, "")
check("Der erkannte Text steht in keiner Protokollzeile", "C" + gw.records[0]["sha"] not in alllines, "")
check("Das Protokoll nennt Abschnitte und Ergebnis", "Gateway |" in alllines, alllines[-300:])

# --------------------------------------------------------------------------- Umwandlung der Antwort
os.environ["GATEWAY_VAD"] = "false"
seg_only_text = mod._gw_segments({"text": " Nur Text hier."}, 100.0, 20.0, False)
check("Antwort nur mit Text: ein Segment über das ganze Stück, um den Start verschoben", len(seg_only_text) == 1 and seg_only_text[0].start == 100.0 and seg_only_text[0].end == 120.0, seg_only_text)
check("Leere Segmente werden übersprungen", mod._gw_segments({"segments": [{"start": 0, "end": 1, "text": "  "}, {"start": 1, "end": 2, "text": "ok"}]}, 0.0, 5.0, False)[0].text.strip() == "ok", "")
try:
    mod._gw_segments({"segments": [{"start": "x", "end": 1, "text": "a"}]}, 0.0, 5.0, False); bad = False
except mod._GatewayError:
    bad = True
check("Kaputte Zeitangaben ergeben einen Gateway-Fehler (und damit den Rückfall)", bad, "")
w = mod._gw_segments({"segments": [{"start": 0, "end": 2, "text": "a b"}], "words": [{"word": "a", "start": 0.1, "end": 0.5}, {"word": "b", "start": 1.0, "end": 1.5}, {"word": "z", "start": 9.0, "end": 9.5}]}, 10.0, 5.0, True)
check("Wörter werden dem Segment zugeordnet und verschoben (nur die, die hineingehören)", [x.word.strip() for x in w[0].words] == ["a", "b"] and abs(w[0].words[0].start - 10.1) < 1e-6, w[0].words)
check("Ohne Anforderung der Wörter bleibt words leer", mod._gw_segments({"segments": [{"start": 0, "end": 2, "text": "a"}], "words": [{"word": "a", "start": 0, "end": 1}]}, 0.0, 5.0, False)[0].words is None, "")
kept = mod._gw_filter_by_speech(mod._gw_segments({"segments": [{"start": 0, "end": 4, "text": "Sprache"}, {"start": 6, "end": 9, "text": "Vielen Dank fürs Zuschauen"}]}, 0.0, 10.0, False), [(0.0, 3.5)], 0.0)
check("Segmente in der Stille werden verworfen, gesprochene bleiben", [s.text.strip() for s in kept] == ["Sprache"], kept)

# --------------------------------------------------------------------------- Sprachaktivität (echte Erkennung)
gw.records.clear(); gw.n = 0
set_env(gw, GATEWAY_VAD="true", GATEWAY_LANGUAGE="de")
silent = make_wav(os.path.join(tmp, "s.wav"), 40, noise_only=True)
segs, _ = run(silent)
check("Stille (nur Rauschen): kein Stück wird verschickt und es kommt kein Text", len(gw.records) == 0 and segs == [], (len(gw.records), len(segs)))
speech_path = os.environ.get("TEST_SPEECH_WAV")
if speech_path and os.path.exists(speech_path):
    gw.records.clear(); gw.n = 0
    segs, r = run(speech_path)
    check("Echte Sprache wird erkannt und verschickt", len(gw.records) >= 1 and len(segs) >= 1, (len(gw.records), len(segs)))
    check("duration_after_vad ist kleiner oder gleich der Dauer", 0 < r.info.duration_after_vad <= r.info.duration + 0.01, (r.info.duration_after_vad, r.info.duration))
else:
    print("SKIP  Test mit echter Sprache (TEST_SPEECH_WAV nicht gesetzt)")

# --------------------------------------------------------------------------- über die API (Antwort, Sprecher)
set_env(gw, GATEWAY_LANGUAGE="de"); gw.records.clear(); gw.n = 0; model.calls.clear()
async def api():
    async with h.client(mod) as c:
        r = await c.post("/v1/audio/transcriptions", files={"file": ("a.wav", open(wav40, "rb").read(), "audio/wav")}, data={"response_format": "verbose_json"})
        return r
r = asyncio.run(api())
body = r.json()
check("Über die API: Antwort 200 mit dem Text des Gateways", r.status_code == 200 and "C" + gw.records[0]["sha"] in body["text"], (r.status_code, r.text[:200]))
check("Über die API: der Schlüssel steht nirgends in der Antwort", KEY not in r.text, "")
check("Über die API: das lokale Modell wurde nicht gebraucht", model.calls == [], len(model.calls))

fake_d = h.make_fake_diarizer()
mod2 = h.load_engine(ENGINE_PATH, "eng_gw2", model=h.FakeModel(), diarize_module=fake_d, diarization=True)
set_env(gw, GATEWAY_LANGUAGE="de")
r2 = mod2._TranscribeRun(wav40, {"beam_size": 5, "word_timestamps": True}); segs2 = list(r2.iter_segments())
speaker_map, refined = mod2._diarize_segments(wav40, segs2, None)
check("Sprecher ohne Wörter: Zuordnung pro Segment (Rückfallweg), keine Fehler", refined is None and speaker_map and len(speaker_map) == len(segs2), (refined, speaker_map))
gw.words = True
set_env(gw, GATEWAY_LANGUAGE="de", GATEWAY_WORD_TIMESTAMPS="true")
r3 = mod2._TranscribeRun(wav40, {"beam_size": 5, "word_timestamps": True}); segs3 = list(r3.iter_segments())
speaker_map3, refined3 = mod2._diarize_segments(wav40, segs3, None)
check("Mit Wörtern vom Gateway (GATEWAY_WORD_TIMESTAMPS=true): wortgenaue Zuordnung", refined3 and all("speaker" in x for x in refined3), refined3)
check("Die Wortzeiten im zweiten Stück sind um dessen Start verschoben", any(w.start > 22 for s in segs3 for w in (s.words or [])), [round(w.start, 1) for s in segs3 for w in (s.words or [])][:5])

gw.stop()
print("\n" + (f"{failed} Prüfung(en) FEHLGESCHLAGEN" if failed else "Alle Prüfungen bestanden"))
sys.exit(1 if failed else 0)
