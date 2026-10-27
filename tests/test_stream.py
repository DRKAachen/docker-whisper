"""Tests für Streaming (Fortschritt, Zeitstempel, Sprecher am Ende) und Abbrechen im Moteur.
Es läuft ein echter uvicorn-Server, damit Streaming wirklich schrittweise ankommt.
Aufruf: python3 test_stream.py"""
import asyncio
import json
import os
import socket
import sys
import time

import httpx
import uvicorn

import harness as h

ENGINE_PATH = next(p for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api_server.py"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_server.py")) if os.path.exists(p))

AUDIO = h.wav_bytes(5)
failed = 0


def check(name, cond, detail=""):
    global failed
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond:
        failed += 1


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, mod):
        self.mod, self.port = mod, free_port()
        self.server = uvicorn.Server(uvicorn.Config(mod.app, host="127.0.0.1", port=self.port, log_level="critical", lifespan="off"))
        self.task = None

    async def __aenter__(self):
        self.task = asyncio.create_task(self.server.serve())
        for _ in range(100):
            if self.server.started:
                break
            await asyncio.sleep(0.05)
        self.base = f"http://127.0.0.1:{self.port}"
        return self

    async def __aexit__(self, *a):
        self.server.should_exit = True
        await self.task


def new_engine(model=None, diarizer=None, diarization=True):
    return h.load_engine(ENGINE_PATH, "eng_stream", model=model or h.FakeModel(),
                         diarize_module=diarizer or h.make_fake_diarizer(), diarization=diarization)


async def sse(client, base, data=None, headers=None, files=None, on_event=None, stop_after=None):
    """Startet einen Streaming-Auftrag und liefert die Ereignisse als Liste. on_event(ev, events) darf async sein."""
    events = []
    data = {"stream": "true", **(data or {})}
    files = files or {"file": ("t.wav", AUDIO, "audio/wav")}
    async with client.stream("POST", base + "/v1/audio/transcriptions", data=data, files=files, headers=headers or {}) as r:
        events.append({"_status": r.status_code, "_ctype": r.headers.get("content-type", "")})
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            ev = "[DONE]" if payload == "[DONE]" else json.loads(payload)
            events.append(ev)
            if on_event:
                res = on_event(ev, events)
                if asyncio.iscoroutine(res):
                    await res
            if stop_after and stop_after(ev):
                break
    return events


def types(events):
    return [e if isinstance(e, str) else e.get("type", "error" if "error" in e else "_meta") for e in events]


async def main():
    # ---------- 1) Ablauf mit Sprechern ----------
    eng = new_engine()
    async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
        ev = await sse(c, srv.base, {"diarize": "true", "num_speakers": "2"}, {"X-Job-Id": "job-1"})
        t = types(ev)
        check("Streaming: Status 200 und text/event-stream", ev[0]["_status"] == 200 and "text/event-stream" in ev[0]["_ctype"], str(ev[0]))
        check("Streaming: Reihenfolge der Ereignisse (progress, 5x delta, phase, result, done, [DONE])",
              t[1:] == ["transcript.progress"] + ["transcript.text.delta"] * 5 + ["transcript.phase", "transcript.result", "transcript.text.done", "[DONE]"], str(t))
        prog = ev[1]
        check("Fortschritt: Gesamtlänge der Datei (5 s) kommt VOR dem ersten Text", prog["duration"] == 5.0, str(prog))
        deltas = [e for e in ev if isinstance(e, dict) and e.get("type") == "transcript.text.delta"]
        check("Zeitstempel je Segment, auch über Abschnitte hinweg richtig verschoben (0-1, 1-2 … 4-5 s)",
              [(d["start"], d["end"]) for d in deltas] == [(float(i), i + 1.0) for i in range(5)], str([(d["start"], d["end"]) for d in deltas]))
        check("Delta-Text wie bisher: erstes ohne, weitere mit führendem Leerzeichen",
              not deltas[0]["delta"].startswith(" ") and all(d["delta"].startswith(" ") for d in deltas[1:]), str([d["delta"] for d in deltas]))
        phase = next(e for e in ev if isinstance(e, dict) and e.get("type") == "transcript.phase")
        check("Sprecher-Phase wird angekündigt", phase["phase"] == "diarize", str(phase))
        result = next(e for e in ev if isinstance(e, dict) and e.get("type") == "transcript.result")["result"]
        check("Ergebnis enthält Sprecher", {s["speaker"] for s in result["segments"]} == {"SPEAKER_00", "SPEAKER_01"}, str(result["segments"]))
        check("Sprecher-Modul bekam die gewünschte Anzahl (2)", ("diarize", 2) in eng_diar_calls(eng), str(eng_diar_calls(eng)))
        done = next(e for e in ev if isinstance(e, dict) and e.get("type") == "transcript.text.done")
        check("transcript.text.done enthält den vollen Text", done["text"] == result["text"] and "Satz 1." in done["text"], str(done))

    # ---------- 2) Ergebnis des Streams == Ergebnis des normalen Ablaufs ----------
    a = new_engine()
    async with Server(a) as srv, httpx.AsyncClient(timeout=30) as c:
        ev = await sse(c, srv.base, {"diarize": "true"})
        streamed = next(e for e in ev if isinstance(e, dict) and e.get("type") == "transcript.result")["result"]
    b = new_engine()
    async with h.client(b) as c:
        r = await c.post("/v1/audio/transcriptions", files={"file": ("t.wav", AUDIO, "audio/wav")}, data={"response_format": "verbose_json", "diarize": "true"})
    check("Ergebnis des Streams ist IDENTISCH mit dem Ergebnis des normalen Ablaufs (verbose_json)", streamed == r.json(), f"{json.dumps(streamed)[:200]} vs {json.dumps(r.json())[:200]}")

    # ---------- 3) Ohne Sprecher ----------
    eng = new_engine()
    async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
        ev = await sse(c, srv.base, {"diarize": "false"})
        t = types(ev)
        res = next(e for e in ev if isinstance(e, dict) and e.get("type") == "transcript.result")["result"]
        check("Ohne Sprecher: keine Sprecher-Phase, Sprecher-Modul wird nicht aufgerufen, keine Sprecher im Ergebnis",
              "transcript.phase" not in t and not eng_diar_calls(eng) and all("speaker" not in s for s in res["segments"]), f"{t} {eng_diar_calls(eng)}")
        check("Ohne Sprecher: Modell lief ohne Wortzeiten (schneller)", all(not kw.get("word_timestamps") for _, kw in eng._model.calls), str(eng._model.calls[0][1]))

    # ---------- 4) Abbrechen mitten im Lauf ----------
    model = h.FakeModel(delay=0.25)
    eng = new_engine(model=model)
    async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
        cancel_resp = {}

        async def on_ev(e, events):
            if isinstance(e, dict) and e.get("type") == "transcript.text.delta" and "sent" not in cancel_resp:
                cancel_resp["sent"] = True
                r = await c.post(srv.base + "/v1/jobs/job-cancel/cancel")
                cancel_resp["r"] = r.json()

        t0 = time.time()
        ev = await sse(c, srv.base, {"diarize": "true"}, {"X-Job-Id": "job-cancel"}, on_event=on_ev)
        took = time.time() - t0
        t = types(ev)
        check("Abbrechen: die Route antwortet {cancelled: true}", cancel_resp.get("r") == {"cancelled": True}, str(cancel_resp))
        check("Abbrechen: Stream endet mit transcript.cancelled, ohne Ergebnis und ohne Sprecher-Phase",
              t[-1] == "transcript.cancelled" and "transcript.result" not in t and "transcript.phase" not in t, str(t))
        check("Abbrechen: es wurden NICHT alle 5 Segmente erzeugt", model.counter < 5, f"counter={model.counter}")
        check("Abbrechen: schnell (nach dem laufenden Segment, nicht nach der ganzen Datei)", took < 1.2, f"{took:.2f}s")
        r2 = await c.post(srv.base + "/v1/jobs/job-cancel/cancel")
        check("Nach dem Abbruch ist die Nummer wieder frei (zweiter Abbruch: cancelled=false)", r2.json() == {"cancelled": False}, str(r2.json()))
        # die Sperre ist frei: ein neuer Auftrag läuft sofort durch
        model.delay = 0
        ev2 = await sse(c, srv.base, {"diarize": "false"})
        check("Nach dem Abbruch läuft ein neuer Auftrag normal durch (Sperre wurde freigegeben)", types(ev2)[-1] == "[DONE]", str(types(ev2)))

    # ---------- 5) Abbrechen, solange der Auftrag noch auf die Sperre wartet ----------
    model = h.FakeModel()
    eng = new_engine(model=model)
    async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
        eng._inference_lock.acquire()  # ein anderer Auftrag "rechnet gerade"
        try:
            task = asyncio.create_task(sse(c, srv.base, {"diarize": "true"}, {"X-Job-Id": "job-wait"}))
            await asyncio.sleep(0.6)
            r = await c.post(srv.base + "/v1/jobs/job-wait/cancel")
            check("Wartender Auftrag lässt sich abbrechen", r.json() == {"cancelled": True}, str(r.json()))
        finally:
            eng._inference_lock.release()
        ev = await asyncio.wait_for(task, 10)
        check("Wartender Auftrag: endet abgebrochen, das Modell wurde nie aufgerufen", types(ev)[-1] == "transcript.cancelled" and not model.calls, f"{types(ev)} calls={len(model.calls)}")

    # ---------- 6) Abbrechen während der Sprecher-Zuordnung ----------
    eng = new_engine(diarizer=h.make_fake_diarizer(delay=0.8))
    async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
        async def on_phase(e, events):
            if isinstance(e, dict) and e.get("type") == "transcript.phase":
                await c.post(srv.base + "/v1/jobs/job-diar/cancel")
        ev = await sse(c, srv.base, {"diarize": "true"}, {"X-Job-Id": "job-diar"}, on_event=on_phase)
        t = types(ev)
        check("Abbrechen in der Sprecher-Phase: endet abgebrochen und liefert KEIN Ergebnis", t[-1] == "transcript.cancelled" and "transcript.result" not in t, str(t))

    # ---------- 7) Verbindung trennen (Client weg) stoppt die Arbeit ----------
    model = h.FakeModel(delay=0.25)
    eng = new_engine(model=model)
    async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
        await sse(c, srv.base, {"diarize": "false"}, {"X-Job-Id": "job-gone"},
                  stop_after=lambda e: isinstance(e, dict) and e.get("type") == "transcript.text.delta")
        await asyncio.sleep(1.8)  # Zeit, in der der Rest (4 weitere Segmente x 0,25 s) fertig wäre
        check("Client trennt die Verbindung: der Worker stoppt, es werden nicht alle Segmente berechnet", model.counter < 5, f"counter={model.counter}")
        got = eng._inference_lock.acquire(blocking=False)
        check("Client trennt die Verbindung: die Sperre ist wieder frei", got, "Sperre bleibt belegt")
        if got:
            eng._inference_lock.release()
        check("Client trennt die Verbindung: die Nummer ist aufgeräumt", "job-gone" not in eng._cancel_events, str(eng._cancel_events))

    # ---------- 8) Fehler: kaputte Datei ----------
    eng = new_engine()
    async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
        ev = await sse(c, srv.base, {"diarize": "true"}, files={"file": ("kaputt.mp3", b"kein audio" * 100, "audio/mpeg")})
        err = ev[-1]
        check("Kaputte Datei im Stream: Fehler-Ereignis mit Text", isinstance(err, dict) and "error" in err and "Invalid data" in err["error"]["message"], str(err))
        check("Nach dem Fehler ist die Sperre frei", eng._inference_lock.acquire(blocking=False) and (eng._inference_lock.release() or True), "")

    # ---------- 9) Auftragsnummern und Schlüssel ----------
    eng = new_engine()
    async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
        r = await c.post(srv.base + "/v1/jobs/unbekannt-123/cancel")
        check("Unbekannte Nummer: cancelled=false (kein Fehler)", r.status_code == 200 and r.json() == {"cancelled": False}, str(r))
        r = await c.post(srv.base + "/v1/jobs/a$b/cancel")
        check("Ungültige Nummer wird abgelehnt (400)", r.status_code == 400, str(r.status_code))
        ev = await sse(c, srv.base, {"diarize": "false"}, {"X-Job-Id": "../../etc/passwd"})
        check("Ungültige Nummer im Kopf wird ignoriert: der Auftrag läuft trotzdem normal und ohne Eintrag", types(ev)[-1] == "[DONE]" and not eng._cancel_events, f"{types(ev)[-1]} {eng._cancel_events}")
    import os
    os.environ["WHISPER_API_KEY"] = "geheim"
    try:
        eng = new_engine()
        async with Server(eng) as srv, httpx.AsyncClient(timeout=30) as c:
            r1 = await c.post(srv.base + "/v1/jobs/x/cancel")
            r2 = await c.post(srv.base + "/v1/jobs/x/cancel", headers={"Authorization": "Bearer geheim"})
            check("Abbruch-Route verlangt den API-Schlüssel (ohne: 401, mit: 200)", r1.status_code == 401 and r2.status_code == 200, f"{r1.status_code}/{r2.status_code}")
    finally:
        os.environ.pop("WHISPER_API_KEY", None)

    print("\n" + (f"{failed} Prüfung(en) FEHLGESCHLAGEN" if failed else "Alle Prüfungen bestanden"))
    sys.exit(1 if failed else 0)


def eng_diar_calls(eng):
    import sys as _s
    return getattr(_s.modules["diarizer"], "calls", [])


asyncio.run(main())
