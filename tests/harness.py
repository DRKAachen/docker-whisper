"""Testumgebung für api_server.py ohne echtes Whisper-Modell.

Das Modell und das Sprecher-Modul werden durch Attrappen ersetzt. Die Audio-Dekodierung (PyAV),
die Aufteilung in Abschnitte, die Zeitstempel-Verschiebung und die Antwortformate sind echter Code.
"""
import asyncio
import dataclasses
import importlib.util
import io
import math
import os
import struct
import sys
import time
import types
import wave

os.environ["WHISPER_CHUNK_THRESHOLD_SECONDS"] = "2"   # 5-s-Testdatei wird in Abschnitten verarbeitet
os.environ["WHISPER_CHUNK_SIZE_SECONDS"] = "2"
os.environ.pop("WHISPER_API_KEY", None)

import httpx
from faster_whisper.transcribe import Segment, Word


@dataclasses.dataclass
class FakeInfo:
    language: str = "de"
    language_probability: float = 0.99
    duration: float = 0.0
    duration_after_vad: float = 0.0


class FakeModel:
    """Liefert pro Sekunde Audio ein Segment ("Satz N") mit einem Wort. Optional mit Verzögerung."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []
        self.counter = 0

    def transcribe(self, audio, **kw):
        n_sec = int(len(audio) / 16000)
        self.calls.append((len(audio), dict(kw)))
        info = FakeInfo(duration=len(audio) / 16000, duration_after_vad=len(audio) / 16000 * 0.9)

        def gen():
            for i in range(n_sec):
                if self.delay:
                    time.sleep(self.delay)
                self.counter += 1
                words = None
                if kw.get("word_timestamps"):
                    words = [Word(start=float(i), end=float(i) + 0.9, word=f" Satz{self.counter}", probability=0.9)]
                yield Segment(
                    id=i, seek=0, start=float(i), end=float(i) + 1.0, text=f" Satz {self.counter}.",
                    tokens=[1, 2], avg_logprob=-0.2, compression_ratio=1.1, no_speech_prob=0.01,
                    words=words, temperature=0.0,
                )

        return gen(), info


def make_fake_diarizer(delay=0.0, fail=False):
    mod = types.ModuleType("diarizer")
    mod.calls = []

    def diarize(path, num_speakers=None):
        mod.calls.append(("diarize", num_speakers))
        if delay:
            time.sleep(delay)
        if fail:
            raise RuntimeError("Diarisierung kaputt")
        return [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}, {"start": 2.5, "end": 999.0, "speaker": "SPEAKER_01"}]

    def resegment_by_word(segments, turns):
        out = []
        any_words = False
        for seg in segments:
            for w in (getattr(seg, "words", None) or []):
                any_words = True
                mid = (w.start + w.end) / 2
                spk = next(t["speaker"] for t in turns if t["start"] <= mid < t["end"])
                text = w.word.strip()
                if out and out[-1]["speaker"] == spk:
                    out[-1]["text"] += " " + text
                    out[-1]["end"] = w.end
                else:
                    out.append({"start": w.start, "end": w.end, "text": text, "speaker": spk})
        return out if any_words else None

    def assign_speakers(segs, turns):
        for s in segs:
            s["speaker"] = "SPEAKER_00"
        return segs

    mod.diarize, mod.resegment_by_word, mod.assign_speakers = diarize, resegment_by_word, assign_speakers
    return mod


def wav_bytes(seconds=5, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(rate * seconds))
        w.writeframes(frames)
    return buf.getvalue()


def load_engine(path, name, model=None, diarize_module=None, diarization=True):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # lädt das Modell NICHT (das passiert erst im Lifespan, den wir nicht starten)
    mod._model = model or FakeModel()
    mod._model_name = "fake"
    mod._diarization_enabled = diarization
    sys.modules["diarizer"] = diarize_module or make_fake_diarizer()
    return mod


def client(mod):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=mod.app), base_url="http://engine", timeout=60)
