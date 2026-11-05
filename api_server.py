#!/usr/bin/env python3
"""
Whisper Speech-to-Text API Server
Provides OpenAI-compatible /v1/audio/transcriptions and
/v1/audio/translations endpoints powered by faster-whisper.

https://github.com/hwdsl2/docker-whisper

Copyright (C) 2026 Lin Song <linsongui@gmail.com>

This work is licensed under the MIT License
See: https://opensource.org/licenses/MIT
"""

import asyncio
import dataclasses
import gc
import hmac
import inspect
import io
import json
import logging
import os
import re
import socket
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_level_str = os.environ.get("WHISPER_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whisper_server")

# ---------------------------------------------------------------------------
# Model — loaded once at startup via the FastAPI lifespan hook
# ---------------------------------------------------------------------------

_model = None       # WhisperModel instance
_model_name = None  # name as loaded (e.g. "base")
_beam_size = 5      # beam size used for transcription
_max_request_beam = 10  # maximum per-request beam override; 0 disables the cap
_word_timestamps = False  # default for word-level timestamps
# Gegen Wiederholungsschleifen ("... bei der Kuhnjahme bei der Kuhnjahme ..."): Whisper darf bei unsicheren oder zu stark
# wiederholten Abschnitten mit höherer Temperatur neu ansetzen, und ein misslungener Abschnitt wird nicht zur Vorlage für den
# nächsten. WHISPER_CONDITION_ON_PREVIOUS_TEXT=true stellt das frühere Verhalten (Text davor als Vorlage) wieder her.
_condition_on_previous = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").strip().lower() == "true"
_TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # dieselben Stufen wie im Standard von faster-whisper
_diarization_enabled = False  # set via WHISPER_DIARIZATION=true
_max_upload_bytes = 1024 * 1024 * 1024  # 1 GiB by default; 0 disables the limit
# Obergrenze der Audiodauer in Minuten (0 = unbegrenzt). Die ganze Datei wird als Zahlenreihe in den Speicher dekodiert: eine
# stark komprimierte Datei von vielen Stunden würde den Speicher füllen und das Moteur beenden.
_max_duration_min = 240
_UPLOAD_CHUNK_SIZE = 1024 * 1024
_UNSUPPORTED_DIARIZE_MODEL = "gpt-4o-transcribe-diarize"

# Serialise all inference calls (batch and streaming) so that CTranslate2 is
# never called concurrently from multiple threads.
_inference_lock = threading.Lock()

# Abbrechen laufender Streaming-Aufträge: Auftragsnummer (Kopfzeile X-Job-Id) -> Event.
# Der Streaming-Weg prüft das Event nach jedem Segment und zwischen den Abschnitten.
_cancel_events: dict = {}
_cancel_lock = threading.Lock()
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


class _Cancelled(Exception):
    """Der Auftrag wurde auf Wunsch des Clients abgebrochen."""


def _clean_job_id(value: Optional[str]) -> Optional[str]:
    """Nur unauffällige Auftragsnummern zulassen (keine Sonderzeichen in Logs oder Verzeichnissen)."""
    if value and _JOB_ID_RE.match(value):
        return value
    return None


def _register_cancel(job_id: Optional[str]) -> threading.Event:
    event = threading.Event()
    if job_id:
        with _cancel_lock:
            _cancel_events[job_id] = event
    return event


def _unregister_cancel(job_id: Optional[str], event: threading.Event) -> None:
    if job_id:
        with _cancel_lock:
            if _cancel_events.get(job_id) is event:
                del _cancel_events[job_id]


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` when unset or empty.

    ``os.environ.get(name, default)`` only applies the default when the variable
    is absent. Because run.sh exports config variables unconditionally, an unset
    option arrives as an empty string, which would otherwise raise ValueError in
    int(). Treat empty/whitespace values as "use the default".
    """
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        logger.error(
            "Invalid value for %s: %r (expected an integer); using default %s",
            name, val, default,
        )
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` when unset or empty."""
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        logger.error(
            "Invalid value for %s: %r (expected a number); using default %s",
            name, val, default,
        )
        return default


def _load_diarizer() -> None:
    """Conditionally load the sherpa-onnx diarization pipeline."""
    global _diarization_enabled
    if os.environ.get("WHISPER_DIARIZATION", "").strip().lower() != "true":
        return
    try:
        import diarizer
        cache_dir = os.environ.get("HF_HOME", "/var/lib/whisper")
        num_speakers = _env_int("WHISPER_DIARIZE_NUM_SPEAKERS", -1)
        threshold = _env_float("WHISPER_DIARIZE_THRESHOLD", 0.5)
        # Modell der Stimm-Merkmale: "eres2net" (Standard, wie bisher) oder "campplus" (in einem Test etwa doppelt so schnell).
        embedding = os.environ.get("WHISPER_DIARIZE_EMBEDDING", "").strip().lower() or "eres2net"
        # Threads der Sprecher-Erkennung: ohne Angabe alle Kerne des Containers, höchstens 8 (sherpa-onnx würde sonst nur EINEN nutzen).
        threads = _env_int("WHISPER_DIARIZE_THREADS", 0)
        logger.info("Loading diarization pipeline...")
        diarizer.load(
            cache_dir=cache_dir,
            num_speakers=num_speakers,
            cluster_threshold=threshold,
            embedding=embedding,
            num_threads=threads if threads > 0 else None,
        )
        _diarization_enabled = True
        logger.info("Diarization enabled.")
    except Exception as exc:
        logger.warning("Failed to load diarizer (diarization disabled): %s", exc)
        _diarization_enabled = False


def _load_model() -> None:
    """Import and initialise the faster-whisper model from environment config."""
    global _model, _model_name, _beam_size, _max_request_beam, _word_timestamps, _max_upload_bytes, _max_duration_min

    from faster_whisper import WhisperModel  # deferred — keeps import fast

    model_name       = os.environ.get("WHISPER_MODEL",        "base").strip()
    device           = os.environ.get("WHISPER_DEVICE",       "cpu").strip()
    compute_type     = os.environ.get("WHISPER_COMPUTE_TYPE", "int8").strip()
    threads          = _env_int("WHISPER_THREADS", 2)
    cache_dir        = os.environ.get("HF_HOME", "/var/lib/whisper")
    local_files_only = bool(os.environ.get("WHISPER_LOCAL_ONLY", "").strip())
    _beam_size       = _env_int("WHISPER_BEAM", 5)
    _max_request_beam = _env_int("WHISPER_MAX_REQUEST_BEAM", 10)
    _word_timestamps = os.environ.get("WHISPER_WORD_TIMESTAMPS", "").strip().lower() == "true"
    max_upload_mb    = _env_int("WHISPER_MAX_UPLOAD_MB", 1024)
    if _max_request_beam < 0:
        logger.error(
            "Invalid value for WHISPER_MAX_REQUEST_BEAM: %d (expected 0 or greater); using default 10",
            _max_request_beam,
        )
        _max_request_beam = 10
    if max_upload_mb < 0:
        logger.error(
            "Invalid value for WHISPER_MAX_UPLOAD_MB: %d (expected 0 or greater); using default 1024",
            max_upload_mb,
        )
        max_upload_mb = 1024
    _max_upload_bytes = max_upload_mb * 1024 * 1024
    max_duration_min = _env_int("WHISPER_MAX_DURATION_MIN", 240)
    if max_duration_min < 0:
        logger.warning("Invalid value for WHISPER_MAX_DURATION_MIN: %d (expected 0 or greater); using default 240", max_duration_min)
        max_duration_min = 240
    _max_duration_min = max_duration_min
    logger.info("Limits | max_upload_mb=%d max_duration_min=%d docs=%s", max_upload_mb, _max_duration_min, _docs_enabled)
    gateway_cfg = _gateway_config()
    if gateway_cfg:
        logger.info("Backend | gateway host=%s model=%s chunk=%.0fs parallel=%d fallback=%s vad=%s",
                    gateway_cfg["host"], gateway_cfg["model"], gateway_cfg["chunk_s"], gateway_cfg["parallel"],
                    gateway_cfg["fallback"], gateway_cfg["vad"])
    else:
        logger.info("Backend | local")

    logger.info(
        "Loading model '%s' | device=%s compute_type=%s threads=%d beam=%d max_request_beam=%d word_ts=%s max_upload_mb=%d local_only=%s cache=%s",
        model_name, device, compute_type, threads, _beam_size, _max_request_beam, _word_timestamps, max_upload_mb, local_files_only, cache_dir,
    )
    logger.info(
        "Decoding | temperature_fallback=%s condition_on_previous_text=%s compression_ratio_threshold=2.4",
        list(_TEMPERATURE_FALLBACK), _condition_on_previous,
    )
    t0 = time.monotonic()
    _model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=threads,
        download_root=cache_dir,
        local_files_only=local_files_only,
    )
    _model_name = model_name
    logger.info("Model '%s' ready in %.1fs", model_name, time.monotonic() - t0)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _load_model()
    _load_diarizer()
    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

# Die interaktive API-Dokumentation (/docs, /redoc, /openapi.json) ist ohne Schlüssel abrufbar und verrät die ganze Schnittstelle.
# Standardmäßig aus; WHISPER_ENABLE_DOCS=true schaltet sie für die Entwicklung ein.
_docs_enabled = os.environ.get("WHISPER_ENABLE_DOCS", "").strip().lower() in ("1", "true", "yes", "on")

app = FastAPI(
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
    title="Whisper Speech-to-Text",
    description=(
        "OpenAI-compatible speech-to-text API powered by faster-whisper.\n\n"
        "https://github.com/hwdsl2/docker-whisper"
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


@app.middleware("http")
async def _log_request_entry(request, call_next):
    """
    Diagnose-Hilfe: loggt den frühestmöglichen Zeitpunkt, zu dem eine Anfrage
    unser ASGI-App überhaupt erreicht — NOCH VOR jeglichem Parsing von
    Formular-/Datei-Daten durch FastAPI/Starlette. Schlägt eine Anfrage schon
    beim Hochladen großer Dateien fehl, zeigt dieser Log-Eintrag, ob der
    Fehler VOR oder NACH diesem Punkt passiert (z. B. in der
    Multipart-Verarbeitung selbst, oder noch davor auf Infrastrukturebene).
    """
    logger.info(
        "Anfrage eingegangen: %s %s (Content-Length: %s)",
        request.method, request.url.path, request.headers.get("content-length", "unbekannt"),
    )
    try:
        response = await call_next(request)
        logger.info("Anfrage beendet: %s %s -> Status %s", request.method, request.url.path, response.status_code)
        # Antworten werden nie zwischengespeichert oder anders gedeutet (Zwischenspeicher-Header des Streams bleiben unberührt)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response
    except Exception:
        logger.exception("Anfrage abgebrochen mit Ausnahme: %s %s", request.method, request.url.path)
        raise

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    """
    If WHISPER_API_KEY is set, require a matching Bearer token.
    If the env var is empty or unset the endpoint is open (no auth).
    """
    required = os.environ.get("WHISPER_API_KEY", "").strip()
    if not required:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    parts = authorization.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header. Expected: Bearer <key>",
        )
    # Vergleich in konstanter Zeit: die Antwortzeit verrät nichts darüber, wie viele Zeichen des Schlüssels stimmen.
    if not hmac.compare_digest(parts[1].encode("utf-8"), required.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _safe_suffix(name: Optional[str]) -> str:
    """Endung der temporären Datei aus dem vom Browser gelieferten Namen: nur kurze, einfache Endungen.
    Pfade, Sonderzeichen und überlange Endungen werden nie übernommen (dann: ".audio")."""
    ext = os.path.splitext(os.path.basename(name or ""))[1]
    return ext.lower() if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", ext or "") else ".audio"


def _merge_form_lists(*values: Optional[List[str]]) -> Optional[List[str]]:
    merged = []
    for value in values:
        if not value:
            continue
        merged.extend(item for item in value if item is not None)
    return merged or None


def _validate_openai_transcription_compat(
    model: str,
    response_format: str,
    include: Optional[List[str]] = None,
    chunking_strategy: Optional[str] = None,
    known_speaker_names: Optional[List[str]] = None,
    known_speaker_references: Optional[List[str]] = None,
) -> None:
    model_id = model.strip().lower()
    if model_id == _UNSUPPORTED_DIARIZE_MODEL:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model 'gpt-4o-transcribe-diarize' is not supported. "
                "Local diarization is available via WHISPER_DIARIZATION=true "
                "and verbose_json output."
            ),
        )
    if response_format.strip().lower() == "diarized_json":
        raise HTTPException(
            status_code=400,
            detail=(
                "response_format='diarized_json' is not supported. Local diarization, "
                "when enabled, annotates verbose_json segments with speaker labels."
            ),
        )
    if include:
        raise HTTPException(
            status_code=400,
            detail="The include field is not supported; OpenAI-compatible logprobs are not available from faster-whisper.",
        )
    if chunking_strategy is not None and chunking_strategy.strip():
        raise HTTPException(
            status_code=400,
            detail="chunking_strategy is not supported by this server.",
        )
    if known_speaker_names:
        raise HTTPException(
            status_code=400,
            detail="known_speaker_names is not supported; local diarization is unsupervised.",
        )
    if known_speaker_references:
        raise HTTPException(
            status_code=400,
            detail="known_speaker_references is not supported; local diarization does not accept speaker reference audio.",
        )


def _decode_options(temperature: float) -> dict:
    """Einstellungen der Whisper-Entschlüsselung, die Wiederholungsschleifen vorbeugen.

    Die Anfrage schickt standardmäßig temperature=0. Ein einzelner Wert schaltet die Ersatzversuche von faster-whisper ab, und eine
    Schleife bleibt dann bis zum Ende der Datei bestehen. Bei 0 gilt deshalb die Standardfolge 0, 0.2 ... 1.0: ein Abschnitt, der zu
    stark wiederholt (compression_ratio über 2,4) oder zu unsicher ist, wird mit höherer Temperatur neu berechnet. Ein ausdrücklich
    angegebener Wert über 0 wird unverändert verwendet."""
    return {
        "temperature": list(_TEMPERATURE_FALLBACK) if temperature == 0 else temperature,
        "condition_on_previous_text": _condition_on_previous,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }


def _probe_duration(path: str, cap_seconds: float):
    """Dauer der Audiodatei in Sekunden, ohne sie ganz zu dekodieren: aus dem Dateikopf. Fehlt die Angabe, wird bis höchstens
    cap_seconds mitgezählt (dann genügt "länger als erlaubt"). None, wenn die Datei nicht lesbar ist (der normale Weg meldet dann
    seinen eigenen Fehler)."""
    try:
        import av
        with av.open(path, metadata_errors="ignore") as container:
            if container.duration:  # Mikrosekunden
                return container.duration / 1_000_000
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return None
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            counted = 0.0
            for frame in container.decode(stream):
                counted += frame.samples / frame.sample_rate
                if counted > cap_seconds:
                    break
            return counted
    except Exception:
        return None


def _check_duration_limit(tmp_path: str) -> None:
    """Löscht die Datei und lehnt mit 413 ab, wenn sie länger als WHISPER_MAX_DURATION_MIN ist."""
    if not _max_duration_min:
        return
    limit_s = _max_duration_min * 60
    duration = _probe_duration(tmp_path, limit_s)
    if duration is not None and duration > limit_s:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=413,
            detail=f"Audio is too long ({duration / 60:.0f} min). Maximum allowed duration is {_max_duration_min} minutes.",
        )


def _validate_temperature(temperature: float) -> None:
    if not 0 <= temperature <= 1:
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 1.",
        )


def _resolve_request_beam(beam: Optional[str]) -> int:
    """Resolve the optional local beam override for a single request."""
    if beam is None:
        return _beam_size

    value = beam.strip()
    if not value or not value.isascii() or not value.isdecimal() or value.startswith("0"):
        raise HTTPException(status_code=400, detail="beam must be a positive integer.")

    request_beam = int(value)
    if _max_request_beam and request_beam > _max_request_beam:
        raise HTTPException(
            status_code=400,
            detail=(
                f"beam must be less than or equal to {_max_request_beam}. "
                "Set WHISPER_MAX_REQUEST_BEAM=0 to disable this per-request cap."
            ),
        )

    return request_beam


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _fmt_ts(seconds: float, fmt: str) -> str:
    """Format a float second offset as an SRT or VTT timestamp string."""
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    sep = "," if fmt == "srt" else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _to_srt(segments, speaker_map=None) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip() if hasattr(seg, "text") else seg.get("text", "").strip()
        start = seg.start if hasattr(seg, "start") else seg["start"]
        end = seg.end if hasattr(seg, "end") else seg["end"]
        speaker = speaker_map.get(i - 1) if speaker_map else None
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(
            f"{i}\n"
            f"{_fmt_ts(start, 'srt')} --> {_fmt_ts(end, 'srt')}\n"
            f"{prefix}{text}\n"
        )
    return "\n".join(lines)


def _to_vtt(segments, speaker_map=None) -> str:
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(segments):
        text = seg.text.strip() if hasattr(seg, "text") else seg.get("text", "").strip()
        start = seg.start if hasattr(seg, "start") else seg["start"]
        end = seg.end if hasattr(seg, "end") else seg["end"]
        speaker = speaker_map.get(i) if speaker_map else None
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(
            f"{_fmt_ts(start, 'vtt')} --> {_fmt_ts(end, 'vtt')}\n"
            f"{prefix}{text}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------


async def _stream_sse(
    tmp_path: str,
    lang: Optional[str],
    prompt: Optional[str],
    temperature: float,
    beam_size: int,
    task: str = "transcribe",
    *,
    job_id: Optional[str] = None,
    diarize_on: bool = False,
    num_speakers: Optional[int] = None,
    wt_flag: bool = False,
):
    """
    Async generator that yields Server-Sent Events (SSE).

    Event types emitted:
      - transcript.progress    — {"duration": <Sekunden>} direkt nach dem Dekodieren (Gesamtlänge)
      - transcript.text.delta  — {"delta", "start", "end"} je Segment, sobald es fertig ist
                                 (OpenAI-Ereignis, start/end sind eine lokale Erweiterung)
      - transcript.phase       — {"phase": "diarize"}: die Sprecher werden jetzt zugeordnet
      - transcript.result      — {"result": {...}}: das vollständige Ergebnis im Format von
                                 response_format=verbose_json, inklusive Sprecher (lokale Erweiterung)
      - transcript.text.done   — final assembled transcript (OpenAI)
      - transcript.cancelled   — der Auftrag wurde über /v1/jobs/{id}/cancel abgebrochen

    Inference runs in a thread-pool worker so the event loop stays responsive
    while the CPU-bound model decodes the audio.  _inference_lock ensures that
    only one transcription (batch or streaming) runs at a time.

    Wird der Auftrag abgebrochen oder trennt der Client die Verbindung, bricht der Worker
    nach dem aktuellen Segment ab. Die temporäre Audiodatei wird am Ende immer gelöscht.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    cancel_event = _register_cancel(job_id)

    def put(kind, value=None) -> None:
        loop.call_soon_threadsafe(q.put_nowait, (kind, value))

    def _run() -> None:
        with _inference_lock:
            try:
                if cancel_event.is_set():
                    raise _Cancelled()
                transcribe_kwargs = dict(
                    language=lang,
                    task=task,
                    initial_prompt=prompt or None,
                    **_decode_options(temperature),
                    beam_size=beam_size,
                    # Wortzeiten braucht die Sprecher-Zuordnung intern (wie im normalen Ablauf).
                    word_timestamps=(wt_flag or diarize_on),
                    vad_filter=True,
                    vad_parameters={"max_speech_duration_s": 30},
                )
                run = _TranscribeRun(
                    tmp_path, transcribe_kwargs,
                    should_cancel=cancel_event.is_set,
                    on_duration=lambda total: put("progress", total),
                )
                segments = []
                for seg in run.iter_segments():
                    segments.append(seg)
                    put("segment", seg)

                speaker_map = refined_segments = None
                if diarize_on:
                    if cancel_event.is_set():
                        raise _Cancelled()
                    put("phase", "diarize")
                    last_p = {"v": -1.0}

                    def on_diar_progress(frac: float) -> bool:
                        """Meldet den Fortschritt der Sprecher-Zuordnung (höchstens alle 1 %) und stoppt sie, wenn abgebrochen wurde."""
                        if cancel_event.is_set():
                            return True
                        if frac - last_p["v"] >= 0.01 or frac >= 1.0:
                            last_p["v"] = frac
                            put("diar_progress", frac)
                        return False

                    speaker_map, refined_segments = _diarize_segments(
                        tmp_path, segments, num_speakers, progress=on_diar_progress
                    )
                    if cancel_event.is_set():
                        raise _Cancelled()
                result = _build_verbose_response(
                    task, run.info, segments, speaker_map, refined_segments, wt_flag, temperature
                )
                put("result", result)
            except _Cancelled:
                put("cancelled")
            except Exception as exc:  # noqa: BLE001
                put("error", exc)
            finally:
                put("end")

    loop.run_in_executor(None, _run)
    text_parts: list = []

    def frame(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    try:
        while True:
            kind, value = await q.get()
            if kind == "end":
                break
            if kind == "progress":
                yield frame({"type": "transcript.progress", "duration": round(value, 3)})
            elif kind == "phase":
                yield frame({"type": "transcript.phase", "phase": value})
            elif kind == "diar_progress":
                # Alte Frontends lesen nur "phase" und ignorieren "progress".
                yield frame({"type": "transcript.phase", "phase": "diarize", "progress": round(value, 3)})
            elif kind == "segment":
                seg_text = value.text.strip()
                if seg_text:
                    # Prepend space to separate from previous segments
                    delta = seg_text if not text_parts else " " + seg_text
                    text_parts.append(seg_text)
                    yield frame({
                        "type": "transcript.text.delta",
                        "delta": delta,
                        "start": round(value.start, 3),
                        "end": round(value.end, 3),
                    })
            elif kind == "result":
                yield frame({"type": "transcript.result", "result": value})
            elif kind == "cancelled":
                logger.info("Streaming-Auftrag abgebrochen (%s)", job_id or "ohne Nummer")
                yield frame({"type": "transcript.cancelled"})
                return
            elif kind == "error":
                logger.error("Streaming transcription error: %s", value)
                yield frame({"error": {"type": "transcription_error", "message": str(value)}})
                return
        # Final frame — complete transcript (OpenAI transcript.text.done event)
        full_text = " ".join(text_parts).strip()
        yield frame({"type": "transcript.text.done", "text": full_text})
        yield "data: [DONE]\n\n"
    finally:
        # Client weg oder fertig: den Worker nicht unnötig weiterlaufen lassen.
        cancel_event.set()
        _unregister_cancel(job_id, cancel_event)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Speicherschonende Verarbeitung langer Dateien
# ---------------------------------------------------------------------------
#
# HINTERGRUND (gemessen, nicht vermutet):
# faster-whisper lädt die gesamte Audiodatei als ein numpy-Array in den
# Speicher (decode_audio, über PyAV) und führt die Sprachaktivitäts-Erkennung
# (VAD) anschließend auf dem KOMPLETTEN Array auf einmal aus. Für eine reale
# 73-minütige Datei wurde Folgendes gemessen:
#   - decode_audio allein:           ~270 MB Rohdaten, Spitzenwert ~740 MB
#   - VAD auf dem GANZEN Array:      Spitzenwert steigt auf ~1130 MB
#   - VAD in 10-Minuten-Scheiben:    Spitzenwert bleibt bei ~740 MB
# Das erklärt den OOM-Absturz ("Killed") auf dem Coolify-Server bei langen
# Dateien: Das Hauptmodell (für "medium" ca. 1-1.5 GB) ist bereits geladen,
# und der zusätzliche VAD-Spitzenwert von über einem weiteren GB bringt den
# Gesamtverbrauch über das verfügbare RAM des geteilten Servers.
#
# LÖSUNG: Die Datei wird nur EINMAL dekodiert (unvermeidbar mit PyAV), aber
# anschließend in 10-Minuten-Abschnitte des bereits geladenen Arrays
# zerlegt. transcribe() akzeptiert laut Bibliotheks-Signatur direkt ein
# numpy-Array (kein Umweg über die Festplatte nötig). Jeder Abschnitt wird
# einzeln durch das Modell geschickt, die Zeitstempel werden um den Start
# des jeweiligen Abschnitts verschoben, und die Ergebnisse werden am Ende
# zu einem einzigen Resultat zusammengeführt.
_CHUNK_THRESHOLD_SECONDS = _env_int("WHISPER_CHUNK_THRESHOLD_SECONDS", 900)  # 15 min
_CHUNK_SIZE_SECONDS = _env_int("WHISPER_CHUNK_SIZE_SECONDS", 600)  # 10 min


# ---------------------------------------------------------------------------
# Optional: Spracherkennung über ein Gateway (LiteLLM/vLLM, OpenAI-kompatibel) statt auf diesem Rechner.
# WHISPER_BACKEND=gateway schaltet es ein. Die Aufnahme wird in Stücke von höchstens 28 Sekunden geschnitten (ein
# Whisper-Fenster, dadurch stimmen die Zeitstempel und jede Anfrage bleibt klein), die Stücke laufen mehrere gleichzeitig.
# Bei einem Fehler läuft der Rest auf dem lokalen Modell (GATEWAY_FALLBACK=false schaltet das ab).
# Der Schlüssel steht nur in der Umgebung dieses Prozesses; er wird nie protokolliert und nie an einen anderen Rechner
# weitergegeben (Weiterleitungen werden abgelehnt).
# ---------------------------------------------------------------------------
_GW_RATE = 16000
_GW_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_GW_MAX_RESPONSE = 8 * 1024 * 1024
_gw_warned: set = set()
_gw_vad_lock = threading.Lock()  # die Erkennung wird nie von zwei Threads gleichzeitig benutzt (Vorsicht bei anderen Bibliotheksversionen)


class _GatewayError(Exception):
    """Fehler beim Gateway. Die Meldung ist bereinigt und enthält nie den Schlüssel."""


def _gw_warn_once(key: str, message: str, *args) -> None:
    if key not in _gw_warned:
        _gw_warned.add(key)
        logger.warning(message, *args)


def _gw_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return default if not val else val in ("1", "true", "yes", "on")


def _gateway_config():
    """Liest die Einstellungen bei jedem Aufruf. None heißt: lokaler Weg (nicht eingeschaltet oder unvollständig/unsicher)."""
    if os.environ.get("WHISPER_BACKEND", "local").strip().lower() != "gateway":
        return None
    url = os.environ.get("GATEWAY_URL", "").strip().rstrip("/")
    key = os.environ.get("GATEWAY_API_KEY", "").strip()
    if not url or not key:
        _gw_warn_once("missing", "WHISPER_BACKEND=gateway, but GATEWAY_URL or GATEWAY_API_KEY is missing; using the local model")
        return None
    parsed = urllib.parse.urlparse(url)
    local_host = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if not parsed.hostname or not (parsed.scheme == "https" or (parsed.scheme == "http" and (local_host or _gw_flag("GATEWAY_ALLOW_HTTP", False)))):
        _gw_warn_once("scheme", "GATEWAY_URL must be an https address (the audio would travel unencrypted); using the local model")
        return None
    if "/audio/transcriptions" in parsed.path:
        endpoint = url
    elif parsed.path.endswith("/v1"):
        endpoint = url + "/audio/transcriptions"
    else:
        endpoint = url + "/v1/audio/transcriptions"
    return {
        "endpoint": endpoint,
        "host": parsed.hostname,
        "key": key,
        "model": os.environ.get("GATEWAY_MODEL", "whisper-large-v3-turbo").strip() or "whisper-large-v3-turbo",
        "chunk_s": min(max(_env_float("GATEWAY_CHUNK_SECONDS", 28.0), 10.0), 29.0),
        "parallel": min(max(_env_int("GATEWAY_PARALLEL", 3), 1), 8),
        "timeout": max(_env_float("GATEWAY_TIMEOUT_S", 300.0), 5.0),
        "retries": min(max(_env_int("GATEWAY_RETRIES", 2), 0), 5),
        "backoff": max(_env_float("GATEWAY_RETRY_BACKOFF", 1.5), 0.0),
        "fallback": _gw_flag("GATEWAY_FALLBACK", True),
        "word_ts": _gw_flag("GATEWAY_WORD_TIMESTAMPS", False),
        "vad": _gw_flag("GATEWAY_VAD", True),
        "language": (os.environ.get("GATEWAY_LANGUAGE", "").strip().lower() or None),
    }


def _gw_clean(text, key: str) -> str:
    text = " ".join(str(text).split())
    if key:
        text = text.replace(key, "***")
    return text[:200]


def _gw_wav(samples) -> bytes:
    import numpy as np
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_GW_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _gw_multipart(fields, filename: str, data: bytes):
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{n}"\r\n\r\n{v}\r\n'.encode("utf-8") for n, v in fields]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode("utf-8")
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Weiterleitungen werden nicht befolgt: sonst ginge der Schlüssel an einen fremden Rechner."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _gateway_call(cfg: dict, chunk, language, prompt, want_words: bool) -> dict:
    """Schickt ein Stück (float32, 16 kHz) ans Gateway und gibt die Antwort (verbose_json) zurück."""
    fields = [("model", cfg["model"]), ("response_format", "verbose_json")]
    if language:
        fields.append(("language", language))
    if prompt:
        fields.append(("prompt", str(prompt)))
    if want_words:
        fields += [("timestamp_granularities[]", "segment"), ("timestamp_granularities[]", "word")]
    body, ctype = _gw_multipart(fields, "audio.wav", _gw_wav(chunk))
    opener = urllib.request.build_opener(_NoRedirect)
    key = cfg["key"]
    for attempt in range(cfg["retries"] + 1):
        retry = False
        try:
            req = urllib.request.Request(
                cfg["endpoint"], data=body, method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": ctype, "Accept": "application/json",
                         "User-Agent": "docker-whisper-engine"},
            )
            with opener.open(req, timeout=cfg["timeout"]) as resp:
                raw = resp.read(_GW_MAX_RESPONSE + 1)
            if len(raw) > _GW_MAX_RESPONSE:
                raise _GatewayError("response too large")
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not an object")
            return data
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(2000).decode("utf-8", "replace")
            except Exception:
                detail = ""
            error = _GatewayError(f"HTTP {exc.code}: {_gw_clean(detail, key)}" if detail else f"HTTP {exc.code}")
            retry = exc.code in _GW_RETRY_STATUS
        except _GatewayError:
            raise
        except (ValueError, UnicodeDecodeError):
            error, retry = _GatewayError("invalid response (not JSON)"), False
        except (urllib.error.URLError, socket.timeout, TimeoutError, ssl.SSLError, OSError) as exc:
            error, retry = _GatewayError(f"connection error: {_gw_clean(getattr(exc, 'reason', exc), key)}"), True
        if not retry or attempt >= cfg["retries"]:
            raise error
        time.sleep(cfg["backoff"] * (2 ** attempt))
    raise _GatewayError("unreachable")  # pragma: no cover


def _gw_spans(audio, max_seconds: float):
    """Schnittstellen (Anfang, Ende in Samples): jedes Stück höchstens max_seconds lang, geschnitten wird an der leisesten
    Stelle in den letzten 8 Sekunden, damit kein Wort zerteilt wird."""
    import numpy as np
    n = len(audio)
    max_n, search_n, frame = int(max_seconds * _GW_RATE), int(8 * _GW_RATE), int(0.1 * _GW_RATE)
    spans, pos = [], 0
    while n - pos > max_n:
        lo = pos + max_n - search_n
        window = audio[lo:pos + max_n]
        k = len(window) // frame
        energy = (np.asarray(window[:k * frame], dtype="float32").reshape(k, frame) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * frame + frame // 2
        spans.append((pos, cut))
        pos = cut
    spans.append((pos, n))
    return spans


def _gw_speech_spans(chunk):
    """Sprechende Abschnitte (Sekunden) im Stück mit der Sprachaktivitäts-Erkennung der Bibliothek; None, wenn sie ausfällt."""
    try:
        from faster_whisper.vad import get_speech_timestamps
        with _gw_vad_lock:
            found = get_speech_timestamps(chunk)
        return [(t["start"] / _GW_RATE, t["end"] / _GW_RATE) for t in found]
    except Exception as exc:
        _gw_warn_once("vad", "Speech detection for the gateway path failed (%s); sending every section", type(exc).__name__)
        return None


def _gw_segments(data: dict, offset: float, chunk_seconds: float, want_words: bool) -> list:
    """Wandelt die Antwort in Segmente um, wie faster-whisper sie liefert (Zeitstempel um offset verschoben)."""
    from faster_whisper.transcribe import Segment, Word
    seg_fields = {f.name for f in dataclasses.fields(Segment)}
    raw = data.get("segments")
    if not isinstance(raw, list) or not raw:
        text = str(data.get("text") or "").strip()
        raw = [{"start": 0.0, "end": chunk_seconds, "text": text}] if text else []
    words_raw = data.get("words") if want_words and isinstance(data.get("words"), list) else []
    out = []
    try:
        for d in raw:
            text = str(d.get("text") or "").strip()
            if not text:
                continue
            start = min(max(float(d.get("start") or 0.0), 0.0), chunk_seconds)
            end = min(max(float(d.get("end") or start), start), chunk_seconds)
            words = None
            if words_raw:
                words = [
                    Word(start=float(w["start"]) + offset, end=float(w["end"]) + offset, word=" " + str(w["word"]).strip(),
                         probability=float(w.get("probability") or 0.0))
                    for w in words_raw if isinstance(w, dict) and start - 0.05 <= float(w["start"]) <= end + 0.05
                ] or None
            values = dict(
                id=0, seek=0, start=start + offset, end=end + offset, text=" " + text, tokens=list(d.get("tokens") or []),
                avg_logprob=float(d.get("avg_logprob") or 0.0), compression_ratio=float(d.get("compression_ratio") or 0.0),
                no_speech_prob=float(d.get("no_speech_prob") or 0.0), words=words, temperature=float(d.get("temperature") or 0.0),
            )
            out.append(Segment(**{k: v for k, v in values.items() if k in seg_fields}))
    except (TypeError, ValueError, KeyError, AttributeError):
        raise _GatewayError("invalid segment data in the response") from None
    return out


def _gw_filter_by_speech(segments: list, speech: list, offset: float) -> list:
    """Verwirft Segmente, die fast nur in Stille liegen (Whisper erfindet dort gern Text wie "Vielen Dank fürs Zuschauen")."""
    kept = []
    for seg in segments:
        a, b = seg.start - offset, seg.end - offset
        length = max(b - a, 1e-6)
        overlap = sum(max(0.0, min(b, e) - max(a, s)) for s, e in speech)
        if overlap / length >= 0.15:
            kept.append(seg)
    return kept


def _shift_segment(seg, offset: float):
    """Verschiebt Segment und Wörter um offset Sekunden (Segment/Word sind Dataclasses)."""
    words = seg.words
    if words:
        words = [dataclasses.replace(w, start=w.start + offset, end=w.end + offset) for w in words]
    return dataclasses.replace(seg, start=seg.start + offset, end=seg.end + offset, words=words if words else seg.words)


class _ChunkedInfo:
    """Leichtgewichtiger Ersatz für das von transcribe() zurückgegebene
    TranscriptionInfo-Objekt, wenn mehrere Chunks zusammengeführt werden.
    Sprache/Wahrscheinlichkeit stammen vom ersten Chunk (dort meist am
    zuverlässigsten erkennbar); Dauer-Werte werden über alle Chunks summiert."""

    def __init__(self, language, language_probability, duration, duration_after_vad):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration
        self.duration_after_vad = duration_after_vad


class _TranscribeRun:
    """
    Dekodiert die Datei EINMAL und liefert die Segmente der Reihe nach (lazy). Lange Dateien
    werden dafür in Abschnitte zerlegt (Begründung siehe Kommentar oben, gemessener Speicherbedarf).
    Normaler Ablauf und Streaming benutzen denselben Kern, damit sie sich nicht auseinanderentwickeln.

    should_cancel: wird nach jedem Segment und vor jedem Abschnitt geprüft; True -> _Cancelled.
    on_duration:   wird direkt nach dem Dekodieren mit der Gesamtlänge (Sekunden) aufgerufen.
    Nach dem Durchlaufen enthält .info die Angaben zu Sprache und Dauer.
    """

    def __init__(self, tmp_path, transcribe_kwargs, should_cancel=None, on_duration=None):
        self.tmp_path = tmp_path
        self.kwargs = transcribe_kwargs
        self.should_cancel = should_cancel or (lambda: False)
        self.on_duration = on_duration
        self.total_seconds = None
        self.info = None

    def _check(self):
        if self.should_cancel():
            raise _Cancelled()

    def iter_segments(self):
        from faster_whisper.audio import decode_audio

        sampling_rate = 16000
        self._check()
        audio = decode_audio(self.tmp_path, sampling_rate=sampling_rate)
        self.total_seconds = len(audio) / sampling_rate
        if _max_duration_min and self.total_seconds > _max_duration_min * 60 * 1.02:
            # Zweite Sicherung: der Dateikopf kann falsche Angaben machen
            raise ValueError(f"Audio is too long ({self.total_seconds / 60:.0f} min). Maximum allowed duration is {_max_duration_min} minutes.")
        if self.on_duration:
            self.on_duration(self.total_seconds)

        gateway_cfg = _gateway_config()
        if gateway_cfg is not None:
            yield from self._iter_gateway(audio, sampling_rate, gateway_cfg)
            return

        if self.total_seconds <= _CHUNK_THRESHOLD_SECONDS:
            segments_gen, info = _model.transcribe(audio, **self.kwargs)
            self.info = info
            for seg in segments_gen:
                self._check()
                yield seg
            return

        logger.info(
            "Lange Datei (%.1f min) wird in %d-Minuten-Abschnitten verarbeitet, "
            "um den gemessenen VAD-Speicherspitzenwert zu vermeiden.",
            self.total_seconds / 60, _CHUNK_SIZE_SECONDS // 60,
        )
        chunk_samples = _CHUNK_SIZE_SECONDS * sampling_rate
        first_info = None
        summed_duration = 0.0
        summed_duration_after_vad = 0.0

        for start_sample in range(0, len(audio), chunk_samples):
            self._check()
            chunk_offset_seconds = start_sample / sampling_rate
            chunk_array = audio[start_sample:start_sample + chunk_samples]

            segs_gen, info = _model.transcribe(chunk_array, **self.kwargs)
            for seg in segs_gen:
                self._check()
                # Zeitstempel um den Start dieses Abschnitts verschieben, damit sie sich weiterhin
                # auf die gesamte Originaldatei beziehen. Segment/Word sind dataclasses.
                shifted_words = None
                if seg.words:
                    shifted_words = [
                        dataclasses.replace(
                            w, start=w.start + chunk_offset_seconds, end=w.end + chunk_offset_seconds
                        )
                        for w in seg.words
                    ]
                yield dataclasses.replace(
                    seg,
                    start=seg.start + chunk_offset_seconds,
                    end=seg.end + chunk_offset_seconds,
                    words=shifted_words if shifted_words is not None else seg.words,
                )

            if first_info is None:
                first_info = info
            summed_duration += info.duration
            summed_duration_after_vad += info.duration_after_vad

            # Audio-Array dieses Abschnitts freigeben, bevor der nächste beginnt.
            del chunk_array
            gc.collect()

        self.info = _ChunkedInfo(
            language=first_info.language if first_info else None,
            language_probability=first_info.language_probability if first_info else 0.0,
            duration=summed_duration,
            duration_after_vad=summed_duration_after_vad,
        )

    def _local_chunk(self, chunk, offset):
        if _model is None:
            raise RuntimeError("local model is not loaded")
        segs_gen, _info = _model.transcribe(chunk, **self.kwargs)
        for seg in segs_gen:
            self._check()
            yield _shift_segment(seg, offset)

    def _iter_gateway(self, audio, sampling_rate, cfg):
        spans = _gw_spans(audio, cfg["chunk_s"])
        n = len(spans)
        prompt = self.kwargs.get("initial_prompt")
        want_words = bool(cfg["word_ts"] and self.kwargs.get("word_timestamps"))
        lang = [cfg["language"] or self.kwargs.get("language")]  # None: wird aus den ersten Stücken bestimmt
        votes = Counter()
        stop = threading.Event()  # nach einem Gateway-Fehler gesetzt: der Rest läuft lokal
        stats = {"gateway": 0, "silent": 0, "local": 0}
        speech_seconds = 0.0
        next_id = 0
        logger.info("Gateway | %.1f min in %d Abschnitten (höchstens %.0f s), %d gleichzeitig, Modell %s", 
                    len(audio) / sampling_rate / 60, n, cfg["chunk_s"], cfg["parallel"], cfg["model"])

        def work(i):
            if stop.is_set():
                return "local", None
            a, b = spans[i]
            chunk = audio[a:b]
            offset = a / sampling_rate
            speech = _gw_speech_spans(chunk) if cfg["vad"] else None
            if speech is not None and not speech:
                return "silent", None
            try:
                data = _gateway_call(cfg, chunk, lang[0], prompt, want_words)
                segs = _gw_segments(data, offset, len(chunk) / sampling_rate, want_words)
            except _GatewayError as exc:
                return "error", exc
            if speech:
                segs = _gw_filter_by_speech(segs, speech, offset)
            return "ok", (segs, data.get("language"), speech, len(chunk) / sampling_rate)

        pool = ThreadPoolExecutor(max_workers=cfg["parallel"])
        futures = {}
        try:
            nxt = 0
            for i in range(n):
                self._check()
                sequential = lang[0] is None and sum(votes.values()) < 3
                limit = 1 if sequential else cfg["parallel"] * 2
                while nxt < n and len(futures) < limit and (not sequential or nxt == i):
                    futures[nxt] = pool.submit(work, nxt)
                    nxt += 1
                fut = futures.pop(i)
                while True:
                    try:
                        status, payload = fut.result(timeout=0.5)
                        break
                    except FutureTimeout:
                        self._check()
                a, b = spans[i]
                offset = a / sampling_rate
                if status == "ok":
                    segs, detected, speech, seconds = payload
                    stats["gateway"] += 1
                    speech_seconds += sum(e - s for s, e in speech) if speech else seconds
                    if lang[0] is None and isinstance(detected, str) and re.fullmatch(r"[a-z]{2,3}", detected.strip().lower()):
                        votes[detected.strip().lower()] += 1
                        if sum(votes.values()) >= 3:
                            lang[0] = votes.most_common(1)[0][0]
                    for seg in segs:
                        self._check()
                        yield dataclasses.replace(seg, id=next_id)
                        next_id += 1
                elif status == "silent":
                    stats["silent"] += 1
                else:  # "error" oder "local": dieses Stück läuft auf dem lokalen Modell
                    if status == "error":
                        if not cfg["fallback"]:
                            raise payload
                        if not stop.is_set():  # nur der erste Fehler wird gemeldet (andere Stücke können gleichzeitig scheitern)
                            logger.warning("Gateway failed (%s); the remaining sections run on the local model", payload)
                            stop.set()
                    stats["local"] += 1
                    speech_seconds += (b - a) / sampling_rate
                    for seg in self._local_chunk(audio[a:b], offset):
                        yield dataclasses.replace(seg, id=next_id)
                        next_id += 1
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        logger.info("Gateway | fertig: %d Abschnitte über das Gateway, %d übersprungen (Stille), %d lokal",
                    stats["gateway"], stats["silent"], stats["local"])
        detected_language = lang[0] or (votes.most_common(1)[0][0] if votes else None)
        self.info = _ChunkedInfo(
            language=detected_language,
            language_probability=1.0 if lang[0] else (0.5 if detected_language else 0.0),
            duration=len(audio) / sampling_rate,
            duration_after_vad=speech_seconds,
        )


def _transcribe_long_audio(tmp_path: str, transcribe_kwargs: dict):
    """
    Normaler (nicht streamender) Ablauf: alle Segmente einsammeln.
    Gibt (segments: list, info: TranscriptionInfo | _ChunkedInfo) zurück.
    """
    run = _TranscribeRun(tmp_path, transcribe_kwargs)
    segments = list(run.iter_segments())
    return segments, run.info


def _diarize_segments(tmp_path: str, segments, num_speakers: Optional[int], progress=None):
    """
    Sprecher-Zuordnung (optionale Nachbearbeitung). Gibt (speaker_map, refined_segments) zurück:
      - refined_segments: wortgenaue Neu-Segmentierung (bevorzugt), sonst None
      - speaker_map: {segment_index: speaker_label} als grober Ersatz, sonst None
    Fällt die Zuordnung aus, kommt die Antwort ohne Sprecher zurück.
    """
    speaker_map = None
    refined_segments = None
    try:
        import diarizer
        if progress is not None and "progress" in inspect.signature(diarizer.diarize).parameters:
            turns = diarizer.diarize(tmp_path, num_speakers=num_speakers, progress=progress)
        else:  # ältere diarizer.py ohne Fortschritt
            turns = diarizer.diarize(tmp_path, num_speakers=num_speakers)

        # Preferred path: resegment at the word level so short
        # interjections (greetings, "yes"/"no", etc.) that would
        # otherwise be blurred into a longer neighbouring segment
        # get their own, correctly-attributed entry.
        refined_segments = diarizer.resegment_by_word(segments, turns)

        if not refined_segments:
            # Fallback: no word-level timing available for this
            # request — assign one speaker per whole Whisper segment.
            seg_dicts = [
                {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                for seg in segments
            ]
            diarizer.assign_speakers(seg_dicts, turns)
            speaker_map = {i: d["speaker"] for i, d in enumerate(seg_dicts)}
    except Exception as exc:
        if type(exc).__name__ == "DiarizationAborted":
            raise _Cancelled() from None  # auf Wunsch abgebrochen: kein Fehler, der Auftrag endet als abgebrochen
        logger.warning("Diarization failed (returning without speakers): %s", exc)
    return speaker_map, refined_segments


def _build_verbose_response(task, info, segments, speaker_map, refined_segments, wt_flag, temperature) -> dict:
    """Baut die Antwort im Format response_format=verbose_json (normaler Ablauf und Streaming-Ergebnis)."""
    full_text = " ".join(seg.text.strip() for seg in segments).strip()
    seg_list = []
    all_words = []

    if refined_segments is not None:
        # Word-accurate segments: one entry per contiguous same-speaker
        # run, rather than one per raw Whisper segment. These carry
        # start/end/text/speaker only (the richer per-segment stats
        # below don't apply to a resegmented run).
        for idx, seg in enumerate(refined_segments):
            seg_list.append({
                "id": idx,
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"],
                "speaker": seg["speaker"],
            })
        if wt_flag:
            for seg in segments:
                if seg.words:
                    all_words.extend(
                        {
                            "word": w.word.strip(),
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 4),
                        }
                        for w in seg.words
                    )
    else:
        for idx, seg in enumerate(segments):
            seg_dict = {
                "id": idx,
                "seek": seg.seek,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "tokens": seg.tokens,
                "temperature": round(seg.temperature, 3) if seg.temperature is not None else temperature,
                "avg_logprob": round(seg.avg_logprob, 4),
                "compression_ratio": round(seg.compression_ratio, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            }
            if speaker_map:
                seg_dict["speaker"] = speaker_map.get(idx, "SPEAKER_00")
            if wt_flag and seg.words:
                all_words.extend(
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                    for w in seg.words
                )
            seg_list.append(seg_dict)
    resp = {
        "task": task,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "duration_after_vad": round(info.duration_after_vad, 3),
        "text": full_text,
        "segments": seg_list,
    }
    if wt_flag:
        resp["words"] = all_words
    return resp


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", include_in_schema=False)
async def health():
    """Container liveness probe — used by run.sh to detect startup completion."""
    return {"status": "ok", "model": _model_name}


@app.get("/v1/models")
async def list_models(_auth: None = Depends(_verify_api_key)):
    """
    List the active model in OpenAI-compatible format.
    Any app that queries /v1/models before sending requests will work correctly.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name or "whisper-1",
                "object": "model",
                "created": 0,
                "owned_by": "faster-whisper",
            }
        ],
    }


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, _auth: None = Depends(_verify_api_key)):
    """
    Bricht einen laufenden Streaming-Auftrag ab (Nummer = Kopfzeile X-Job-Id der Anfrage).
    Der Auftrag stoppt nach dem aktuellen Segment. Antwort: {"cancelled": true|false}
    (false = es gibt keinen laufenden Auftrag mit dieser Nummer, z. B. weil er schon fertig ist).
    """
    clean = _clean_job_id(job_id)
    if clean is None:
        raise HTTPException(status_code=400, detail="Invalid job id.")
    with _cancel_lock:
        event = _cancel_events.get(clean)
    if event is None:
        return JSONResponse({"cancelled": False})
    event.set()
    logger.info("Abbruch angefordert für Auftrag %s", clean)
    return JSONResponse({"cancelled": True})


async def _handle_audio(
    task: str,
    file: UploadFile,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    temperature: float,
    stream: Optional[str],
    beam: Optional[str],
    timestamp_granularities: Optional[List[str]] = None,
    num_speakers: Optional[int] = None,
    diarize: Optional[bool] = None,
    job_id: Optional[str] = None,
):
    """
    Shared implementation for transcription and translation endpoints.
    ``task`` is either ``"transcribe"`` or ``"translate"``.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please retry.")

    _validate_temperature(temperature)
    request_beam = _resolve_request_beam(beam)

    # Block translation on English-only models
    if task == "translate" and _model_name and _model_name.endswith(".en"):
        raise HTTPException(
            status_code=400,
            detail=f"Translation is not supported with English-only model '{_model_name}'. "
                   "Use a multilingual model (e.g. base, small, large-v3-turbo).",
        )

    # Normalise the stream form field: the string "true" (case-insensitive) enables streaming.
    # Using Optional[str] instead of bool avoids Pydantic version-dependent coercion differences.
    stream_flag: bool = stream is not None and stream.strip().lower() == "true"

    # Resolve word_timestamps:
    #   timestamp_granularities (spec-compliant) > WHISPER_WORD_TIMESTAMPS env var > False
    wt_flag: bool
    if timestamp_granularities and "word" in timestamp_granularities:
        wt_flag = True
    else:
        wt_flag = _word_timestamps

    # Validate response_format (only relevant for non-streaming responses)
    valid_formats = {"json", "text", "verbose_json", "srt", "vtt"}
    if not stream_flag and response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            # Der vom Client gesendete Wert wird absichtlich NICHT in die
            # Fehlermeldung zurückgespiegelt (auch nicht in gekürzter Form) —
            # das verhindert, dass beliebiger, nicht vertrauenswürdiger Text
            # unverändert in einer HTTP-Antwort landet.
            detail=f"Invalid response_format. Must be one of: {', '.join(sorted(valid_formats))}",
        )

    # Resolve language: per-request param > WHISPER_LANGUAGE env var > None (autodetect)
    env_lang = os.environ.get("WHISPER_LANGUAGE", "auto").strip()
    if language and language.lower() != "auto":
        lang = language
    elif env_lang and env_lang.lower() != "auto":
        lang = env_lang
    else:
        lang = None  # faster-whisper autodetects when None

    # Persist uploaded bytes to a temp file so faster-whisper can open it.
    original_name = file.filename or "audio"
    suffix = _safe_suffix(original_name)
    tmp_path: Optional[str] = None
    upload_size = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                upload_size += len(chunk)
                if _max_upload_bytes and upload_size > _max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded audio is too large. "
                            f"Maximum allowed size is {_max_upload_bytes // (1024 * 1024)} MB."
                        ),
                    )
                tmp.write(chunk)
    except HTTPException:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    except Exception as exc:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        logger.exception("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    logger.info(
        "%s '%s' (%d bytes) | lang=%s format=%s stream=%s beam=%d word_ts=%s",
        "Translating" if task == "translate" else "Transcribing",
        original_name, upload_size, lang or "auto", response_format, stream_flag, request_beam, wt_flag,
    )

    _check_duration_limit(tmp_path)

    # ------------------------------------------------------------------
    # Streaming path — inference runs in a thread; temp file cleaned up
    # inside the generator when the stream ends or the client disconnects.
    # Auch hier werden Sprecher zugeordnet (am Ende, als transcript.result).
    # ------------------------------------------------------------------
    if stream_flag:
        return StreamingResponse(
            _stream_sse(
                tmp_path, lang, prompt, temperature, request_beam, task,
                job_id=job_id,
                diarize_on=(_diarization_enabled and diarize is not False),
                num_speakers=num_speakers,
                wt_flag=wt_flag,
            ),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",   # disable nginx proxy buffering
                "Cache-Control": "no-cache",
            },
        )

    # ------------------------------------------------------------------
    # Batch path
    # ------------------------------------------------------------------
    try:
        try:
            with _inference_lock:
                # Siehe _transcribe_long_audio: lange Dateien werden intern in
                # Abschnitten verarbeitet, um den gemessenen VAD-Speicher-
                # spitzenwert auf dem gesamten Array zu vermeiden (~1.1 GB bei
                # einer 73-minütigen Datei statt ~740 MB bei Abschnitten).
                segments, info = _transcribe_long_audio(
                    tmp_path,
                    dict(
                        language=lang,
                        task=task,
                        initial_prompt=prompt or None,
                        **_decode_options(temperature),
                        beam_size=request_beam,
                        # Word-level timing is needed internally for accurate
                        # per-speaker segment boundaries even if the caller
                        # did not request word timestamps in the response —
                        # a whole Whisper segment can otherwise span several
                        # real speaker turns (e.g. short greetings) and get
                        # blurred onto a single speaker.
                        word_timestamps=(wt_flag or (_diarization_enabled and diarize is not False)),
                        vad_filter=True,
                        # Verhindert, dass ein einzelner, sehr langer
                        # durchgehender Sprechabschnitt (keine natürliche
                        # Pause) für sich allein bereits zu einem
                        # Speicher-Absturz (OOM) führt.
                        vad_parameters={"max_speech_duration_s": 30},
                    ),
                )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Transcription failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

        # ------------------------------------------------------------------
        # Diarization (optional post-processing)
        # ------------------------------------------------------------------
        speaker_map = None  # {segment_index: speaker_label} — coarse fallback
        refined_segments = None  # word-accurate resegmentation, when available
        if _diarization_enabled and diarize is not False:
            speaker_map, refined_segments = _diarize_segments(tmp_path, segments, num_speakers)

    finally:
        # Clean up temp file after both transcription and diarization
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    full_text = " ".join(seg.text.strip() for seg in segments).strip()

    if response_format == "text":
        if speaker_map:
            lines = []
            prev_speaker = None
            for i, seg in enumerate(segments):
                spk = speaker_map.get(i, "SPEAKER_00")
                text = seg.text.strip()
                if spk != prev_speaker:
                    lines.append(f"[{spk}] {text}")
                    prev_speaker = spk
                else:
                    lines.append(text)
            return PlainTextResponse("\n".join(lines))
        return PlainTextResponse(full_text)

    if response_format == "srt":
        return PlainTextResponse(_to_srt(segments, speaker_map), media_type="text/plain")

    if response_format == "vtt":
        return PlainTextResponse(_to_vtt(segments, speaker_map), media_type="text/plain")

    if response_format == "verbose_json":
        return JSONResponse(
            _build_verbose_response(task, info, segments, speaker_map, refined_segments, wt_flag, temperature)
        )

    # Default: json — matches OpenAI's minimal response shape
    return JSONResponse({"text": full_text})


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(
        default="whisper-1",
        description="Model identifier (ignored — active model is used)",
    ),
    language: Optional[str] = Form(
        default=None,
        description="BCP-47 language code (e.g. 'en'). Omit or set to 'auto' for autodetect.",
    ),
    prompt: Optional[str] = Form(
        default=None,
        description="Optional text to guide the model's style or continue a previous segment.",
    ),
    response_format: str = Form(
        default="json",
        description="Output format: json, text, verbose_json, srt, vtt",
    ),
    temperature: float = Form(
        default=0.0,
        description="Sampling temperature between 0 and 1.",
    ),
    beam: Optional[str] = Form(
        default=None,
        description=(
            "Local faster-whisper extension: per-request beam size. "
            "Omit to use WHISPER_BEAM."
        ),
    ),
    stream: Optional[str] = Form(
        default=None,
        description=(
            "Stream segments as Server-Sent Events (text/event-stream). "
            "When true, the response is a series of 'data:' frames — one per "
            "decoded segment — followed by a final 'done' frame. "
            "response_format is ignored when stream=true."
        ),
    ),
    timestamp_granularities: Optional[List[str]] = Form(
        default=None,
        alias="timestamp_granularities[]",
        description=(
            "The timestamp granularities to populate for this transcription. "
            "response_format must be verbose_json. Supported values: 'word', 'segment'. "
            "Default: ['segment']."
        ),
    ),
    include: Optional[List[str]] = Form(
        default=None,
        description="Unsupported OpenAI response inclusions such as logprobs.",
    ),
    include_brackets: Optional[List[str]] = Form(default=None, alias="include[]"),
    chunking_strategy: Optional[str] = Form(
        default=None,
        description="Unsupported OpenAI chunking strategy.",
    ),
    known_speaker_names: Optional[List[str]] = Form(
        default=None,
        description="Unsupported OpenAI diarization speaker-name hints.",
    ),
    known_speaker_names_brackets: Optional[List[str]] = Form(default=None, alias="known_speaker_names[]"),
    known_speaker_references: Optional[List[str]] = Form(
        default=None,
        description="Unsupported OpenAI diarization speaker-reference audio.",
    ),
    known_speaker_references_brackets: Optional[List[str]] = Form(default=None, alias="known_speaker_references[]"),
    num_speakers: Optional[int] = Form(
        default=None,
        description=(
            "Local extension: expected number of speakers for this request's "
            "diarization. Overrides WHISPER_DIARIZE_NUM_SPEAKERS for this call "
            "only. Omit for automatic speaker-count detection."
        ),
    ),
    diarize: Optional[bool] = Form(
        default=None,
        description=(
            "Local extension: per-request override for whether diarization "
            "runs at all, even when WHISPER_DIARIZATION=true server-wide. "
            "Set to false to skip diarization for this call (faster — useful "
            "for single-speaker dictation or very large files). Omit to use "
            "the server-wide default."
        ),
    ),
    x_job_id: Optional[str] = Header(
        default=None,
        alias="X-Job-Id",
        description=(
            "Local extension: Auftragsnummer. Mit stream=true kann der laufende Auftrag darüber "
            "per POST /v1/jobs/{id}/cancel abgebrochen werden."
        ),
    ),
    _auth: None = Depends(_verify_api_key),
):
    """
    Transcribe an audio file.

    Supports standard Whisper-style OpenAI multipart/form-data parameters.
    Newer OpenAI-only diarization and logprob fields that this server cannot
    implement return clear 400 errors instead of being silently ignored.

    When stream=true the response is text/event-stream (SSE) using the OpenAI
    streaming transcription protocol.  Each event carries a JSON object:
      - delta frames: {"type":"transcript.text.delta","delta":"..."}
      - final frame:  {"type":"transcript.text.done","text":"full transcript"}
      - error frame:  {"error":{"type":"...","message":"..."}}

    Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac
    (all formats supported by ffmpeg).
    """
    _validate_openai_transcription_compat(
        model=model,
        response_format=response_format,
        include=_merge_form_lists(include, include_brackets),
        chunking_strategy=chunking_strategy,
        known_speaker_names=_merge_form_lists(known_speaker_names, known_speaker_names_brackets),
        known_speaker_references=_merge_form_lists(known_speaker_references, known_speaker_references_brackets),
    )
    return await _handle_audio(
        task="transcribe",
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        stream=stream,
        beam=beam,
        timestamp_granularities=timestamp_granularities,
        num_speakers=num_speakers,
        diarize=diarize,
        job_id=_clean_job_id(x_job_id),
    )


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(..., description="Audio file to translate to English"),
    model: str = Form(
        default="whisper-1",
        description="Model identifier (ignored — active model is used)",
    ),
    language: Optional[str] = Form(
        default=None,
        description="BCP-47 language code of the source audio (e.g. 'fr'). Omit for autodetect.",
    ),
    prompt: Optional[str] = Form(
        default=None,
        description="Optional text to guide the model's style or continue a previous segment.",
    ),
    response_format: str = Form(
        default="json",
        description="Output format: json, text, verbose_json, srt, vtt",
    ),
    temperature: float = Form(
        default=0.0,
        description="Sampling temperature between 0 and 1.",
    ),
    beam: Optional[str] = Form(
        default=None,
        description=(
            "Local faster-whisper extension: per-request beam size. "
            "Omit to use WHISPER_BEAM."
        ),
    ),
    stream: Optional[str] = Form(
        default=None,
        description=(
            "Stream segments as Server-Sent Events (text/event-stream). "
            "When true, the response is a series of 'data:' frames — one per "
            "decoded segment — followed by a final 'done' frame. "
            "response_format is ignored when stream=true."
        ),
    ),
    _auth: None = Depends(_verify_api_key),
):
    """
    Translate audio to English text.

    Compatible with common OpenAI translation request fields. The output is
    always in English. language and stream are local extensions.

    Not supported with English-only (.en) models — use a multilingual model.

    Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac
    (all formats supported by ffmpeg).
    """
    return await _handle_audio(
        task="translate",
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        stream=stream,
        beam=beam,
    )


# ---------------------------------------------------------------------------
# Entry point (used by run.sh: python3 /opt/src/api_server.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = _env_int("WHISPER_PORT", 9000)
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        log_level=_log_level_str.lower(),
        workers=1,  # single worker — model is loaded into process memory
    )
