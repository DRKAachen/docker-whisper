"""
Speaker diarization via sherpa-onnx (ONNX Runtime, no PyTorch).

Downloads ONNX models on first use and provides speaker-segment alignment
for Whisper transcription output.

https://github.com/hwdsl2/docker-whisper

Copyright (C) 2026 Lin Song <linsongui@gmail.com>

This work is licensed under the MIT License
See: https://opensource.org/licenses/MIT
"""

import hashlib
import logging
import os
import subprocess
import tarfile
import time

import numpy as np
import sherpa_onnx

logger = logging.getLogger("whisper_server.diarizer")

# ---------------------------------------------------------------------------
# Model URLs (GitHub releases from k2-fsa/sherpa-onnx)
# ---------------------------------------------------------------------------

_SEG_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_EMB_RELEASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
)

# Speaker-embedding models. "eres2net" is the original default. "campplus" is about 3x faster per
# embedding (measured: ~130 ms vs ~394 ms per 10 s of audio on one core); pick it with
# WHISPER_DIARIZE_EMBEDDING=campplus, then compare the speaker labels on your own recordings.
_EMB_MODELS = {
    "eres2net": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "campplus": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
}

_SEG_MODEL_REL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
_SEG_ARCHIVE = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"

# SHA-256 of every model file, taken from the official k2-fsa/sherpa-onnx release assets. A downloaded or cached file that does not
# match is deleted and refused (a model is code-like input for the ONNX runtime: a swapped file must never be loaded).
# WHISPER_DIARIZE_SKIP_CHECKSUM=true switches the check off (e.g. if upstream re-publishes a model); this is logged loudly.
_SHA256 = {
    _SEG_ARCHIVE: "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    _SEG_MODEL_REL: "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx": "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b",
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx": "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
}


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _verify_file(path: str, key: str) -> None:
    """Raise RuntimeError (and delete the file) if the file does not match its pinned SHA-256."""
    if os.environ.get("WHISPER_DIARIZE_SKIP_CHECKSUM", "").strip().lower() in ("1", "true", "yes", "on"):
        logger.warning("Checksum check SKIPPED for %s (WHISPER_DIARIZE_SKIP_CHECKSUM is set)", key)
        return
    expected = _SHA256.get(key)
    if not expected:
        raise RuntimeError(f"No pinned checksum for {key}")
    actual = _sha256_of(path)
    if actual != expected:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(
            f"Checksum mismatch for {key} (expected {expected[:12]}..., got {actual[:12]}...); the file was deleted and not loaded"
        )


class DiarizationAborted(Exception):
    """Raised when the caller asked to stop (the progress callback returned True)."""


# Values given to load() (the server passes the WHISPER_DIARIZE_* settings); None = use the environment / defaults.
_cfg_embedding = None
_cfg_threads = None


def _embedding_name() -> str:
    name = (_cfg_embedding or os.environ.get("WHISPER_DIARIZE_EMBEDDING", "") or "eres2net").strip().lower()
    if name not in _EMB_MODELS:
        logger.warning(
            "Diarization embedding model %r is unknown, using 'eres2net' (allowed: %s)",
            name, ", ".join(_EMB_MODELS),
        )
        return "eres2net"
    return name


def _num_threads() -> int:
    """CPU threads for the ONNX sessions. sherpa-onnx defaults to ONE thread, which leaves
    every other core idle on long recordings. Order: the value given to load(), then
    WHISPER_DIARIZE_THREADS, otherwise all cores available to the container, at most 8."""
    if isinstance(_cfg_threads, int) and _cfg_threads > 0:
        return min(_cfg_threads, 32)
    raw = os.environ.get("WHISPER_DIARIZE_THREADS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return min(int(raw), 32)
    try:
        cpus = len(os.sched_getaffinity(0))  # respects cpuset limits of the container
    except AttributeError:  # not available on every platform
        cpus = os.cpu_count() or 1
    return max(1, min(cpus, 8))

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_sd = None  # OfflineSpeakerDiarization instance (default, loaded at startup)
_seg_path = None  # cached path, kept so per-request pipelines can be rebuilt cheaply
_emb_path = None
_default_threshold = 0.5
_embedding_extractor = None  # lazily built standalone embedding extractor, for sliver clean-up


def _download_file(url: str, dest: str) -> None:
    """Download a file using curl (available in the container)."""
    logger.info("Downloading %s -> %s", url, dest)
    subprocess.run(
        ["curl", "-fSL", "--retry", "3", "--retry-delay", "2", "-o", dest, url],
        check=True,
    )


def _ensure_models(cache_dir: str) -> tuple:
    """Download ONNX models if not already present. Returns (seg_path, emb_path)."""
    seg_path = os.path.join(cache_dir, _SEG_MODEL_REL)
    emb_file = _EMB_MODELS[_embedding_name()]
    emb_path = os.path.join(cache_dir, emb_file)

    if not os.path.isfile(seg_path):
        archive = os.path.join(cache_dir, _SEG_ARCHIVE)
        _download_file(_SEG_MODEL_URL, archive)
        _verify_file(archive, _SEG_ARCHIVE)  # check BEFORE unpacking
        logger.info("Extracting segmentation model...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=cache_dir, filter="data")
        os.unlink(archive)
    _verify_file(seg_path, _SEG_MODEL_REL)  # also for a model that was cached by an earlier start

    if not os.path.isfile(emb_path):
        _download_file(_EMB_RELEASE_URL + emb_file, emb_path)
    _verify_file(emb_path, emb_file)

    return seg_path, emb_path


def load(
    cache_dir: str = "/var/lib/whisper",
    num_speakers: int = -1,
    cluster_threshold: float = 0.5,
    embedding: str = None,
    num_threads: int = None,
) -> None:
    """Download ONNX models if needed, initialize the diarization pipeline.

    embedding: "eres2net" (default) or "campplus" (about 3x faster per embedding).
    num_threads: CPU threads for the ONNX sessions (default: see _num_threads).
    """
    global _sd, _seg_path, _emb_path, _default_threshold, _cfg_embedding, _cfg_threads
    _cfg_embedding, _cfg_threads = embedding, num_threads

    seg_path, emb_path = _ensure_models(cache_dir)
    _seg_path, _emb_path = seg_path, emb_path
    _default_threshold = cluster_threshold

    _sd = _build_pipeline(seg_path, emb_path, num_speakers, cluster_threshold)
    logger.info(
        "Diarization pipeline ready (sample_rate=%d, num_clusters=%d, threshold=%.2f, threads=%d, embedding=%s)",
        _sd.sample_rate,
        num_speakers if num_speakers > 0 else -1,
        cluster_threshold,
        _num_threads(),
        _embedding_name(),
    )


def _build_pipeline(seg_path: str, emb_path: str, num_speakers: int, cluster_threshold: float):
    """Construct a fresh OfflineSpeakerDiarization pipeline for a given speaker
    count. Cheap to call repeatedly: model files are already on disk (no
    download), this just builds the in-memory ONNX sessions + clustering
    config. Used both at startup and, when a caller asks for a different
    speaker count, per individual request."""
    num_clusters = num_speakers if num_speakers and num_speakers > 0 else -1
    threads = _num_threads()

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_path,
            ),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_path, num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=cluster_threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    if not config.validate():
        raise RuntimeError(
            "Diarization config validation failed. Check that model files exist."
        )

    return sherpa_onnx.OfflineSpeakerDiarization(config)


def is_loaded() -> bool:
    """Return True if the diarization pipeline is initialized."""
    return _sd is not None


def _load_audio(audio_path: str, target_sr: int = 16000):
    """
    Load and resample audio file to mono float32 at target_sr using PyAV.
    Uses the same approach as faster-whisper's decode_audio (AudioResampler).
    PyAV is already installed as a dependency of faster-whisper and supports
    all formats that FFmpeg supports (mp3, ogg, m4a, webm, wav, flac, etc.).
    """
    import io

    import av

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sr,
    )

    raw_buffer = io.BytesIO()
    with av.open(audio_path, mode="r", metadata_errors="ignore") as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                raw_buffer.write(resampled.to_ndarray())
        # Flush remaining buffered samples from the resampler
        for resampled in resampler.resample(None):
            if resampled.samples > 0:
                raw_buffer.write(resampled.to_ndarray())

    audio = np.frombuffer(raw_buffer.getvalue(), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    return audio


def diarize(audio_path: str, num_speakers: int = None, progress=None):
    """
    Run diarization on an audio file.

    num_speakers: optional per-request override of the expected speaker
    count. When given (and different from what's currently loaded), a fresh
    pipeline is built just for this call using the already-cached model
    files (no re-download) — this lets each request specify how many
    participants it expects (e.g. "4" for a known meeting size) instead of
    relying on a single server-wide guess, which tends to over-split a
    single voice into many spurious speakers on long or noisy recordings.

    progress: optional callable progress(fraction) -> bool. It is called with a value from 0.0
    to 1.0 while the work runs; if it returns True the work stops at the next step and
    DiarizationAborted is raised (the pipeline object stays usable afterwards).

    Returns a list of (start, end, speaker_label) tuples sorted by start time.
    """
    if _sd is None:
        raise RuntimeError("Diarizer not loaded. Call diarizer.load() first.")

    t_start = time.monotonic()

    def report(fraction: float) -> None:
        if progress is not None and progress(min(1.0, max(0.0, fraction))):
            raise DiarizationAborted()

    pipeline = _sd
    if num_speakers is not None and num_speakers > 0:
        pipeline = _build_pipeline(_seg_path, _emb_path, num_speakers, _default_threshold)

    report(0.0)
    audio = _load_audio(audio_path, target_sr=pipeline.sample_rate)
    t_loaded = time.monotonic()
    report(0.0)

    def on_chunk(done, total):
        # The embedding step is the bulk of the work (~90 %): report 0 - 95 % while it runs.
        report(0.95 * done / total if total else 0.0)
        return 0

    result = pipeline.process(audio, callback=on_chunk if progress is not None else None).sort_by_start_time()
    t_processed = time.monotonic()
    turns = []
    for r in result:
        turns.append((r.start, r.end, f"SPEAKER_{r.speaker:02d}"))

    # Clean-up pass: the underlying clustering occasionally splits off a
    # tiny, short-lived "phantom" speaker from a brief burst of noise or an
    # onset artifact (observed: a real 2-person recording producing 4
    # labels, two of which totalled under 1 second each). Such slivers are
    # reassigned based on actual VOICE similarity (cosine similarity of
    # speaker embeddings) to the other, larger speaker clusters — NOT by
    # temporal proximity, which was tried first and shown (by direct test)
    # to often merge a short reply into whichever speaker merely happened
    # to talk nearby in time, rather than the speaker who actually said it.
    turns = _merge_sliver_speakers_by_voice(turns, audio, pipeline.sample_rate)
    report(1.0)
    logger.info(
        "Diarization done: %.0f s of audio, %d turns, %d speakers in %.1f s (load %.1f s, analyse %.1f s, clean-up %.1f s, threads=%d)",
        len(audio) / pipeline.sample_rate, len(turns), len({t[2] for t in turns}),
        time.monotonic() - t_start, t_loaded - t_start, t_processed - t_loaded,
        time.monotonic() - t_processed, _num_threads(),
    )
    return turns


def _get_embedding_extractor():
    global _embedding_extractor
    if _embedding_extractor is None:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=_emb_path, num_threads=_num_threads())
        _embedding_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    return _embedding_extractor


def _embed_segment(audio, sample_rate, start, end, max_seconds: float = 10.0):
    extractor = _get_embedding_extractor()
    stream = extractor.create_stream()
    if end - start > max_seconds:  # a 10 s sample is enough to characterise a voice; long turns only cost time
        middle = (start + end) / 2
        start, end = middle - max_seconds / 2, middle + max_seconds / 2
    chunk = audio[int(start * sample_rate):int(end * sample_rate)]
    stream.accept_waveform(sample_rate=sample_rate, waveform=chunk)
    stream.input_finished()
    return np.array(extractor.compute(stream))


def _cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _merge_sliver_speakers_by_voice(
    turns, audio, sample_rate, min_total_seconds: float = 1.5, min_share: float = 0.08,
    max_turns_per_speaker: int = 12,
):
    """
    Fold speakers with very little total talk time into whichever other
    speaker they actually sound most like (voice-embedding cosine
    similarity), rather than whichever speaker merely talks nearby in time.
    """
    if len(turns) <= 1:
        return turns

    totals = {}
    for start, end, speaker in turns:
        totals[speaker] = totals.get(speaker, 0.0) + (end - start)
    if len(totals) <= 1:
        return turns

    longest_total = max(totals.values())
    slivers = {
        spk for spk, total in totals.items()
        if total < min_total_seconds and total < min_share * longest_total
    }
    if not slivers:
        return turns

    main_speakers = [spk for spk in totals if spk not in slivers]
    if not main_speakers:
        return turns  # everything is a "sliver" (very short clip) — nothing sensible to merge into

    # Centroid embedding per main (non-sliver) speaker. Averaged over the speaker's longest turns
    # only (at most max_turns_per_speaker): on a long recording a speaker can have thousands of
    # turns, and embedding every one of them took minutes for a result that barely changes.
    centroids = {}
    for spk in main_speakers:
        own = sorted((t for t in turns if t[2] == spk), key=lambda t: t[1] - t[0], reverse=True)
        embs = [
            _embed_segment(audio, sample_rate, s, e)
            for s, e, _sp in own[:max_turns_per_speaker]
        ]
        centroids[spk] = np.mean(embs, axis=0)

    cleaned = list(turns)
    for i, (start, end, speaker) in enumerate(cleaned):
        if speaker not in slivers:
            continue
        seg_emb = _embed_segment(audio, sample_rate, start, end)
        best_spk = max(main_speakers, key=lambda spk: _cosine_sim(seg_emb, centroids[spk]))
        cleaned[i] = (start, end, best_spk)

    return cleaned


def resegment_by_word(segments_with_words, diarization_turns):
    """
    Rebuild segments at the word level instead of the whisper-segment level.

    Problem this solves: a single Whisper segment can span several real
    speaker turns (e.g. a 6-second segment covering "good morning steve" /
    "good morning katie" / the start of a longer sentence) — assigning one
    speaker to the whole segment blurs or loses short interjections like
    greetings. By checking each individual word's overlap with the
    diarization turns and only starting a new output segment when the
    speaker actually changes, short exchanges are preserved correctly.

    Args:
        segments_with_words: iterable of Whisper segments, each exposing
            .words (iterable of objects/dicts with word/start/end).
        diarization_turns: list of (start, end, speaker_label) tuples.

    Returns:
        List of dicts: {"start", "end", "text", "speaker"} — one per
        contiguous run of same-speaker words. Falls back to the original
        whisper-segment granularity for any segment that has no word-level
        timing available.
    """
    if not diarization_turns:
        return None  # caller should fall back to whole-segment assignment

    def best_speaker_for(w_start, w_end):
        best_speaker, best_overlap = "SPEAKER_00", 0.0
        for turn_start, turn_end, speaker in diarization_turns:
            overlap = max(0.0, min(w_end, turn_end) - max(w_start, turn_start))
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
        return best_speaker

    out = []
    current = None
    for seg in segments_with_words:
        words = getattr(seg, "words", None) or (seg.get("words") if isinstance(seg, dict) else None)
        if not words:
            # No word-level timing for this segment (e.g. it was pure
            # silence/music) — treat the whole segment as one unit.
            words = [type("W", (), {
                "word": (seg.text if hasattr(seg, "text") else seg.get("text", "")),
                "start": seg.start if hasattr(seg, "start") else seg.get("start"),
                "end": seg.end if hasattr(seg, "end") else seg.get("end"),
            })]

        for w in words:
            w_word = w.word if hasattr(w, "word") else w.get("word", "")
            w_start = w.start if hasattr(w, "start") else w.get("start")
            w_end = w.end if hasattr(w, "end") else w.get("end")
            if w_start is None or w_end is None:
                continue
            speaker = best_speaker_for(w_start, w_end)

            if current and current["speaker"] == speaker:
                current["text"] += w_word
                current["end"] = w_end
            else:
                if current:
                    out.append(current)
                current = {"start": w_start, "end": w_end, "text": w_word, "speaker": speaker}

    if current:
        out.append(current)

    for seg in out:
        seg["text"] = seg["text"].strip()

    return out


def assign_speakers(segments, diarization_turns):
    """
    Assign a speaker label to each Whisper segment based on maximum time overlap
    with diarization turns.

    Args:
        segments: list of segment dicts with 'start' and 'end' keys (seconds).
        diarization_turns: list of (start, end, speaker_label) tuples.

    Returns:
        The same segments list with a 'speaker' key added to each segment.
    """
    if not diarization_turns:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for turn_start, turn_end, speaker in diarization_turns:
            # Calculate overlap
            overlap_start = max(seg_start, turn_start)
            overlap_end = min(seg_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        seg["speaker"] = best_speaker

    return segments
