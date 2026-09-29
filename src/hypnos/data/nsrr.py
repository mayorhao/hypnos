"""NSRR polysomnography preprocessing for pretraining.

Turns one NSRR night (EDF + ``*-nsrr.xml`` annotations) into one ``.npz`` on a common 1 Hz
grid, for the tokenizer / language-model pipeline:

* **Waveforms** (EEG, EOG, chin EMG, ECG, respiratory effort, airflow, nasal pressure):
  resampled to the modality rate and passed through the same causal preprocessing as the
  released Hypnos tokenizers (causal filters, rolling EMA z-score, log compression), so the
  signals stay drop-in compatible with Hypnos-style tokenizers.
* **Scale**: the rolling z-score removes absolute amplitude. The per-second normaliser scale
  (``log σ_t``, exactly the σ used for normalisation) is kept alongside every waveform, so a
  model can be given (and must generate) amplitude explicitly.
* **SpO2**: no filtering or normalisation; cleaned per-second value plus 1 %-bin tokens.
* **Physiological descriptors** for semantic targets: EEG/EOG band log-powers per second
  (causal trailing window), ECG R-peaks detected at the *native* sampling rate, per-second
  RR, and per-epoch RMSSD (long-window HRV; do not use per-second HRV).
* **Annotations** from NSRR XML: 30 s stages and per-second event masks (arousal, apnea
  types, hypopnea, desaturation, SpO2 artifact, limb movement).

CLI::

    python -m hypnos.data.nsrr inventory --edf-dir shhs/polysomnography/edfs --out labels.csv
    python -m hypnos.data.nsrr preprocess --edf-dir .../edfs --xml-dir .../annotations-events-nsrr \\
        --out-dir processed/shhs --notch 60 --workers 8

Always run ``inventory`` on a new cohort first: channel labels differ between cohorts and
visits, and the aliases below must be checked against the actual EDF headers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyedflib
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import skew

from .edf import get_column_match, load_psg_channels
from .preprocessing import (
    MODALITY_CONFIGS,
    causal_bandpass_filter,
    causal_highpass_filter,
    causal_notch_filter,
    causal_preprocess_signal,
    resample_signal,
)

logger = logging.getLogger(__name__)

PREPROCESS_VERSION = 1
EPOCH_SEC = 30


@dataclass(frozen=True)
class ModalitySpec:
    """How one modality is extracted and preprocessed."""

    channel: str  # canonical channel resolved by load_psg_channels
    sample_rate: int  # target rate in Hz
    preprocess: str | None  # key into MODALITY_CONFIGS; None = no filtering / normalisation
    microvolts: bool  # electrophysiology: convert EDF physical units to uV


# Target rates mirror the released Hypnos tokenizers (128 Hz electrophysiology, 32 Hz respiration).
MODALITIES: dict[str, ModalitySpec] = {
    "eeg_c3": ModalitySpec("C3", 128, "eeg", True),
    "eeg_c4": ModalitySpec("C4", 128, "eeg", True),
    "eog_e1": ModalitySpec("E1", 128, "eeg", True),
    "eog_e2": ModalitySpec("E2", 128, "eeg", True),
    "emg_chin": ModalitySpec("Chin", 128, "emg", True),
    "ecg": ModalitySpec("ECG", 128, "ecg", True),
    "resp_abd": ModalitySpec("ABD", 32, "respiratory", False),
    "resp_thx": ModalitySpec("THX", 32, "respiratory", False),
    "airflow": ModalitySpec("AIRFLOW", 32, "respiratory", False),
    "nasal_pressure": ModalitySpec("NASAL_PRESSURE", 32, "respiratory", False),
    "spo2": ModalitySpec("SPO2", 1, None, False),
}

# Labels for channels the base loader does not know. Verify against `inventory` output:
# "Flow" is a thermistor in some cohorts and a pressure cannula in others.
NSRR_CHANNEL_ALIASES: dict[str, list[str]] = {
    "SPO2": ["SpO2", "SaO2", "SAO2", "SpO2 BB", "SpO2-BB", "OSAT", "Osat"],
    "AIRFLOW": ["Airflow", "AIRFLOW", "Flow", "Therm", "THERM", "Thermistor", "Oral Thermistor", "Airflow Therm"],
    "NASAL_PRESSURE": ["Pres", "PRES", "Nasal Pressure", "NASAL PRES", "Nasal Pres", "Cannula Flow", "PTAF", "NEW AIR"],
}

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

# NSRR XML EventConcept (text before '|', lower-cased) -> stage code. Stage 4 merges into N3.
STAGE_MAP: dict[str, int] = {
    "wake": 0,
    "stage 1 sleep": 1,
    "stage 2 sleep": 2,
    "stage 3 sleep": 3,
    "stage 4 sleep": 3,
    "rem sleep": 4,
}
STAGE_UNSCORED = -1

# First matching class wins, so RERA is tested before the generic arousal class.
EVENT_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rera", ("respiratory effort related arousal", "rera")),
    ("arousal", ("arousal",)),
    ("obstructive_apnea", ("obstructive apnea",)),
    ("central_apnea", ("central apnea",)),
    ("mixed_apnea", ("mixed apnea",)),
    ("hypopnea", ("hypopnea",)),
    ("desaturation", ("desaturation",)),
    ("spo2_artifact", ("spo2 artifact",)),
    ("limb_movement", ("limb movement", "leg movement")),
)

# Per-second quality bits.
QC_FLAT = 1
QC_CLIPPED = 2
QC_NONFINITE = 4

SPO2_FLOOR = 60  # tokens: 0 invalid, 1 below floor, 2.. = FLOOR..100 in 1 % bins
SPO2_VOCAB = 100 - SPO2_FLOOR + 3

_UV_SCALE = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}


# ---------------------------------------------------------------------------
# Signal-level helpers
# ---------------------------------------------------------------------------


def to_microvolts(signal: np.ndarray, unit: str) -> tuple[np.ndarray, bool]:
    """Convert an electrophysiology signal to uV from its EDF physical unit.

    Returns ``(signal, converted)``; ``converted`` is False for unknown units, in which case
    the signal is returned unchanged and absolute-amplitude features must be treated as
    uncalibrated for that record.
    """
    scale = _UV_SCALE.get(unit.strip().lower().replace(" ", ""))
    if scale is None:
        return signal, False
    return signal * scale, True


def per_second_qc(signal: np.ndarray, fs: int, n_seconds: int, physical_range: tuple[float, float]) -> np.ndarray:
    """Per-second quality bitmask at the native rate: flat line, clipping, non-finite samples."""
    seg = np.asarray(signal[: n_seconds * fs], dtype=np.float64).reshape(n_seconds, fs)
    qc = np.zeros(n_seconds, dtype=np.uint8)
    nonfinite = ~np.isfinite(seg)
    qc[nonfinite.any(axis=1)] |= QC_NONFINITE
    seg = np.where(nonfinite, 0.0, seg)

    ptp = seg.max(axis=1) - seg.min(axis=1)
    ref = np.median(ptp[ptp > 0]) if np.any(ptp > 0) else 0.0
    qc[ptp <= max(1e-3 * ref, 1e-12)] |= QC_FLAT

    lo, hi = physical_range
    margin = 1e-3 * (hi - lo)
    at_rail = (seg <= lo + margin) | (seg >= hi - margin)
    qc[at_rail.mean(axis=1) > 0.05] |= QC_CLIPPED
    return qc


def causal_filter(signal: np.ndarray, fs: int, modality: str, notch_freq: float) -> np.ndarray:
    """The filtering part of ``causal_preprocess_signal`` without normalisation (keeps units)."""
    config = MODALITY_CONFIGS[modality]
    x = np.asarray(signal, dtype=np.float64)
    if config["lowpass"] is not None:
        x = causal_bandpass_filter(x, config["highpass"], config["lowpass"], fs)
    else:
        x = causal_highpass_filter(x, config["highpass"], fs)
    if config["notch"]:
        x = causal_notch_filter(x, notch_freq, fs)
    return x


def band_log_power(x: np.ndarray, fs: int, n_seconds: int, window_sec: float = 4.0) -> np.ndarray:
    """Per-second log10 band power (uV^2) over a causal trailing window.

    Row ``t`` uses samples ending at the end of second ``t``; rows without a full window are
    NaN. Returns ``(n_seconds, len(BANDS))`` float32.
    """
    win = int(window_sec * fs)
    out = np.full((n_seconds, len(BANDS)), np.nan, dtype=np.float32)
    x = np.asarray(x[: n_seconds * fs], dtype=np.float64)
    first = int(np.ceil(win / fs)) - 1  # first second with a full trailing window
    if n_seconds <= first:
        return out
    taper = np.hanning(win)
    norm = fs * np.sum(taper**2)
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    df = freqs[1] - freqs[0]
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS.values()]
    windows = sliding_window_view(x, win)[::fs]  # window k covers [k*fs, k*fs + win)
    for start in range(0, n_seconds - first, 2048):  # chunk to bound memory
        stop = min(start + 2048, n_seconds - first)
        seg = windows[start:stop]
        seg = seg - seg.mean(axis=1, keepdims=True)
        psd = np.abs(np.fft.rfft(seg * taper, axis=1)) ** 2 / norm
        psd[:, 1:-1] *= 2.0  # one-sided spectrum
        for b, mask in enumerate(masks):
            out[first + start : first + stop, b] = np.log10(psd[:, mask].sum(axis=1) * df + 1e-12)
    return out


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """R-peak sample indices at the native rate (Pan-Tompkins-style energy detector).

    Polarity is inferred from the skewness of the QRS-band signal, so inverted leads work.
    Uses zero-phase filtering: R-peak *times* are local, and descriptors built on them are
    evaluated per second, so the few-ms lookahead does not leak across tokens.
    """
    ecg = np.nan_to_num(np.asarray(ecg, dtype=np.float64))
    sos = butter(3, [5.0, min(20.0, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    qrs = sosfiltfilt(sos, ecg)
    if skew(qrs) < 0:
        qrs = -qrs
    energy = np.gradient(qrs) ** 2
    width = max(1, int(0.15 * fs))
    integrated = np.convolve(energy, np.ones(width) / width, mode="same")

    block = int(10 * fs)  # adaptive threshold per 10 s block
    threshold = np.empty_like(integrated)
    for start in range(0, len(integrated), block):
        seg = integrated[start : start + block]
        threshold[start : start + block] = 0.3 * np.percentile(seg, 98)
    candidates, _ = find_peaks(integrated, height=threshold, distance=max(1, int(0.3 * fs)))

    half = max(1, int(0.075 * fs))  # refine to the QRS maximum (integration shifts the peak)
    peaks = []
    for p in candidates:
        lo = max(0, p - half)
        peaks.append(lo + int(np.argmax(qrs[lo : p + half + 1])))
    return np.unique(np.asarray(peaks, dtype=np.int64))


def rr_descriptors(
    r_times: np.ndarray, n_seconds: int, max_gap_sec: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Beat-level RR descriptors on the 1 Hz grid.

    Returns ``(rr_sec, beats, rmssd_ms, valid_rr)``:
      * ``rr_sec`` (n_seconds,): last valid RR interval ending within second ``t`` or the
        preceding ``max_gap_sec`` (causal), NaN otherwise. Heart rate is 60 / rr, so it is
        not stored separately.
      * ``beats`` (n_seconds,): number of R-peaks in each second.
      * ``rmssd_ms`` (n_seconds // 30,): RMSSD over valid successive NN pairs per epoch.
      * ``valid_rr`` (len(r_times) - 1,): physiological-range and local-median plausibility.
    """
    beats = np.bincount(np.floor(r_times).astype(np.int64).clip(0, max(n_seconds - 1, 0)), minlength=n_seconds)
    beats = beats[:n_seconds].astype(np.uint8)
    n_epochs = n_seconds // EPOCH_SEC
    rr_sec = np.full(n_seconds, np.nan, dtype=np.float32)
    rmssd = np.full(n_epochs, np.nan, dtype=np.float32)
    if len(r_times) < 3:
        return rr_sec, beats, rmssd, np.zeros(max(len(r_times) - 1, 0), dtype=bool)

    rr = np.diff(r_times)
    t_rr = r_times[1:]
    local = median_filter(rr, size=5, mode="nearest")
    valid = (rr >= 0.3) & (rr <= 2.0) & (np.abs(rr - local) <= 0.2 * local)

    tv, rv = t_rr[valid], rr[valid]
    if tv.size:
        ends = np.arange(1, n_seconds + 1, dtype=np.float64)
        idx = np.searchsorted(tv, ends, side="right") - 1
        ok = (idx >= 0) & (tv[idx.clip(0)] > ends - max_gap_sec)
        rr_sec[ok] = rv[idx[ok]]

    succ = valid[1:] & valid[:-1]  # successive differences only between two valid intervals
    diffs = np.diff(rr)[succ]
    t_diff = t_rr[1:][succ]
    epoch_of = np.floor(t_diff / EPOCH_SEC).astype(np.int64)
    for e in range(n_epochs):
        d = diffs[epoch_of == e]
        if d.size >= 5:
            rmssd[e] = 1e3 * np.sqrt(np.mean(d**2))
    return rr_sec, beats, rmssd, valid


def spo2_per_second(signal: np.ndarray, fs: int, n_seconds: int, valid_range=(50.0, 100.0)) -> np.ndarray:
    """Per-second SpO2 (median of samples); implausible values become NaN."""
    x = np.asarray(signal[: n_seconds * fs], dtype=np.float64)
    x = np.where((x >= valid_range[0]) & (x <= valid_range[1]), x, np.nan)
    seg = x.reshape(n_seconds, fs)
    out = np.full(n_seconds, np.nan, dtype=np.float32)
    has = np.isfinite(seg).any(axis=1)
    out[has] = np.nanmedian(seg[has], axis=1)
    return out


def spo2_tokens(spo2: np.ndarray, floor: int = SPO2_FLOOR) -> np.ndarray:
    """Discretise per-second SpO2: 0 invalid, 1 below ``floor``, 2.. = floor..100 % in 1 % bins."""
    tokens = np.zeros(len(spo2), dtype=np.int16)
    valid = np.isfinite(spo2)
    v = np.round(spo2[valid])
    tokens[valid] = np.where(v < floor, 1, v - floor + 2).astype(np.int16)
    return tokens


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


@dataclass
class Annotations:
    stages: np.ndarray  # (n_epochs,) int8, STAGE_UNSCORED where unscored
    masks: dict[str, np.ndarray]  # event class -> (n_seconds,) uint8
    events: list[tuple[str, float, float]]  # (class, start_sec, duration_sec)


def _concept(text: str | None) -> str:
    return (text or "").split("|")[0].strip().lower()


def classify_event(concept: str) -> str | None:
    """Map an NSRR EventConcept to one of ``EVENT_CLASSES`` (None if not tracked)."""
    for name, keys in EVENT_CLASSES:
        if any(k in concept for k in keys):
            return name
    return None


def parse_nsrr_xml(path: str | Path, n_seconds: int) -> Annotations:
    """Parse an NSRR ``*-nsrr.xml`` file onto the recording's 1 Hz / 30 s grids."""
    root = ET.parse(str(path)).getroot()
    n_epochs = n_seconds // EPOCH_SEC
    stages = np.full(n_epochs, STAGE_UNSCORED, dtype=np.int8)
    masks = {name: np.zeros(n_seconds, dtype=np.uint8) for name, _ in EVENT_CLASSES}
    events: list[tuple[str, float, float]] = []

    for ev in root.iter("ScoredEvent"):
        concept = _concept(ev.findtext("EventConcept"))
        try:
            start = float(ev.findtext("Start") or "nan")
            duration = float(ev.findtext("Duration") or "0")
        except ValueError:
            continue
        if not np.isfinite(start):
            continue
        event_type = _concept(ev.findtext("EventType"))
        if event_type.startswith("stages"):
            code = STAGE_MAP.get(concept)
            if code is None:
                continue
            e0 = int(np.floor(start / EPOCH_SEC + 1e-6))
            e1 = int(np.floor((start + duration) / EPOCH_SEC + 1e-6))
            stages[max(e0, 0) : min(e1, n_epochs)] = code
            continue
        name = classify_event(concept)
        if name is None:
            continue
        s0 = max(int(np.floor(start)), 0)
        s1 = min(int(np.ceil(start + duration)), n_seconds)
        if s1 > s0:
            masks[name][s0:s1] = 1
        events.append((name, start, duration))
    return Annotations(stages=stages, masks=masks, events=events)


# ---------------------------------------------------------------------------
# Record pipeline
# ---------------------------------------------------------------------------


def preprocess_record(
    edf_path: str | Path,
    xml_path: str | Path | None = None,
    *,
    modalities: Sequence[str] | None = None,
    notch_freq: float = 60.0,
    tau_seconds: float = 60.0,
    channel_aliases: Mapping[str, Sequence[str]] | None = None,
    float16: bool = True,
) -> dict[str, np.ndarray]:
    """Preprocess one night into a flat ``{key: array}`` dict (see module docstring).

    Keys are ``"<modality>.<field>"`` plus ``stages``, ``events.*``, ``event.<class>`` and a
    JSON ``meta`` string. All per-second arrays share length ``n_seconds``; stages have
    ``n_seconds // 30`` entries. NSRR cohorts are US recordings, hence ``notch_freq=60``.
    """
    names = list(modalities) if modalities is not None else list(MODALITIES)
    aliases: dict[str, list[str]] = {k: list(v) for k, v in NSRR_CHANNEL_ALIASES.items()}
    for k, v in (channel_aliases or {}).items():
        aliases[k] = list(v) + aliases.get(k, [])

    with pyedflib.EdfReader(str(edf_path)) as f:
        channels = sorted({MODALITIES[m].channel for m in names})
        resolved = load_psg_channels(f, channels, drop_unreferenced=True, channel_aliases=aliases)

    present = [
        m for m in names if MODALITIES[m].channel in resolved and resolved[MODALITIES[m].channel].sampling_rate > 0
    ]
    if not present:
        raise ValueError(f"{edf_path}: none of the requested channels were found")
    n_seconds = min(
        len(resolved[MODALITIES[m].channel].signal) // resolved[MODALITIES[m].channel].sampling_rate for m in present
    )
    if n_seconds < EPOCH_SEC:
        raise ValueError(f"{edf_path}: recording shorter than one epoch ({n_seconds} s)")

    out: dict[str, np.ndarray] = {}
    meta: dict = {"version": PREPROCESS_VERSION, "edf": Path(edf_path).name, "n_seconds": n_seconds, "modalities": {}}
    wave_dtype = np.float16 if float16 else np.float32

    for m in present:
        spec = MODALITIES[m]
        rc = resolved[spec.channel]
        native_fs = rc.sampling_rate
        raw = np.asarray(rc.signal, dtype=np.float64)
        converted = False
        if spec.microvolts:
            raw, converted = to_microvolts(raw, rc.unit)
        raw = np.nan_to_num(raw[: n_seconds * native_fs])
        meta["modalities"][m] = {
            "edf_labels": rc.edf_labels,
            "method": rc.method,
            "native_fs": native_fs,
            "unit": rc.unit,
            "microvolts": converted,
            "target_fs": spec.sample_rate,
        }

        if spec.preprocess is None:  # SpO2: values are the information, no normalisation
            # 1 Hz values are constant within a second and 100 % sits on the rail, so the
            # flat/clipping checks do not apply; invalid seconds are the NaNs.
            out[f"{m}.value"] = spo2_per_second(raw, native_fs, n_seconds)
            continue
        out[f"{m}.qc"] = per_second_qc(np.asarray(rc.signal), native_fs, n_seconds, (rc.physical_min, rc.physical_max))

        fs = spec.sample_rate
        sig = resample_signal(raw, native_fs, fs)[: n_seconds * fs]
        processed, _, std_track = causal_preprocess_signal(
            sig, fs=fs, modality=spec.preprocess, notch_freq=notch_freq, tau_seconds=tau_seconds
        )
        out[f"{m}.signal"] = processed.astype(wave_dtype)
        # σ at the last sample of each second: the scale actually used by the normaliser.
        out[f"{m}.log_scale"] = np.log(std_track[fs - 1 :: fs][:n_seconds]).astype(np.float32)

        if m.startswith(("eeg_", "eog_")):
            filtered = causal_filter(sig, fs, spec.preprocess, notch_freq)
            out[f"{m}.band_logpow"] = band_log_power(filtered, fs, n_seconds)
        if m == "ecg":
            r_idx = detect_r_peaks(raw, native_fs)  # native rate: better R-peak timing
            r_times = r_idx / native_fs
            rr_sec, beats, rmssd, _ = rr_descriptors(r_times, n_seconds)
            out["ecg.r_peaks_sec"] = r_times.astype(np.float64)
            out["ecg.rr_sec"] = rr_sec
            out["ecg.beats"] = beats
            out["ecg.rmssd_ms"] = rmssd

    if xml_path is not None:
        ann = parse_nsrr_xml(xml_path, n_seconds)
        out["stages"] = ann.stages
        for name, mask in ann.masks.items():
            out[f"event.{name}"] = mask
        out["events.label"] = np.asarray([e[0] for e in ann.events], dtype="<U24")
        out["events.start_sec"] = np.asarray([e[1] for e in ann.events], dtype=np.float64)
        out["events.duration_sec"] = np.asarray([e[2] for e in ann.events], dtype=np.float64)
        meta["xml"] = Path(xml_path).name
        if "spo2.value" in out:  # scorer-marked oximetry artifact -> invalid
            out["spo2.value"] = np.where(ann.masks["spo2_artifact"] > 0, np.nan, out["spo2.value"]).astype(np.float32)
    if "spo2.value" in out:
        out["spo2.tokens"] = spo2_tokens(out["spo2.value"])
        out["spo2.qc"] = np.where(np.isfinite(out["spo2.value"]), 0, QC_NONFINITE).astype(np.uint8)

    out["meta"] = np.asarray(json.dumps(meta))
    return out


# ---------------------------------------------------------------------------
# Batch / CLI
# ---------------------------------------------------------------------------


def find_annotations(edf_paths: Sequence[Path], xml_dir: Path | None) -> dict[Path, Path | None]:
    """Pair EDFs with NSRR XMLs by name (``<stem>-nsrr.xml``, falling back to ``<stem>.xml``)."""
    if xml_dir is None:
        return {p: None for p in edf_paths}
    by_name = {x.name: x for x in xml_dir.rglob("*.xml")}
    return {p: by_name.get(f"{p.stem}-nsrr.xml") or by_name.get(f"{p.stem}.xml") for p in edf_paths}


def _process_one(edf: Path, xml: Path | None, out_dir: Path, kwargs: dict) -> dict:
    target = out_dir / f"{edf.stem}.npz"
    row = {"record": edf.stem, "edf": str(edf), "xml": str(xml) if xml else "", "status": "ok"}
    try:
        if not target.exists():
            arrays = preprocess_record(edf, xml, **kwargs)
            tmp = target.with_suffix(".tmp.npz")
            np.savez(tmp, **arrays)
            tmp.replace(target)
        with np.load(target) as d:
            meta = json.loads(str(d["meta"]))
            row["n_seconds"] = meta["n_seconds"]
            row["modalities"] = " ".join(sorted(meta["modalities"]))
            for m in meta["modalities"]:
                row[f"good_frac.{m}"] = round(float((d[f"{m}.qc"] == 0).mean()), 4)
    except Exception as e:  # keep the batch going; the manifest records the failure
        row["status"] = f"error: {type(e).__name__}: {e}"
    return row


def run_batch(edf_dir: Path, xml_dir: Path | None, out_dir: Path, workers: int = 1, **kwargs) -> Path:
    """Preprocess every EDF under ``edf_dir`` (resumable) and write ``manifest.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    edfs = sorted(edf_dir.rglob("*.edf")) + sorted(edf_dir.rglob("*.EDF"))
    pairs = find_annotations(edfs, xml_dir)
    rows: list[dict] = []
    if workers <= 1:
        rows = [_process_one(e, x, out_dir, kwargs) for e, x in pairs.items()]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, e, x, out_dir, kwargs) for e, x in pairs.items()]
            rows = [fut.result() for fut in as_completed(futures)]
    rows.sort(key=lambda r: r["record"])
    manifest = out_dir / "manifest.csv"
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("good_frac"), k))
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def inventory(edf_dir: Path, out_csv: Path) -> Counter:
    """List every EDF signal label (header only) and the modality it resolves to."""
    aliases = {k: list(v) for k, v in NSRR_CHANNEL_ALIASES.items()}
    counts: Counter = Counter()
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["edf", "label", "fs", "unit", "resolves_to"])
        for edf in sorted(edf_dir.rglob("*.edf")):
            try:
                with pyedflib.EdfReader(str(edf)) as f:
                    labels = f.getSignalLabels()
                    headers = [
                        (lab, f.getSampleFrequency(i), f.getPhysicalDimension(i)) for i, lab in enumerate(labels)
                    ]
            except OSError as e:
                logger.warning("cannot read %s: %s", edf, e)
                continue
            for lab, fs, unit in headers:
                hit = [m for m, s in MODALITIES.items() if get_column_match(s.channel, [lab], aliases) == lab]
                writer.writerow([edf.name, lab, fs, unit.strip(), " ".join(hit)])
                counts[(lab, " ".join(hit))] += 1
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory", help="list EDF channel labels and how they resolve")
    inv.add_argument("--edf-dir", type=Path, required=True)
    inv.add_argument("--out", type=Path, required=True)
    pre = sub.add_parser("preprocess", help="preprocess every EDF into one .npz per night")
    pre.add_argument("--edf-dir", type=Path, required=True)
    pre.add_argument("--xml-dir", type=Path, default=None)
    pre.add_argument("--out-dir", type=Path, required=True)
    pre.add_argument("--notch", type=float, default=60.0)
    pre.add_argument("--tau", type=float, default=60.0)
    pre.add_argument("--workers", type=int, default=1)
    pre.add_argument("--modalities", nargs="*", default=None, choices=list(MODALITIES))
    pre.add_argument("--float32", action="store_true", help="store waveforms as float32 instead of float16")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "inventory":
        counts = inventory(args.edf_dir, args.out)
        for (label, hit), n in counts.most_common():
            print(f"{n:6d}  {label!r:28s} -> {hit or '-'}")
    else:
        manifest = run_batch(
            args.edf_dir,
            args.xml_dir,
            args.out_dir,
            workers=args.workers,
            modalities=args.modalities,
            notch_freq=args.notch,
            tau_seconds=args.tau,
            float16=not args.float32,
        )
        print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
