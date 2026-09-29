"""Probe how the causal preprocessing treats amplitude information.

Supports scheme A2 in ``sleep_fm_improvement_proposals.md``. Uses the repo's own
``causal_preprocess_signal`` (causal filters + rolling EMA z-score with tau=60 s +
log amplitude compression) on synthetic signals:

1. Global gain invariance: ``pre(x)`` vs ``pre(0.3 * x)``.
2. Hypopnea-like excursion drops (30/50/90 %) on a 0.25 Hz respiratory-like signal
   at 32 Hz: how much of the drop is still visible after preprocessing, as a
   function of event duration.
3. A slow overnight decline of slow-wave amplitude (EEG-like, 128 Hz, 6 h).

Run from the repo root (needs numpy + scipy only)::

    python docs/research/amplitude_probe.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hypnos.data.preprocessing import causal_preprocess_signal

rng = np.random.default_rng(0)


def envelope(x: np.ndarray, fs: float, win_s: float = 4.0) -> np.ndarray:
    """Peak-to-peak envelope over non-overlapping windows of ``win_s`` seconds."""
    w = int(win_s * fs)
    n = len(x) // w
    seg = x[: n * w].reshape(n, w)
    return seg.max(axis=1) - seg.min(axis=1)


def gain_invariance() -> None:
    fs = 128
    t = np.arange(0, 600, 1 / fs)
    eeg = np.sin(2 * np.pi * 1.0 * t) * 60 + rng.normal(0, 10, t.size)
    a, _, _ = causal_preprocess_signal(eeg, fs=fs, modality="eeg")
    b, _, _ = causal_preprocess_signal(eeg * 0.3, fs=fs, modality="eeg")
    print("[1] gain invariance: max |pre(x) - pre(0.3x)| =", float(np.max(np.abs(a - b))))


def hypopnea_like_drops() -> None:
    fs = 32
    drops = (0.3, 0.5, 0.9)
    print("\n[2] 0.25 Hz respiratory-like signal, event starts at t=600 s")
    print("    rows: event duration; cols: true excursion drop -> drop still visible after preprocessing")
    print("    dur(s) " + " ".join(f"{int(d * 100):>8d}%" for d in drops))
    for dur in (10, 20, 30, 60, 120):
        row = []
        for drop in drops:
            t = np.arange(0, 1200, 1 / fs)
            amp = np.ones_like(t)
            amp[(t >= 600) & (t < 600 + dur)] = 1 - drop
            resp = amp * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.02, t.size)
            y, _, _ = causal_preprocess_signal(resp, fs=fs, modality="respiratory")
            env = envelope(y, fs)
            tt = np.arange(env.size) * 4.0
            base = np.median(env[(tt >= 400) & (tt < 590)])
            # Second half of the event: the worst case for the adaptive normaliser.
            sel = (tt >= 600 + dur / 2) & (tt < 600 + dur - 4)
            if not sel.any():
                sel = (tt >= 600) & (tt < 600 + dur - 4)
            row.append(1 - np.median(env[sel]) / base)
        print(f"    {dur:>6d} " + " ".join(f"{s * 100:>8.1f}%" for s in row))


def overnight_decline() -> None:
    fs = 128
    t = np.arange(0, 6 * 3600, 1 / fs)
    amp = np.linspace(150, 60, t.size)  # slow-wave peak-to-peak amplitude, uV
    so = amp / 2 * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0, 5, t.size)
    y, _, _ = causal_preprocess_signal(so, fs=fs, modality="eeg")
    raw_env = envelope(so, fs, 30.0)
    pre_env = envelope(y, fs, 30.0)
    k = raw_env.size // 6
    raw_first, raw_last = raw_env[:k].mean(), raw_env[-k:].mean()
    pre_first, pre_last = pre_env[:k].mean(), pre_env[-k:].mean()
    print("\n[3] slow-wave amplitude decline over 6 h (first vs last hour, p2p envelope)")
    print(f"    raw:          {raw_first:7.1f} -> {raw_last:7.1f}  (ratio {raw_last / raw_first:.2f})")
    print(f"    preprocessed: {pre_first:7.2f} -> {pre_last:7.2f}  (ratio {pre_last / pre_first:.2f})")


if __name__ == "__main__":
    gain_invariance()
    hypopnea_like_drops()
    overnight_decline()
