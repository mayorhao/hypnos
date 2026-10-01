"""Probe: what does a reconstruction-driven per-second codebook encode?

Supports failure mode F1 (reconstruction-driven tokens track the high-power
aperiodic background, not the low-power oscillatory events that carry sleep
semantics) in ``sleep_fm_hypothesis_generation_v2.md``.

Synthetic EEG-like signals (128 Hz) are generated for several "subjects". Each
subject has its own aperiodic 1/f^alpha background (alpha and scale differ between
subjects). Low-power oscillatory events (alpha burst, sigma spindle, beta burst)
are inserted at random times; every second is labelled by the event class
present (``none`` if none). Signals go through the repo's own causal preprocessing
(causal filters + rolling EMA z-score + log compression) and are cut into 1 s
windows, exactly the token duration of the released Hypnos tokenizers.

Three per-second "tokenizers" (k-means codebooks, the optimal quantizer for a
mean-squared reconstruction objective in the chosen feature space) are compared:

  A. waveform MSE     : k-means on the raw 1 s waveform (what a reconstruction
                        objective optimises).
  B. flat spectrum    : k-means on the log power spectrum with every frequency bin
                        standardised across the corpus (spectrally balanced
                        target, cf. FAME).
  C. band descriptors : k-means on standardised delta/theta/alpha/sigma/beta
                        log band powers (an explicit physiological descriptor
                        target, cf. proposal P1).

For each codebook we report normalised mutual information between the code and
(i) the event class (semantic), (ii) the subject (nuisance), and the accuracy of a
majority-vote class decoder from the code (train/test split over subjects).

Run from the repo root (numpy + scipy only)::

    python docs/research/semantic_vq_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hypnos.data.preprocessing import causal_preprocess_signal

FS = 128
SECONDS_PER_SUBJECT = 400
N_SUBJECTS = 24
CODEBOOK = 64
CLASSES = ("none", "kcomplex", "alpha", "sigma", "beta")
# Low-power oscillatory events (6-25 % of background power) plus one high-amplitude
# event (K-complex-like biphasic wave at 1.5-2x background) as a positive control.
EVENT_FREQ = {"alpha": (8.5, 11.5), "sigma": (12.0, 15.0), "beta": (18.0, 28.0)}
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}

rng = np.random.default_rng(0)


def aperiodic_background(n: int, alpha: float, scale: float) -> np.ndarray:
    """1/f^alpha Gaussian noise via spectral shaping."""
    freqs = np.fft.rfftfreq(n, 1 / FS)
    amp = np.zeros_like(freqs)
    amp[1:] = freqs[1:] ** (-alpha / 2)
    phase = rng.uniform(0, 2 * np.pi, freqs.size)
    spec = amp * np.exp(1j * phase)
    x = np.fft.irfft(spec, n)
    return scale * x / x.std()


def make_subject(subject_id: int) -> tuple[np.ndarray, np.ndarray]:
    n = SECONDS_PER_SUBJECT * FS
    alpha = rng.uniform(0.8, 2.0)  # subject-specific aperiodic exponent
    scale = rng.uniform(20, 60)  # subject-specific background amplitude (uV)
    x = aperiodic_background(n, alpha, scale)
    labels = np.zeros(SECONDS_PER_SUBJECT, dtype=np.int64)
    t = np.arange(n) / FS
    # Events: one per ~4 s on average, 0.6-1.0 s long, low relative power.
    n_events = int(SECONDS_PER_SUBJECT / 4)
    starts = np.sort(rng.uniform(0, SECONDS_PER_SUBJECT - 1.5, n_events))
    for s in starts:
        sec = int(s)
        if labels[sec] != 0 or (sec + 1 < SECONDS_PER_SUBJECT and labels[sec + 1] != 0):
            continue
        cls = rng.integers(1, len(CLASSES))
        name = CLASSES[cls]
        if name == "kcomplex":
            dur = rng.uniform(0.7, 1.0)
            onset = sec + rng.uniform(0, 1 - dur)
            mask = (t >= onset) & (t < onset + dur)
            amp = rng.uniform(1.5, 2.0) * scale
            x[mask] += amp * np.sin(2 * np.pi * (t[mask] - onset) / dur)  # one biphasic cycle
        else:
            lo, hi = EVENT_FREQ[name]
            f = rng.uniform(lo, hi)
            dur = rng.uniform(0.6, 1.0)
            # Event confined to one second so the label is unambiguous.
            onset = sec + rng.uniform(0, 1 - dur)
            mask = (t >= onset) & (t < onset + dur)
            env = np.hanning(mask.sum())
            amp = rng.uniform(0.25, 0.5) * scale  # 6-25 % of background power
            x[mask] += amp * env * np.sin(2 * np.pi * f * t[mask] + rng.uniform(0, 2 * np.pi))
        labels[sec] = cls
    y, _, _ = causal_preprocess_signal(x, fs=FS, modality="eeg")
    return y.reshape(SECONDS_PER_SUBJECT, FS), labels


def log_spectrum(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    taper = np.hanning(FS)
    spec = np.abs(np.fft.rfft(windows * taper, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FS, 1 / FS)
    return np.log(spec + 1e-8), freqs


def band_powers(logspec: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    out = []
    for lo, hi in BANDS.values():
        m = (freqs >= lo) & (freqs < hi)
        out.append(np.log(np.exp(logspec[:, m]).sum(axis=1) + 1e-8))
    return np.stack(out, axis=1)


def standardise(f: np.ndarray, train: np.ndarray) -> np.ndarray:
    mu = f[train].mean(axis=0)
    sd = f[train].std(axis=0) + 1e-8
    return (f - mu) / sd


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information (arithmetic mean normalisation)."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / (0.5 * (ha + hb) + 1e-12))


def fit_codebook(features: np.ndarray, train: np.ndarray, seed: int) -> np.ndarray:
    centroids, _ = kmeans2(features[train].astype(np.float64), CODEBOOK, minit="++", seed=seed, iter=30)
    d = ((features[:, None, :] - centroids[None]) ** 2).sum(-1)
    return d.argmin(1)


def majority_vote_acc(codes: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    table = np.zeros((CODEBOOK, len(CLASSES)))
    np.add.at(table, (codes[train], labels[train]), 1)
    pred = table.argmax(1)[codes[test]]
    return float((pred == labels[test]).mean())


def balanced_acc(codes: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    table = np.zeros((CODEBOOK, len(CLASSES)))
    np.add.at(table, (codes[train], labels[train]), 1)
    # Class-balanced majority vote: normalise columns by class frequency first.
    table = table / (table.sum(0, keepdims=True) + 1e-8)
    pred = table.argmax(1)[codes[test]]
    accs = [(pred[labels[test] == c] == c).mean() for c in range(len(CLASSES))]
    return float(np.mean(accs))


def background_phase_bins(windows: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Phase of the 1 Hz component of each window (a nuisance variable)."""
    spec = np.fft.rfft(windows, axis=1)
    phase = np.angle(spec[:, 1])
    return ((phase + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins


def per_class_recall(codes: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    table = np.zeros((CODEBOOK, len(CLASSES)))
    np.add.at(table, (codes[train], labels[train]), 1)
    table = table / (table.sum(0, keepdims=True) + 1e-8)
    pred = table.argmax(1)[codes[test]]
    return np.array([(pred[labels[test] == c] == c).mean() for c in range(len(CLASSES))])


def main() -> None:
    windows, labels, subjects = [], [], []
    for s in range(N_SUBJECTS):
        w, lab = make_subject(s)
        windows.append(w)
        labels.append(lab)
        subjects.append(np.full(lab.size, s))
    X = np.concatenate(windows)
    y = np.concatenate(labels)
    subj = np.concatenate(subjects)
    train = subj < N_SUBJECTS * 2 // 3
    test = ~train

    phase = background_phase_bins(X)
    logspec, freqs = log_spectrum(X)
    feats = {
        "A. waveform MSE": X,
        "B. flat spectrum": standardise(logspec, train),
        "C. band descriptors": standardise(band_powers(logspec, freqs), train),
    }
    print(f"{N_SUBJECTS} subjects x {SECONDS_PER_SUBJECT} s, {CODEBOOK}-code codebooks, 3 seeds")
    print("class balance:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})
    header = f"{'tokenizer':22s} {'NMI(code,class)':>16s} {'NMI(code,subject)':>18s} {'NMI(code,phase)':>16s} {'bal.acc':>8s}"
    print(header)
    recalls = {}
    for name, f in feats.items():
        rows, rec = [], []
        for seed in range(3):
            codes = fit_codebook(f, train, seed)
            rows.append((nmi(codes, y), nmi(codes, subj), nmi(codes, phase), balanced_acc(codes, y, train, test)))
            rec.append(per_class_recall(codes, y, train, test))
        r = np.array(rows).mean(0)
        recalls[name] = np.array(rec).mean(0)
        print(f"{name:22s} {r[0]:16.3f} {r[1]:18.3f} {r[2]:16.3f} {r[3]:8.3f}")
    print("\nchance balanced accuracy = %.3f" % (1 / len(CLASSES)))
    print("\nper-class recall of the majority-vote decoder (test subjects):")
    print(f"{'tokenizer':22s} " + " ".join(f"{c:>9s}" for c in CLASSES))
    for name, r in recalls.items():
        print(f"{name:22s} " + " ".join(f"{v:9.3f}" for v in r))


if __name__ == "__main__":
    main()
