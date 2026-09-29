"""Offline tests for NSRR preprocessing on a synthetic SHHS-style EDF + NSRR XML."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyedflib

from hypnos.data.nsrr import (
    EPOCH_SEC,
    SPO2_VOCAB,
    detect_r_peaks,
    parse_nsrr_xml,
    preprocess_record,
    rr_descriptors,
    run_batch,
    spo2_tokens,
    to_microvolts,
)

DURATION = 300  # seconds

XML = """<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
<EpochLength>30</EpochLength>
<ScoredEvents>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept>
<Start>0</Start><Duration>60.0</Duration></ScoredEvent>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 2 sleep|2</EventConcept>
<Start>60</Start><Duration>120.0</Duration></ScoredEvent>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 4 sleep|4</EventConcept>
<Start>180</Start><Duration>60.0</Duration></ScoredEvent>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>REM sleep|5</EventConcept>
<Start>240</Start><Duration>30.0</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType>
<EventConcept>Obstructive apnea|Obstructive Apnea</EventConcept><Start>100.5</Start><Duration>20.0</Duration>
</ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>Hypopnea|Hypopnea</EventConcept>
<Start>150</Start><Duration>15</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>SpO2 artifact|SpO2 artifact</EventConcept>
<Start>200</Start><Duration>10</Duration></ScoredEvent>
<ScoredEvent><EventType>Arousals|Arousals</EventType><EventConcept>ASDA arousal|Arousal (ASDA)</EventConcept>
<Start>121</Start><Duration>5</Duration></ScoredEvent>
</ScoredEvents>
</PSGAnnotation>
"""


def synthetic_ecg(fs, duration, rng):
    """Gaussian QRS-like spikes at jittered beat times, plus baseline wander and noise (uV)."""
    rr = np.clip(
        0.9 + 0.1 * np.sin(np.arange(int(duration * 1.5)) / 5) + rng.normal(0, 0.02, int(duration * 1.5)), 0.6, 1.3
    )
    beats = np.cumsum(rr)
    beats = beats[beats < duration - 1]
    t = np.arange(int(fs * duration)) / fs
    ecg = np.zeros_like(t)
    half = int(0.1 * fs)
    for b in beats:  # add each complex locally (O(beats), not O(beats * samples))
        c = int(round(b * fs))
        sl = slice(max(c - half, 0), min(c + half, t.size))
        tt = t[sl] - b
        ecg[sl] += 1000 * np.exp(-0.5 * (tt / 0.012) ** 2) - 150 * np.exp(-0.5 * ((tt - 0.03) / 0.015) ** 2)
    ecg += 200 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 20, t.size)
    return ecg, beats


def write_edf(path, gain=1.0, seed=0):
    """SHHS-style labels: EEG(sec)=C3, EEG=C4, EOG(L/R), EMG, ECG, THOR/ABDO RES, SaO2, AIRFLOW."""
    rng = np.random.default_rng(seed)
    t256 = np.arange(256 * DURATION) / 256
    t32 = np.arange(32 * DURATION) / 32
    ecg, beats = synthetic_ecg(256, DURATION, rng)
    breath = np.sin(2 * np.pi * 0.25 * t32)
    spo2 = np.full(DURATION, 96.0)
    spo2[100:130] = np.linspace(96, 88, 30)
    spo2[205] = 0.0  # probe dropout
    channels = [
        ("EEG(sec)", 256, "uV", gain * (40 * np.sin(2 * np.pi * 1.5 * t256) + rng.normal(0, 10, t256.size)), 500),
        ("EEG", 256, "uV", gain * (40 * np.sin(2 * np.pi * 10 * t256) + rng.normal(0, 10, t256.size)), 500),
        ("EOG(L)", 256, "uV", rng.normal(0, 30, t256.size), 500),
        ("EOG(R)", 256, "uV", rng.normal(0, 30, t256.size), 500),
        ("EMG", 256, "uV", rng.normal(0, 5, t256.size), 500),
        ("ECG", 256, "mV", ecg / 1000.0, 5),  # mV on purpose: exercises unit conversion
        ("THOR RES", 32, "", breath + rng.normal(0, 0.02, t32.size), 5),
        ("ABDO RES", 32, "", 0.8 * breath + rng.normal(0, 0.02, t32.size), 5),
        ("AIRFLOW", 32, "", breath + rng.normal(0, 0.02, t32.size), 5),
        ("SaO2", 1, "%", spo2, None),
    ]
    headers, data = [], []
    for label, fs, unit, sig, rng_abs in channels:
        pmin, pmax = (0.0, 100.0) if rng_abs is None else (-rng_abs, rng_abs)
        headers.append(
            {
                "label": label,
                "dimension": unit,
                "sample_frequency": fs,
                "physical_min": pmin,
                "physical_max": pmax,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
        )
        data.append(np.clip(sig, pmin, pmax).astype(np.float64))
    w = pyedflib.EdfWriter(str(path), len(channels))
    w.setSignalHeaders(headers)
    w.writeSamples(data)
    w.close()
    return beats


def test_spo2_tokens_and_units():
    tok = spo2_tokens(np.array([np.nan, 55.0, 60.0, 95.4, 100.0]))
    assert tok.tolist() == [0, 1, 2, 37, 42]
    assert tok.max() < SPO2_VOCAB
    x, ok = to_microvolts(np.array([1.0]), "mV")
    assert ok and x[0] == 1000.0
    _, ok = to_microvolts(np.array([1.0]), "a.u.")
    assert not ok


def test_r_peaks_and_rr():
    rng = np.random.default_rng(1)
    ecg, beats = synthetic_ecg(256, 120, rng)
    r = detect_r_peaks(ecg, 256) / 256
    matched = np.min(np.abs(r[:, None] - beats[None, :]), axis=0)
    assert np.mean(matched < 0.01) > 0.98, np.mean(matched < 0.01)
    rr_sec, n_beats, rmssd, valid = rr_descriptors(r, 120)
    assert np.isfinite(rr_sec).mean() > 0.95
    assert abs(np.nanmedian(rr_sec) - np.median(np.diff(beats))) < 0.05
    assert rmssd.shape == (120 // EPOCH_SEC,) and np.isfinite(rmssd).all()
    assert valid.mean() > 0.95


def test_parse_nsrr_xml():
    with tempfile.TemporaryDirectory() as d:
        xml = Path(d) / "rec-nsrr.xml"
        xml.write_text(XML)
        ann = parse_nsrr_xml(xml, DURATION)
    assert ann.stages.tolist() == [0, 0, 2, 2, 2, 2, 3, 3, 4, -1]  # stage 4 -> N3
    assert ann.masks["obstructive_apnea"][100:121].all() and not ann.masks["obstructive_apnea"][121]
    assert ann.masks["hypopnea"][150:165].sum() == 15
    assert ann.masks["arousal"][121:126].all()
    assert {e[0] for e in ann.events} == {"obstructive_apnea", "hypopnea", "spo2_artifact", "arousal"}


def test_preprocess_record_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        edf = d / "shhs1-200001.edf"
        beats = write_edf(edf)
        (d / "shhs1-200001-nsrr.xml").write_text(XML)
        out = preprocess_record(edf, d / "shhs1-200001-nsrr.xml")

        meta = json.loads(str(out["meta"]))
        assert meta["n_seconds"] == DURATION
        expected = {
            "eeg_c3",
            "eeg_c4",
            "eog_e1",
            "eog_e2",
            "emg_chin",
            "ecg",
            "resp_abd",
            "resp_thx",
            "airflow",
            "spo2",
        }
        assert set(meta["modalities"]) == expected, set(meta["modalities"])
        assert meta["modalities"]["ecg"]["microvolts"] is True

        assert out["eeg_c3.signal"].shape == (128 * DURATION,) and out["eeg_c3.signal"].dtype == np.float16
        assert out["resp_abd.signal"].shape == (32 * DURATION,)
        for m in expected - {"spo2"}:
            assert out[f"{m}.log_scale"].shape == (DURATION,) and np.isfinite(out[f"{m}.log_scale"]).all()
            assert out[f"{m}.qc"].shape == (DURATION,)
        assert out["eeg_c3.band_logpow"].shape == (DURATION, 5)
        # C4 carries 10 Hz alpha, C3 1.5 Hz delta: descriptors should reflect that.
        c3 = np.nanmean(out["eeg_c3.band_logpow"][60:], axis=0)
        c4 = np.nanmean(out["eeg_c4.band_logpow"][60:], axis=0)
        assert c3[0] > c4[0] and c4[2] > c3[2]

        r = out["ecg.r_peaks_sec"]
        assert abs(len(r) - len(beats)) <= 2
        assert out["ecg.rr_sec"].shape == (DURATION,) and out["ecg.beats"].shape == (DURATION,)

        tok = out["spo2.tokens"]
        assert tok[50] == 96 - 60 + 2
        assert tok[205] == 0  # dropout (0 %) -> invalid
        assert (tok[200:210] == 0).all()  # scorer-marked SpO2 artifact -> invalid
        assert out["spo2.qc"][205] != 0

        assert out["stages"].shape == (DURATION // EPOCH_SEC,)
        assert out["event.obstructive_apnea"].sum() == 21

        # Normalised waveforms are gain-invariant; the scale channel keeps the gain.
        edf2 = d / "gain.edf"
        write_edf(edf2, gain=0.5)
        out2 = preprocess_record(edf2)
        a = out["eeg_c4.signal"].astype(np.float32)
        b = out2["eeg_c4.signal"].astype(np.float32)
        assert np.max(np.abs(a - b)) < 0.05
        diff = out["eeg_c4.log_scale"] - out2["eeg_c4.log_scale"]
        assert abs(np.median(diff) - np.log(2.0)) < 0.02, np.median(diff)


def test_run_batch_manifest():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "edfs").mkdir()
        (d / "xml").mkdir()
        write_edf(d / "edfs" / "rec1.edf")
        (d / "xml" / "rec1-nsrr.xml").write_text(XML)
        (d / "edfs" / "broken.edf").write_bytes(b"not an edf")
        manifest = run_batch(d / "edfs", d / "xml", d / "out", modalities=["eeg_c3", "ecg", "spo2"])
        import csv

        with open(manifest) as fh:
            rows = {r["record"]: r for r in csv.DictReader(fh)}
        assert set(rows) == {"rec1", "broken"}
        assert rows["rec1"]["status"] == "ok" and rows["rec1"]["n_seconds"] == str(DURATION)
        assert rows["broken"]["status"].startswith("error")
        assert (d / "out" / "rec1.npz").exists()
