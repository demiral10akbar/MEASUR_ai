"""Feature extraction for pump condition monitoring (Table A1, predictive maintenance)."""
from __future__ import annotations

import numpy as np


def _amplitude_at(freqs: np.ndarray, amps: np.ndarray, f0: float, band: float) -> float:
    mask = np.abs(freqs - f0) <= band
    return float(amps[mask].max()) if mask.any() else 0.0


def vibration_features(signal, fs_hz: float, rpm: float, band_hz: float = 2.0) -> dict:
    """Features of one vibration window (e.g. 1 s of accelerometer velocity data).

    Returns the RMS value, the spectral amplitudes at 1x and 2x running speed
    (imbalance / misalignment indicators) and the broadband energy above 3x running
    speed (bearing and cavitation indicator).
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 8:
        raise ValueError("window too short")
    window = np.hanning(n)
    spec = np.fft.rfft(x * window)
    amps = 2.0 * np.abs(spec) / window.sum()          # single-sided peak amplitude
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    f1 = rpm / 60.0
    psd = amps ** 2 / 2.0
    return {
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "amp_1x": _amplitude_at(freqs, amps, f1, band_hz),
        "amp_2x": _amplitude_at(freqs, amps, 2 * f1, band_hz),
        "broadband_energy": float(psd[freqs > 3 * f1].sum()),
    }
