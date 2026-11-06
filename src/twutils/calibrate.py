"""Mass calibration of a run, following its drift with the retention time.

Method of the legacy alpinac (``alpinac.mass_calibration``), with the peak
detection and fits of peaxart:

1. the run is split in rt windows of ``averaging`` s, every ``spacing`` s;
2. for each calibrant (an ion present during the whole run: PFTBA, air,
   filament, ...) and window, the spectra are summed over the window, in a
   domain of +- ``mz_range`` around the calibrant (initial calibration);
3. the peaks of the summed spectrum are detected and fitted with
   pseudo-Voigt functions of free sigma and alpha; the calibrant is the
   peak closest to its predicted position;
4. the calibration of each window is fitted to the calibrants, removing the
   worst one while its residual is above ``max_ppm`` (see :func:`fit_calibration`);
5. steps 3 and 4 are repeated with the calibration of step 4 (the initial
   calibration can be off by a few tof indices);
6. the peak shape (sigma vs tof index, alpha) and the calibration
   uncertainty of each calibrant m/z are computed from the calibrants used.

Calibrants injected at given rts (spikes) are added to the closest window.
The calibrants are formulas (m/z from :func:`alpinac.formula.formula_mass`)
or m/z values, given directly or in the yaml file of the tof-processing
pipeline (:meth:`Calibrants.from_yaml`).

Differences with the legacy code: the calibrants are ions (the legacy code
used the neutral masses: ``ionised=False``); the calibration is fitted before
the outlier test (the legacy code tested the initial guess first); the
outliers are also removed by leverage (one wrong low-mass peak otherwise
removes the good calibrants); the uncertainty uses the absolute bias; a
window whose fit fails is left out (the legacy file then shifted the rts of
the next windows); the run is read once (in blocks) for all calibrants.

Masses below the lowest calibrant are extrapolated: with the PFTBA ions only
(m/z >= 69), the error at m/z 20-40 is tens of ppm. Low-mass ions (air: N2,
O2, Ar, CO2; F) are needed for the full mass range (see the calibrants of
``/shared/scripts/tof-processing/config_calib.yml``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import peaxart
import yaml
from peaxart.leastsq import fit
from peaxart.mzfit import EMPTY, ScanPeaks, detect_mass_peaks, noise_median
from peaxart.shapes import C_PV, SQRT_2PI, pseudo_voigt_height
from scipy.optimize import least_squares

from twutils.calibration import (
    N_PARAMS,
    MassCalibration,
    mass_to_tof,
    mc_path,
    tof_to_mass,
)
from twutils.h5 import TofFile

logger = logging.getLogger(__name__)

Calibrant = str | float | tuple[str, float]
"""A formula (``"CF3"``, ``"Re+2"``, ``"[13C]O"``), an m/z, or (name, m/z)."""


@dataclass
class Calibrants:
    """Calibrants of a run: present all along (``continuous``) and injected at rts (``spikes``)."""

    continuous: list[Calibrant] = field(default_factory=list)
    spikes: dict[float, list[Calibrant]] = field(default_factory=dict)
    initial: MassCalibration | None = None
    """Calibration used to find the calibrants (``default`` of the yaml file)."""
    source: str = ""

    @classmethod
    def from_yaml(cls, path: str | PathLike) -> Calibrants:
        """The calibration file of the tof-processing pipeline (legacy ``dict_spikes``)::

            default: {mode: 2, p1: 2829.1, p2: -3731.25, p3: 0.49994}  # optional
            "0.0": [F, N2, "[13C]O", CF3, ...]   # continuous
            "1200": [C4F9]                       # spike at 1200 s
        """
        path = Path(path)
        content = yaml.safe_load(path.read_text()) or {}
        initial = None
        if "default" in content:
            d = content.pop("default")
            mode = int(d["mode"])
            initial = MassCalibration(mode=mode, rt=[0.0], params=[[float(d[f"p{k}"]) for k in
                                                                    range(1, N_PARAMS[mode] + 1)]],
                                      source=f"{path} (default)")
        # keys: rt of the spikes, 0 (written 0, 0.0, "0" or "0.0") for the continuous calibrants
        continuous = [c for rt, cals in content.items() if float(rt) == 0.0 for c in cals]
        spikes = {float(rt): list(cals) for rt, cals in content.items() if float(rt) != 0.0}
        return cls(continuous=continuous, spikes=spikes, initial=initial, source=str(path))


DETECTION = peaxart.ExtractionConfig(mz_slope_threshold=9.0, mz_noise_k=None)
"""Peak detection in the summed spectra (legacy ``PTtof_mass_cal``, no noise test)."""
PEAK_WIDTH = 3.0
"""Peak width of the detection, tof indices (legacy ``WTtofidx``)."""
EXTRAPOLATION_WARNING = 50.0
"""A warning is logged if no calibrant below this m/z is used."""


@dataclass
class CalibrationConfig:
    """Parameters of :func:`calibrate` (the run-specific inputs are arguments of :func:`calibrate`)."""

    #: TOFWERK calibration mode (see :mod:`twutils.calibration`): 0, 1, 2 or 5
    mode: int = 2
    #: duration of the rt windows, s
    averaging: float = 360.0
    #: interval between the rt windows, s
    spacing: float = 120.0
    #: half width of the domain of each calibrant, m/z
    mz_range: float = 0.2
    #: calibrants with a larger residual are left out, ppm
    max_ppm: float = 20.0
    #: m/z of the formulas without the electrons (False: neutral masses, legacy)
    ionised: bool = True
    #: duration over which a spike is summed, s
    spike_duration: float = 5.0
    #: (intercept, slope) of the starting sigma vs tof index
    sigma_guess: tuple[float, float] = (0.36456746757561603, 4.8076576034728846e-5)
    #: starting alpha (Lorentzian fraction)
    alpha_guess: float = 0.6
    #: number of scans read at once
    block: int = 600

    def __post_init__(self):
        if self.mode not in N_PARAMS or self.mode in (3, 4):  # 3, 4: no m/z of tof indices (twutils.calibration)
            raise NotImplementedError(f"Mass calibration mode {self.mode}")


@dataclass
class CalibrationResult:
    """Result of :func:`calibrate`."""

    calibration: MassCalibration
    peaks: pd.DataFrame
    """One row per fitted peak: ``calibrant, mz_exact, window, rt, tof, area,
    sigma, alpha, baseline, selected`` (the peak taken as the calibrant),
    ``ppm`` (residual of the calibration), ``used`` (in the fit of its window)."""
    windows: pd.DataFrame
    """One row per rt window: ``rt_start, rt_stop, rt, n_calibrants, n_used,
    rms_ppm, max_ppm, ok`` (fitted) and the parameters ``p1, p2, ...``."""
    file: TofFile
    config: CalibrationConfig
    runtime: float

    def write(self, path: str | PathLike | None = None, peaks_csv: bool = False,
              output_dir: str | PathLike | None = None) -> Path:
        """Write the calibration file (``*_mc.json``) and optionally the peaks (``.peaks.csv``).

        Default path: :func:`twutils.calibration.mc_path`, where :meth:`MassCalibration.for_file`
        finds it; or its file name in ``output_dir``.
        """
        path = self.default_path(output_dir) if path is None else Path(path)
        self.calibration.to_json(path)
        if peaks_csv:
            self.peaks.to_csv(path.with_suffix(".peaks.csv"), index=False)
        return path

    def default_path(self, output_dir: str | PathLike | None = None) -> Path:
        """``<run>.seg0_mc.json`` file: :func:`twutils.calibration.mc_path` (of the segments calibrated),
        or its file name in ``output_dir``."""
        path = mc_path(self.file, self.calibration.segments)
        return path if output_dir is None else Path(output_dir) / path.name

    def plot(self, path: str | PathLike | None = None, plots=None, show: bool = False,
             write: bool = True) -> dict:
        """Calibration plots (see :mod:`twutils.plots`, needs matplotlib), written next to the
        calibration file ``path`` (default as :meth:`write`): ``<run>.seg0_mc.<plot>.png``.

        :return: {name: png path}, or {name: figure} if not ``write``.
        """
        from twutils.plots import PLOTS, plot_calibration

        path = self.default_path() if path is None else Path(path)
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
        return plot_calibration(self.calibration, self.peaks, path if write else None, plots or PLOTS, show,
                                title=self.file.stem)

    def summary(self) -> str:
        w = self.windows[self.windows["ok"]]
        cal = self.calibration
        masses = [50.0, 150.0, 250.0]
        drift = cal.drift_ppm(np.array(masses)[:, None], cal.rt[None, :])
        rms = np.sqrt(np.mean(w["rms_ppm"] ** 2))
        used = self.peaks.loc[self.peaks["used"], "mz_exact"]
        return "\n".join([
            f"{self.file.path.name}: mode {cal.mode}, {len(w)}/{len(self.windows)} windows, "
            + f"{len(used)} calibrant peaks (m/z {used.min():.1f} - {used.max():.1f}), rms {rms:.2f} ppm, "
            + f"{self.runtime:.1f} s",
            "  drift over the run (ppm): "
            + ", ".join(f"m/z {m:.0f}: {np.ptp(d):.1f}" for m, d in zip(masses, drift, strict=True)),
            f"  peak shape: sigma = {cal.sigma_intercept:.4g} + {cal.sigma_slope:.4g} x tof, alpha {cal.alpha:.3f}",
        ])


# --- calibrants and windows -----------------------------------------------


def h5_calibrants(file: TofFile) -> list[tuple[str, float]]:
    """Calibrants of the acquisition calibration (h5 attributes), as (name, m/z)."""
    attrs = file.attrs("FullSpectra")
    n = int(attrs.get("MassCalibration nbrPoints", 0))
    out = []
    for k in range(1, n + 1):
        mz = float(attrs[f"MassCalibration m{k}"])
        out.append((str(attrs.get(f"MassCalibration l{k}", "")) or f"{mz:.4f}", mz))
    return out


def formula_mz(formula: str, ionised: bool = True) -> float:
    """m/z of a formula (:func:`alpinac.formula.formula_mass`).

    A charge (``Re+2``) removes its electrons; without one, the ion is singly
    charged, or neutral with ``ionised=False`` (legacy calibration).
    """
    try:
        from alpinac.formula import formula_mass
    except ImportError:
        raise ImportError("The m/z of formulas needs the alpinac package (gitlabs/alpinac, not the legacy "
                          "alpinac of halosearch); give the calibrants as m/z values otherwise.") from None
    return formula_mass(formula, ionised=ionised)


def resolve_calibrants(calibrants: Sequence[Calibrant], ionised: bool = True) -> list[tuple[str, float]]:
    """(name, m/z) of the calibrants (formulas through :func:`formula_mz`)."""
    out = []
    for c in calibrants:
        if isinstance(c, tuple):
            out.append((str(c[0]), float(c[1])))
        elif isinstance(c, str):
            out.append((c, formula_mz(c, ionised)))
        else:
            out.append((f"{float(c):.4f}", float(c)))
    return out


def calibration_windows(time_axis: np.ndarray, averaging: float = 360.0, spacing: float = 120.0,
                        rt_start: float | None = None, rt_stop: float | None = None) -> pd.DataFrame:
    """rt windows of ``averaging`` s, starting every ~``spacing`` s, covering [rt_start, rt_stop].

    :return: ``rt_start, rt_stop, rt`` (centre), ``scan_start, scan_stop`` (scans [start, stop[).
    """
    t = np.asarray(time_axis, dtype=float)
    t0 = float(t[0]) if rt_start is None else max(float(rt_start), float(t[0]))
    t1 = float(t[-1]) if rt_stop is None else min(float(rt_stop), float(t[-1]))
    if t1 <= t0:
        raise ValueError(f"Empty rt range {t0} - {t1} s")
    averaging = min(averaging, t1 - t0)
    n = max(1, round((t1 - t0 - averaging) / spacing) + 1)
    starts = np.linspace(t0, t1 - averaging, n)
    stops = starts + averaging
    windows = pd.DataFrame({"rt_start": starts, "rt_stop": stops, "rt": (starts + stops) / 2})
    windows["scan_start"] = np.searchsorted(t, starts, side="left")
    windows["scan_stop"] = np.searchsorted(t, stops, side="right")
    return windows


# --- peaks of the summed spectra ------------------------------------------


def _sum_spectra(file: TofFile, ranges: np.ndarray, columns: np.ndarray, segments: tuple[int, int] | None,
                 block: int) -> np.ndarray:
    """Spectra summed over the scan ranges (n, 2), at the tof indices ``columns``.

    The run is read once, in blocks of ``block`` scans (a whole run does not fit in memory).
    """
    sums = np.zeros((len(ranges), len(columns)))
    tof0, tof1 = int(columns.min()), int(columns.max()) + 1
    first, last = int(ranges[:, 0].min()), int(ranges[:, 1].max())
    for scans, _, data in file.iter_blocks(block, tof0, tof1, segments, slice(first, last)):
        b0, b1 = scans.start, scans.stop
        data = data[:, columns - tof0]
        for k, (s0, s1) in enumerate(ranges):
            lo, hi = max(s0, b0), min(s1, b1)
            if hi > lo:
                sums[k] += data[lo - b0:hi - b0].sum(axis=0, dtype=float)
    return sums


def fit_peak_shapes(x: np.ndarray, y: np.ndarray, found: np.ndarray, sigma_guess: np.ndarray,
                    alpha_guess: float = 0.6, height_threshold: float = 1.3,
                    max_iterations: int | None = 2000) -> ScanPeaks:
    """Fit pseudo-Voigt peaks of free position, area, sigma and alpha and a baseline.

    Legacy ``fit_pseudoVoigt(mode="MassCal")``: sigma in [guess - 0.4, min(3, 2
    guess)], alpha in [0.2, 1]; peaks whose fitted height is below
    ``height_threshold`` x the noise median are removed and the fit is redone.
    Legacy Gaussian convention (as the sigma used by the extraction).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    median = noise_median(y, 1 / 3)
    baseline = max(0.0, median)
    baseline_lo, baseline_hi = max(0.0, -2 * y.min()), max(3.0, 3 * baseline, 3 * median)
    mu, sigma = found[:, 0], np.asarray(sigma_guess, dtype=float)
    alpha = np.full(len(mu), alpha_guess)
    area = (found[:, 1] - baseline) / pseudo_voigt_height(1.0, sigma, alpha)
    residual, jac = _pseudo_voigt_model(x, y)

    while True:
        # legacy bounds: sigma <= 3 tof indices caps the width of the peaks above m/z ~440
        # (guess ~3.25 at tof 60000, PFTBA 502), which biases the sigma vs tof regression there
        sigma_lo = np.maximum(0.05, sigma - 0.4)
        sigma_hi = np.maximum(np.minimum(3.0, 2 * sigma), sigma_lo + 0.01)
        ones = np.ones(len(mu))
        p0 = np.r_[baseline, np.c_[mu, np.maximum(1.0, area), sigma, alpha].ravel()]
        lo = np.r_[baseline_lo, np.c_[mu - 2.0, np.maximum(1.0, 0.1 * area), sigma_lo, 0.2 * ones].ravel()]
        hi = np.r_[baseline_hi, np.c_[mu + 2.0, np.maximum(2.0, 10 * area), sigma_hi, ones].ravel()]

        res = fit(residual, p0, lo, hi, jac=jac, exact=False, max_iterations=max_iterations)
        baseline = float(res.params[0])
        mu, area, sigma, alpha = res.params[1:].reshape(-1, 4).T
        keep = baseline + pseudo_voigt_height(area, sigma, alpha) >= height_threshold * median
        if keep.all():
            return ScanPeaks(mu, area, sigma, alpha, baseline)
        if not keep.any():
            return EMPTY
        mu, area, sigma, alpha = mu[keep], area[keep], sigma[keep], alpha[keep]


def _pseudo_voigt_model(x: np.ndarray, y: np.ndarray):
    """Residual and analytic Jacobian of a baseline + pseudo-Voigt peaks (legacy convention).

    Parameters: baseline, then (mu, area, sigma, alpha) for each peak.
    """

    def shapes(p):
        mu, area, sigma, alpha = (v[:, None] for v in p[1:].reshape(-1, 4).T)
        d = x[None, :] - mu
        k = 2 * sigma**2  # legacy Gaussian: exp(-d^2 / (2 k))
        gauss = np.exp(-(d**2) / (2 * k)) / (sigma * SQRT_2PI)
        gamma = sigma * C_PV
        den = d**2 + gamma**2
        lorentz = gamma / np.pi / den
        return mu, area, sigma, alpha, d, k, gauss, gamma, den, lorentz

    def residual(p):
        _, area, _, alpha, _, _, gauss, _, _, lorentz = shapes(p)
        return p[0] + (area * ((1 - alpha) * gauss + alpha * lorentz)).sum(axis=0) - y

    def jac(p):
        _, area, sigma, alpha, d, k, gauss, gamma, den, lorentz = shapes(p)
        J = np.empty((len(x), len(p)))
        J[:, 0] = 1.0
        J[:, 1::4] = (area * ((1 - alpha) * gauss * d / k + alpha * lorentz * 2 * d / den)).T
        J[:, 2::4] = ((1 - alpha) * gauss + alpha * lorentz).T
        dgauss = gauss * (d**2 / (k * sigma) - 1 / sigma)
        dlorentz = C_PV / np.pi * (d**2 - gamma**2) / den**2
        J[:, 3::4] = (area * ((1 - alpha) * dgauss + alpha * dlorentz)).T
        J[:, 4::4] = (area * (lorentz - gauss)).T
        return J

    return residual, jac


# --- calibration fit -------------------------------------------------------


def _bounds(mode: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    if mode == 2:
        lo[2], hi[2] = 0.498, 0.502  # legacy
    return lo, hi


def initial_params(mode: int, tof: np.ndarray, mz: np.ndarray, guess: MassCalibration | None = None,
                   rt: float = 0.0) -> np.ndarray:
    """Starting parameters: those of ``guess`` (same mode), else a fit of tof vs sqrt(m/z)."""
    if guess is not None and guess.mode == mode:
        return np.array([float(v) for v in guess.params_at(rt).values()])
    if mode == 5:
        return np.polyfit(tof, mz, 2)
    p1, p2 = np.polyfit(1 / np.sqrt(mz) if mode == 1 else np.sqrt(mz), tof, 1)
    extra = [0.5] if mode == 2 else [0.0] * (N_PARAMS[mode] - 2)
    return np.array([p1, p2, *extra])


def fit_calibration(tof: np.ndarray, mz: np.ndarray, mode: int, p0: np.ndarray,
                    max_ppm: float = 20.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the calibration parameters to calibrant peaks (tof index, exact m/z).

    The residuals are in ppm of m/z (to first order). While a residual is
    above ``max_ppm``, the worst point is removed and the fit is redone,
    keeping at least n_params + 2 points. The worst point is the one of
    largest residual (legacy), or of largest deleted residual (residual / (1 -
    leverage)): a wrong peak at the end of the mass range pulls the fit to
    itself, and the largest residuals are then those of good points. Both are
    done; the fit that uses more points is kept (the legacy one if equal).
    For mode 4, p5 (redundant with p2) stays at its start value.

    :return: parameters, residuals (ppm, all points), mask of the points used.
    """
    tof, mz = np.asarray(tof, dtype=float), np.asarray(mz, dtype=float)
    n = N_PARAMS[mode]
    lo, hi = _bounds(mode, n)
    p0 = np.clip(np.asarray(p0, dtype=float), lo, hi)
    free = np.arange(n) if mode != 4 else np.arange(4)

    def as_dict(values):
        return {f"p{k + 1}": v for k, v in enumerate(values)}

    # m dtof/dm: converts tof residuals into relative m/z residuals
    h = 1e-6
    scale = (mass_to_tof(mz * (1 + h), as_dict(p0), mode) - mass_to_tof(mz * (1 - h), as_dict(p0), mode)) / (2 * h)

    def remove_outliers(by_leverage: bool):
        p = p0.copy()
        used = np.ones(len(tof), dtype=bool)
        while True:
            def residual(q, used=used):
                full = p.copy()
                full[free] = q
                return (tof[used] - mass_to_tof(mz[used], as_dict(full), mode)) / scale[used] * 1e6

            res = least_squares(residual, p[free], bounds=(lo[free], hi[free]), x_scale="jac",
                                ftol=1e-12, xtol=1e-12, gtol=1e-12)
            p[free] = res.x
            ppm = (tof_to_mass(tof, as_dict(p), mode) - mz) / mz * 1e6
            if np.abs(ppm[used]).max() <= max_ppm or used.sum() <= n + 2:
                return p, ppm, used
            worst = np.abs(ppm[used])
            if by_leverage:
                u, s, _ = np.linalg.svd(res.jac, full_matrices=False)
                worst = worst / np.maximum(1 - (u[:, s > s[0] * 1e-10] ** 2).sum(axis=1), 1e-6)
            used[np.flatnonzero(used)[np.argmax(worst)]] = False

    legacy, leverage = remove_outliers(False), remove_outliers(True)
    return leverage if leverage[2].sum() > legacy[2].sum() else legacy


def fit_windows(peaks: pd.DataFrame, windows: pd.DataFrame, mode: int, guess: MassCalibration,
                max_ppm: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the calibrant peaks with ``guess`` and fit the calibration of each window.

    Sets the ``selected``, ``ppm`` and ``used`` columns of ``peaks``.

    :return: the window table (see :attr:`CalibrationResult.windows`) and ``peaks``.
    """
    n = N_PARAMS[mode]
    peaks = peaks.assign(selected=_closest(peaks, guess), ppm=np.nan, used=False)
    stats = []
    for w, rt in enumerate(windows["rt"]):
        sel = peaks.index[peaks["selected"] & (peaks["window"] == w)]
        if len(sel) < n + 2:
            stats.append([len(sel), 0, np.nan, np.nan, False] + [np.nan] * n)
            continue
        tof, mz = peaks.loc[sel, "tof"].to_numpy(), peaks.loc[sel, "mz_exact"].to_numpy()
        params, ppm, used = fit_calibration(tof, mz, mode, initial_params(mode, tof, mz, guess, rt), max_ppm)
        peaks.loc[sel, "ppm"] = ppm
        peaks.loc[sel, "used"] = used
        r = np.abs(ppm[used])
        stats.append([len(sel), int(used.sum()), float(np.sqrt(np.mean(r**2))), float(r.max()), r.max() <= max_ppm,
                      *params])
    table = pd.DataFrame(stats, columns=["n_calibrants", "n_used", "rms_ppm", "max_ppm", "ok"]
                         + [f"p{k + 1}" for k in range(n)])
    table = pd.concat([windows[["rt_start", "rt_stop", "rt"]].reset_index(drop=True), table], axis=1)
    table["ok"] = table["ok"].astype(bool)
    peaks.loc[peaks["window"].isin(np.flatnonzero(~table["ok"])), "used"] = False
    return table, peaks


def _closest(peaks: pd.DataFrame, cal: MassCalibration) -> np.ndarray:
    """Mask of the peaks closest to the position of their calibrant predicted by ``cal``."""
    distance = pd.Series(np.abs(peaks["tof"] - cal.tof(peaks["rt"], peaks["mz_exact"])), index=peaks.index)
    return peaks.index.isin(distance.groupby(peaks["measurement"]).idxmin().to_numpy())


def _calibration_of(windows: pd.DataFrame, mode: int) -> MassCalibration:
    ok = windows[windows["ok"]]
    return MassCalibration(mode=mode, rt=ok["rt"], params=ok[[f"p{k + 1}" for k in range(N_PARAMS[mode])]])


def peak_shape(peaks: pd.DataFrame, cal: MassCalibration) -> dict:
    """Peak shape of the calibrant peaks used: sigma vs tof index and vs m/z (linear), mean alpha."""
    sigma_slope, sigma_b = np.polyfit(peaks["tof"], peaks["sigma"], 1)
    sigma_mz = cal.mz(peaks["rt"], peaks["tof"] + peaks["sigma"]) - cal.mz(peaks["rt"], peaks["tof"])
    sigma_mz_slope, sigma_mz_b = np.polyfit(peaks["mz_exact"], sigma_mz, 1)
    return {"sigma_slope": float(sigma_slope), "sigma_intercept": float(sigma_b),
            "sigma_mz_slope": float(sigma_mz_slope), "sigma_mz_intercept": float(sigma_mz_b),
            "alpha": float(peaks["alpha"].mean())}


def calibration_uncertainty(peaks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Uncertainty (ppm) of each calibrant m/z: max(|mean| / 2 + std, 1.1 max|residual| / 2).

    Legacy rule (bias + 2 sigma = 2 u), with the absolute bias. Calibrants
    used in fewer than 2 windows are left out.
    """
    g = peaks.groupby("mz_exact")["ppm"]
    u = np.maximum(g.mean().abs() / 2 + g.std(), g.agg(lambda v: np.abs(v).max()) * 1.1 / 2)
    u = u[g.count() >= 2]
    return u.index.to_numpy(dtype=float), u.to_numpy(dtype=float)


# --- the calibration of a run -------------------------------------------------


def _measurements(file: TofFile, continuous: list[tuple[str, float]], spiked: dict[float, list[tuple[str, float]]],
                  windows: pd.DataFrame | None, spike_duration: float):
    """The summed spectra to fit: calibrants (name, m/z) and measurements (calibrant, window, rt, scans).

    A spike is fitted with the window closest to it; its peaks keep the rt of the spike (the drift
    of the calibration between the spike and the centre of the window is neglected in the fit).
    """
    calibrants, rows = [], []
    if windows is None:  # spikes only: one window per spike
        rts = np.array(sorted(spiked))
        windows = pd.DataFrame({"rt_start": rts - spike_duration / 2, "rt_stop": rts + spike_duration / 2, "rt": rts})
    for name, mz in continuous:
        calibrants.append((name, mz))
        rows += [(len(calibrants) - 1, w, rt, s0, s1)
                 for w, rt, s0, s1 in windows[["rt", "scan_start", "scan_stop"]].itertuples()]
    for rt, cals in spiked.items():
        w = int(np.argmin(np.abs(windows["rt"].to_numpy() - rt)))
        scans = file.scan_range(rt - spike_duration / 2, rt + spike_duration / 2)
        if scans.stop <= scans.start:
            logger.warning("%s: spike at %g s outside the run: left out", file.path.name, rt)
            continue
        for name, mz in cals:
            calibrants.append((name, mz))
            rows.append((len(calibrants) - 1, w, rt, scans.start, scans.stop))
    columns = ["calibrant", "window", "rt", "scan_start", "scan_stop"]
    return calibrants, pd.DataFrame(rows, columns=columns), windows


def find_calibrant_peaks(file: TofFile, calibrants: list[tuple[str, float]], measurements: pd.DataFrame,
                         windows: pd.DataFrame, initial: MassCalibration, config: CalibrationConfig | None = None,
                         segments: tuple[int, int] | None = None) -> pd.DataFrame:
    """Sum the spectra of each measurement in the domain of its calibrant, detect and fit the peaks.

    :return: one row per peak: ``measurement, calibrant, mz_exact, window, rt, tof, area, sigma, alpha, baseline``.
    """
    config = config or CalibrationConfig()
    mz_range, sigma_guess = config.mz_range, config.sigma_guess
    names = np.array([c[0] for c in calibrants])
    mzs = np.array([c[1] for c in calibrants])
    rts = windows["rt"].to_numpy()[:, None]
    tof_lo = np.floor(np.nanmin(initial.tof(rts, mzs - mz_range), axis=0)).astype(int)
    tof_hi = np.ceil(np.nanmax(initial.tof(rts, mzs + mz_range), axis=0)).astype(int) + 1
    inside = (tof_lo >= 0) & (tof_hi <= file.n_points)
    for k in np.flatnonzero(~inside):
        logger.warning("Calibrant %s (m/z %.4f) is outside the spectra: left out", names[k], mzs[k])
    measurements = measurements[inside[measurements["calibrant"]]]
    columns = np.unique(np.concatenate([np.arange(tof_lo[k], tof_hi[k]) for k in np.flatnonzero(inside)]))
    ranges, range_of = np.unique(measurements[["scan_start", "scan_stop"]].to_numpy(), axis=0, return_inverse=True)
    logger.info("%s: %d calibrants, %d windows, %d tof indices", file.path.name, inside.sum(), len(windows),
                len(columns))
    sums = _sum_spectra(file, ranges, columns, segments, config.block)

    rows = []
    for m, r in zip(measurements.itertuples(), range_of.ravel(), strict=True):
        i, k = m.Index, int(m.calibrant)
        x = np.arange(tof_lo[k], tof_hi[k], dtype=float)
        y = sums[r, np.searchsorted(columns, x)]
        found = detect_mass_peaks(x, y, PEAK_WIDTH, 1.0, DETECTION)
        if len(found) == 0:
            continue
        fitted = fit_peak_shapes(x, y, found, sigma_guess[0] + sigma_guess[1] * found[:, 0], config.alpha_guess)
        rows += [(i, names[k], mzs[k], int(m.window), float(m.rt), mu, a, s, al, fitted.baseline)
                 for mu, a, s, al in zip(fitted.mu, fitted.area, fitted.sigma, fitted.alpha, strict=True)]
    return pd.DataFrame(rows, columns=["measurement", "calibrant", "mz_exact", "window", "rt", "tof", "area",
                                       "sigma", "alpha", "baseline"])


def calibrate(
    file: TofFile | str | PathLike,
    calibrants: Sequence[Calibrant] | Calibrants | str | PathLike | None = None,
    spikes: Mapping[float, Sequence[Calibrant]] | None = None,
    rt_start: float | None = None,
    rt_stop: float | None = None,
    initial: MassCalibration | str | PathLike | None = None,
    segments: tuple[int, int] | None = None,
    config: CalibrationConfig | None = None,
    **options,
) -> CalibrationResult:
    """Mass calibration of an h5 file, varying with the retention time (see the module).

    :param calibrants: present during the whole run (default: the calibrants of
        the acquisition calibration, :func:`h5_calibrants`). ``[]``: spikes only.
        Or a :class:`Calibrants` / its yaml file (a ``Path``, or a ``.yml`` / ``.yaml``
        name: continuous, spikes and initial calibration).
    :param spikes: {rt (s): calibrants} injected at rt, summed over ``spike_duration`` s.
        Added to those of a :class:`Calibrants`.
    :param rt_start, rt_stop: rt range calibrated (default: the whole run).
    :param initial: calibration used to find the calibrants (default: that of the
        :class:`Calibrants`, else the h5 attributes).
    :param segments: segment range [start, stop[ calibrated (summed): required for files with
        several segments (EI/CI instruments), default (0, 1). The file is named after it.
    :param config: parameters (default ``CalibrationConfig()``).
    :param options: fields of :class:`CalibrationConfig` changed, e.g. ``mode=0, averaging=60``.
    """
    t_start = time.perf_counter()
    config = replace(config or CalibrationConfig(), **options)  # TypeError for an unknown option
    mode, max_ppm = config.mode, config.max_ppm
    file = file if isinstance(file, TofFile) else TofFile(file)
    segments = file.resolve_segments(segments)  # required for files with several segments
    if isinstance(calibrants, PathLike) or (isinstance(calibrants, str)
                                            and calibrants.endswith((".yml", ".yaml"))):
        calibrants = Calibrants.from_yaml(calibrants)
    elif isinstance(calibrants, str):  # a single formula
        calibrants = [calibrants]
    if isinstance(calibrants, Calibrants):
        spikes = {**calibrants.spikes, **(spikes or {})}
        initial = calibrants.initial if initial is None else initial
        calibrants = calibrants.continuous
    if initial is None:
        initial = MassCalibration.from_h5(file)
    elif not isinstance(initial, MassCalibration):
        initial = MassCalibration.from_file(initial)
    continuous = h5_calibrants(file) if calibrants is None else resolve_calibrants(calibrants, config.ionised)
    spiked = {float(rt): resolve_calibrants(c, config.ionised) for rt, c in (spikes or {}).items()}
    if not continuous and not spiked:
        raise ValueError(f"No calibrant for {file.path.name}")

    windows = (calibration_windows(file.time_axis, config.averaging, config.spacing, rt_start, rt_stop)
               if continuous else None)
    cals, measurements, windows = _measurements(file, continuous, spiked, windows, config.spike_duration)
    peaks = find_calibrant_peaks(file, cals, measurements, windows, initial, config, segments)
    if peaks.empty:
        raise RuntimeError(f"No calibrant peak found in {file.path.name}")

    # the initial calibration can be off by a few tof indices: select and fit
    # again with the calibration fitted
    cal = initial
    for _ in range(2):
        table, peaks = fit_windows(peaks, windows, mode, cal, max_ppm)
        if not table["ok"].any():
            raise RuntimeError(f"The mass calibration of {file.path.name} failed in all windows "
                               f"(fewer than {N_PARAMS[mode] + 2} calibrant peaks within {max_ppm} ppm)")
        cal = _calibration_of(table, mode)
    for row in table[~table["ok"]].itertuples():
        logger.warning("%s: window at %.0f s not calibrated (%d calibrants, %d used, max %.1f ppm)",
                       file.path.name, row.rt, row.n_calibrants, row.n_used, row.max_ppm)

    used = peaks[peaks["used"]]
    if used["mz_exact"].min() > EXTRAPOLATION_WARNING:
        logger.warning("%s: the lowest calibrant is at m/z %.1f, the masses below are extrapolated (tens of "
                       "ppm). Add low mass calibrants, e.g. the air ions (N2, O2, Ar, CO2).",
                       file.path.name, used["mz_exact"].min())
    u_masses, u_ppm = calibration_uncertainty(used)
    cal = MassCalibration(mode=mode, rt=cal.rt, params=cal.params, **peak_shape(used, cal), u_masses=u_masses,
                          u_ppm=u_ppm, source=f"twutils.calibrate of {file.path}", segments=segments,
                          metadata=_metadata(file, config, cals, initial, rt_start, rt_stop, table))
    result = CalibrationResult(calibration=cal, peaks=peaks.drop(columns="measurement"), windows=table,
                               file=file, config=config, runtime=time.perf_counter() - t_start)
    logger.info(result.summary())
    return result


def _metadata(file: TofFile, config: CalibrationConfig, calibrants: list[tuple[str, float]],
              initial: MassCalibration, rt_start, rt_stop, windows: pd.DataFrame) -> dict:
    """How the calibration was made, stored in its file."""
    from datetime import datetime

    from twutils import __version__

    return {
        "source": {
            "file": str(file.path.absolute()),
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "versions": {"twutils": __version__, "peaxart": peaxart.__version__},
            "method": "twutils.calibrate",
            "config": asdict(config),
            "calibrants": [{"name": name, "mz": mz} for name, mz in calibrants],
            "initial": initial.source,
            "rt_range": [rt_start, rt_stop],
        },
        "windows": windows.to_dict(orient="records"),
    }


def calibrate_to_file(file: TofFile | str | PathLike, output: str | PathLike | None = None,
                      peaks_csv: bool = False, output_dir: str | PathLike | None = None,
                      **kwargs) -> CalibrationResult:
    """:func:`calibrate` and write the ``*_mc.json`` file (see :meth:`CalibrationResult.write`)."""
    result = calibrate(file, **kwargs)
    path = result.write(output, peaks_csv, output_dir)
    logger.info("Written %s", path)
    return result
