"""Plots of the mass calibration (legacy ``make_final_plots``).

From a :class:`~twutils.calibrate.CalibrationResult` (:meth:`~twutils.calibrate.CalibrationResult.plot`)
or from its output files: the ``*_mc.json`` (or legacy ``*_mc.txt``) file and the ``.peaks.csv`` file
next to it (``twutils calibrate --peaks-csv``). Without the peaks, only the
plots of the calibration itself (parameters, drift) are made.

- ``parameters``: p1, p2, ... and the calibrant residuals vs rt;
- ``drift``: mass drift vs rt, of the calibration (lines) and of each calibrant
  (points: its m/z with the calibration of the first window);
- ``residuals``: calibrant residuals vs m/z, with the calibration uncertainty;
- ``peak_shape``: sigma (m/z) and alpha of the calibrant peaks vs m/z, with their fits.

Needs matplotlib.
"""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError("The twutils plots need matplotlib: pip install 'twutils[plots]'") from e

from twutils.calibration import MassCalibration

logger = logging.getLogger(__name__)

PLOTS = ("parameters", "drift", "residuals", "peak_shape")
NEEDS_PEAKS = ("residuals", "peak_shape")
DRIFT_MASSES = (20.0, 50.0, 100.0, 200.0, 300.0)
CMAP = "viridis"


def read_outputs(mc_file: str | PathLike, peaks_csv: str | PathLike | None = None
                 ) -> tuple[MassCalibration, pd.DataFrame | None]:
    """Calibration and calibrant peaks written by :meth:`CalibrationResult.write`.

    ``peaks_csv`` defaults to the ``.peaks.csv`` file next to ``mc_file`` (None if absent).
    """
    mc_file = Path(mc_file)
    peaks_csv = mc_file.with_suffix(".peaks.csv") if peaks_csv is None else Path(peaks_csv)
    peaks = pd.read_csv(peaks_csv) if peaks_csv.is_file() else None
    return MassCalibration.from_file(mc_file), peaks


def _selected(peaks: pd.DataFrame) -> pd.DataFrame:
    """The calibrant peaks (not the other peaks of their domains), with a residual."""
    return peaks[peaks["selected"].astype(bool) & peaks["ppm"].notna()]


def _scatter(ax, x, y, c, used, norm, **kwargs):
    """Points used in the calibration filled, left out hollow."""
    used = np.asarray(used, dtype=bool)
    sc = ax.scatter(x[used], y[used], c=c[used], cmap=CMAP, norm=norm, s=14, **kwargs)
    if (~used).any():
        ax.scatter(x[~used], y[~used], facecolors="none", edgecolors=plt.get_cmap(CMAP)(norm(c[~used])),
                   s=18, label="left out", **kwargs)
    return sc


def _limit(ax, values, pad: float = 0.15, minimum: float = 1.0):
    """y range of ``values`` (the points used: left-out calibrants can be hundreds of ppm off)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values):
        lo, hi = values.min(), values.max()
        margin = max(hi - lo, minimum) * pad
        ax.set_ylim(lo - margin, hi + margin)


def _mz_colorbar(fig, ax, norm, label="calibrant m/z"):
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, label=label)


def plot_parameters(cal: MassCalibration, peaks: pd.DataFrame | None = None, title: str = "") -> Figure:
    """Parameters of the calibration vs rt, and the calibrant residuals vs rt."""
    n = cal.params.shape[1]
    fig, axes = plt.subplots(n + (peaks is not None), 1, sharex=True, figsize=(8, 1.8 * n + 3), squeeze=False)
    axes = axes[:, 0]
    for k in range(n):
        axes[k].plot(cal.rt, cal.params[:, k], ".-", color="k")
        axes[k].set_ylabel(f"p{k + 1}")
    if peaks is not None:
        p = _selected(peaks)
        norm = Normalize(p["mz_exact"].min(), p["mz_exact"].max())
        ax = axes[-1]
        _scatter(ax, p["rt"].to_numpy(), p["ppm"].to_numpy(), p["mz_exact"].to_numpy(), p["used"], norm)
        ax.axhline(0, color="grey", lw=0.8)
        used = p["used"].astype(bool)
        _limit(ax, np.r_[-np.abs(p.loc[used, "ppm"]), np.abs(p.loc[used, "ppm"])])
        ax.set_ylabel("residual (ppm)")
        if (~p["used"].astype(bool)).any():
            ax.legend(loc="upper right")
        _mz_colorbar(fig, axes, norm)
    axes[0].set_title(title or "Calibration parameters")
    axes[-1].set_xlabel("retention time (s)")
    return fig


def plot_drift(cal: MassCalibration, peaks: pd.DataFrame | None = None, masses=DRIFT_MASSES,
               title: str = "") -> Figure:
    """Mass drift vs rt (ppm): error if the calibration of the first window were used."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rt = np.linspace(cal.rt[0], cal.rt[-1], 200)
    lo, hi = min(masses), max(masses)
    if peaks is not None:
        p = _selected(peaks)
        lo, hi = min(lo, p["mz_exact"].min()), max(hi, p["mz_exact"].max())
    norm = Normalize(lo, hi)
    cmap = plt.get_cmap(CMAP)
    drift = [cal.drift_ppm(mz, rt) for mz in masses]
    for mz, d in zip(masses, drift, strict=True):
        ax.plot(rt, d, color=cmap(norm(mz)), lw=1.5, label=f"m/z {mz:g}")
    shown = np.ravel(drift)
    if peaks is not None:
        measured = (cal.mz(cal.rt[0], p["tof"]) - p["mz_exact"]) / p["mz_exact"] * 1e6
        _scatter(ax, p["rt"].to_numpy(), measured.to_numpy(), p["mz_exact"].to_numpy(), p["used"], norm, zorder=3)
        _mz_colorbar(fig, ax, norm, "m/z")
        shown = np.r_[shown, measured[p["used"].astype(bool)]]
    _limit(ax, np.r_[shown, 0.0])
    ax.axhline(0, color="grey", lw=0.8)
    ax.legend(title="calibration", fontsize="small")
    ax.set_xlabel("retention time (s)")
    ax.set_ylabel(f"drift since {cal.rt[0]:.0f} s (ppm)")
    ax.set_title(title or "Mass drift")
    ax.grid(True, ls="--", lw=0.5)
    return fig


def plot_residuals(cal: MassCalibration, peaks: pd.DataFrame, title: str = "") -> Figure:
    """Calibrant residuals vs m/z (colour: rt), with +- the calibration uncertainty."""
    p = _selected(peaks)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mz = np.linspace(min(p["mz_exact"].min(), 20.0), p["mz_exact"].max() * 1.05, 300)
    u = cal.u_ppm_at(mz)
    ax.fill_between(mz, -u, u, color="tab:blue", alpha=0.15, label="uncertainty")
    norm = Normalize(p["rt"].min(), p["rt"].max())
    _scatter(ax, p["mz_exact"].to_numpy(), p["ppm"].to_numpy(), p["rt"].to_numpy(), p["used"], norm)
    _mz_colorbar(fig, ax, norm, "retention time (s)")
    ax.axhline(0, color="grey", lw=0.8)
    used = np.abs(p.loc[p["used"].astype(bool), "ppm"])
    _limit(ax, np.r_[-used, used, -u.min(), u.min()])
    ax.legend(loc="upper right", fontsize="small")
    ax.set_xlabel("m/z")
    ax.set_ylabel("residual (ppm)")
    ax.set_title(title or "Calibrant residuals")
    ax.grid(True, ls="--", lw=0.5)
    return fig


def plot_peak_shape(cal: MassCalibration, peaks: pd.DataFrame, title: str = "") -> Figure:
    """Sigma (m/z) and alpha of the calibrant peaks vs m/z, with the fitted peak shape."""
    p = _selected(peaks)
    sigma_mz = cal.mz(p["rt"], p["tof"] + p["sigma"]) - cal.mz(p["rt"], p["tof"])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    norm = Normalize(p["rt"].min(), p["rt"].max())
    x, c, used = p["mz_exact"].to_numpy(), p["rt"].to_numpy(), p["used"]
    _scatter(ax1, x, np.asarray(sigma_mz), c, used, norm)
    _scatter(ax2, x, p["alpha"].to_numpy(), c, used, norm)
    mz = np.linspace(0, x.max() * 1.05, 2)
    if np.isfinite(cal.sigma_mz_slope):
        ax1.plot(mz, cal.sigma_mz_intercept + cal.sigma_mz_slope * mz, "--", color="tab:red",
                 label=f"{cal.sigma_mz_intercept:.3g} + {cal.sigma_mz_slope:.3g} x m/z")
        ax1.legend(fontsize="small")
    ax2.axhline(cal.alpha, ls="--", color="tab:red", label=f"mean {cal.alpha:.3f}")
    ax2.legend(fontsize="small")
    _mz_colorbar(fig, [ax1, ax2], norm, "retention time (s)")
    ax1.set_ylabel("sigma (m/z)")
    ax2.set_ylabel("alpha")
    ax2.set_xlabel("m/z")
    ax1.set_title(title or "Peak shape")
    for ax in (ax1, ax2):
        ax.grid(True, ls="--", lw=0.5)
    return fig


PLOTTERS = {"parameters": plot_parameters, "drift": plot_drift, "residuals": plot_residuals,
            "peak_shape": plot_peak_shape}


def plot_calibration(cal: MassCalibration, peaks: pd.DataFrame | None = None, output: str | PathLike | None = None,
                     plots=PLOTS, show: bool = False, title: str = "") -> dict[str, Path | Figure]:
    """Make the calibration plots.

    :param output: path of the calibration file: the plots are written next to
        it (``<run>.seg0_mc.<plot>.png``). None: not written.
    :param plots: names in :data:`PLOTS`; those needing the peaks are skipped without them.
    :param show: show the figures (else closed once written).
    :return: {name: png path} if written, else {name: figure}.
    """
    out = {}
    for name in plots:
        if name not in PLOTTERS:
            raise ValueError(f"Unknown plot '{name}', expected one of {PLOTS}")
        if peaks is None and name in NEEDS_PEAKS:
            logger.info("No calibrant peaks: plot '%s' skipped (write them with peaks_csv=True)", name)
            continue
        fig = PLOTTERS[name](cal, peaks, title=f"{title}: {name.replace('_', ' ')}" if title else "")
        if output is None:
            out[name] = fig
            continue
        path = Path(output).with_suffix(f".{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        out[name] = path
        if not show:
            plt.close(fig)
    if show:
        plt.show()
    return out


def plot_outputs(mc_file: str | PathLike, peaks_csv: str | PathLike | None = None, plots=PLOTS,
                 show: bool = False, write: bool = True) -> dict[str, Path | Figure]:
    """:func:`plot_calibration` from the files written by ``twutils calibrate``."""
    cal, peaks = read_outputs(mc_file, peaks_csv)
    mc_file = Path(mc_file)
    return plot_calibration(cal, peaks, mc_file if write else None, plots, show,
                            title=mc_file.name.removesuffix(".txt"))
