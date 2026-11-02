"""Mass calibration: tof index <-> m/z, and the mass peak shape.

The calibration functions are the ones of TOFWERK (``MassCalibMode``):

====  ===========================================
mode  tof index i of mass m
====  ===========================================
0     i = p1 sqrt(m) + p2
1     i = p1 / sqrt(m) + p2
2     i = p1 m^p3 + p2
3     i = p1 sqrt(m) + p2 + p3 (m - p4)^2
4     i = p1 sqrt(m) + p2 + p3 m^2 + p4 m + p5
5     m = p1 i^2 + p2 i + p3
====  ===========================================

The calibration of the GC-TOF runs drifts with the retention time. It is
fitted at several rts (:func:`twutils.calibrate`, or the legacy alpinac) and
written in a ``*_mc.txt`` file, with the peak shape (pseudo-Voigt sigma vs tof
index, Lorentzian fraction) and the calibration uncertainty vs m/z. Without
that file, the acquisition calibration stored in the h5 file is used, with
default peak shape and uncertainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np

from twutils.h5 import TofFile

logger = logging.getLogger(__name__)

N_PARAMS = {0: 2, 1: 2, 2: 3, 3: 4, 4: 5, 5: 3}


def mass_to_tof(mass, params: dict, mode: int) -> np.ndarray:
    """Tof index of masses (``params``: p1, p2, ... numbers or arrays broadcast with ``mass``)."""
    m = np.asarray(mass, dtype=float)
    p = params
    match mode:
        case 0:
            return p["p1"] * np.sqrt(m) + p["p2"]
        case 1:
            return p["p1"] / np.sqrt(m) + p["p2"]
        case 2:
            return p["p1"] * m ** p["p3"] + p["p2"]
        case 3:
            return p["p1"] * np.sqrt(m) + p["p2"] + p["p3"] * (m - p["p4"]) ** 2
        case 4:
            return p["p1"] * np.sqrt(m) + p["p2"] + p["p3"] * m**2 + p["p4"] * m + p["p5"]
        case 5:
            a, b, c = p["p1"], p["p2"], p["p3"] - m
            if np.abs(a).max() == 0:
                return -c / b
            return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    raise NotImplementedError(f"Mass calibration mode {mode}")


def tof_to_mass(tof, params: dict, mode: int) -> np.ndarray:
    """m/z of tof indices (inverse of :func:`mass_to_tof`)."""
    i = np.asarray(tof, dtype=float)
    p = params
    match mode:
        case 0:
            return ((i - p["p2"]) / p["p1"]) ** 2
        case 1:
            return (p["p1"] / (i - p["p2"])) ** 2
        case 2:
            return ((i - p["p2"]) / p["p1"]) ** (1.0 / p["p3"])
        case 5:
            return p["p1"] * i**2 + p["p2"] * i + p["p3"]
        case 3 | 4:
            if any(np.ndim(v) for v in p.values()):
                raise NotImplementedError(f"Mode {mode} with rt-dependent parameters")
            grid = np.geomspace(0.5, 5000.0, 200_000)
            return np.interp(i, mass_to_tof(grid, p, mode), grid, left=np.nan, right=np.nan)
    raise NotImplementedError(f"Mass calibration mode {mode}")


def mc_path(file: TofFile, ionisation: str = "EI") -> Path:
    """Calibration file of a run, as the legacy alpinac: ``<run>.alpinac/<run>.<ionisation>_mc.txt``."""
    return file.path.parent / f"{file.stem}.alpinac" / f"{file.stem}.{ionisation}_mc.txt"


@dataclass
class MassCalibration:
    """Mass calibration of a run, optionally varying with the retention time.

    :param mode: TOFWERK calibration mode.
    :param rt: (n,) retention times (s) of the calibration points.
    :param params: (n, n_params) parameters p1, p2, ... at each rt; linearly
        interpolated in between, constant outside.
    :param sigma_slope, sigma_intercept: mass peak sigma = intercept + slope x tof index.
    :param sigma_mz_slope, sigma_mz_intercept: the same in m/z units vs m/z
        (only written in the ``*_mc.txt`` file).
    :param alpha: Lorentzian fraction of the pseudo-Voigt mass peaks.
    :param u_masses, u_ppm: calibration uncertainty (ppm) vs m/z.
    :param source: where the calibration was read.
    """

    mode: int
    rt: np.ndarray
    params: np.ndarray
    sigma_slope: float = 3.6e-05
    sigma_intercept: float = 0.51
    sigma_mz_slope: float = float("nan")
    sigma_mz_intercept: float = float("nan")
    alpha: float = 0.48
    u_masses: np.ndarray = field(default_factory=lambda: np.array([0.0, 400.0]))
    u_ppm: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0]))
    source: str = ""

    def __post_init__(self):
        self.rt = np.atleast_1d(np.asarray(self.rt, dtype=float))
        self.params = np.atleast_2d(np.asarray(self.params, dtype=float))
        self.u_masses = np.asarray(self.u_masses, dtype=float)
        self.u_ppm = np.asarray(self.u_ppm, dtype=float)

    # --- readers ----------------------------------------------------------

    @classmethod
    def from_h5(cls, file: TofFile | str | PathLike) -> MassCalibration:
        """Acquisition calibration of the h5 file (constant), default peak shape and uncertainty."""
        file = file if isinstance(file, TofFile) else TofFile(file)
        attrs = file.attrs("FullSpectra")
        mode = int(attrs["MassCalibMode"])
        params = [float(attrs[f"MassCalibration p{k}"]) for k in range(1, N_PARAMS[mode] + 1)]
        return cls(mode=mode, rt=[0.0], params=[params], source=f"{file.path} (h5 attributes)")

    @classmethod
    def from_mc_file(cls, path: str | PathLike) -> MassCalibration:
        """Calibration file of the legacy alpinac (``*_mc.txt``).

        9 header lines (number of calibration rts, number of uncertainty
        points, mode, sigma slope and intercept vs tof index, sigma slope and
        intercept vs mass, alpha, s per bin), then one line per rt (rt index,
        p1, p2, ...) and one line per uncertainty point (m/z, ppm).
        """
        path = Path(path)
        lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
        head = [line[0] for line in lines[:9]]
        n_rt, n_u, mode = int(head[0]), int(head[1]), int(head[2])
        s_per_bin = float(head[8])
        n_params = N_PARAMS[mode]
        rts, params = [], []
        for line in lines[9:9 + n_rt]:
            values = line[1:1 + n_params]
            if mode == 2 and (len(values) < 3 or values[2] == "[]"):
                logger.warning(f"{path.name}: p3 missing at rt index {line[0]}, using 0.5")
                values = values[:2] + ["0.5"]
            rts.append(float(line[0]) * s_per_bin)
            params.append([float(v) for v in values])
        kwargs = {}
        if n_u > 0:
            u = np.array([[float(v) for v in line[:2]] for line in lines[9 + n_rt:9 + n_rt + n_u]])
            kwargs = {"u_masses": u[:, 0], "u_ppm": u[:, 1]}
        order = np.argsort(rts)
        return cls(mode=mode, rt=np.array(rts)[order], params=np.array(params)[order],
                   sigma_slope=float(head[3]), sigma_intercept=float(head[4]),
                   sigma_mz_slope=float(head[5]), sigma_mz_intercept=float(head[6]), alpha=float(head[7]),
                   source=str(path), **kwargs)

    def to_mc_file(self, path: str | PathLike, s_per_bin: float) -> Path:
        """Write the calibration file of the legacy alpinac (read by :meth:`from_mc_file`).

        The rts are written as scan indices (``rt / s_per_bin``, rounded).
        """
        path = Path(path)
        head = [
            (len(self.rt), "Number of points, mass cal. vs RT"),
            (len(self.u_masses), "Number of points, mass cal. u vs mass"),
            (self.mode, "Mass cal. mode"),
            (self.sigma_slope, "Peak shape, sigma_slope vs TOF index"),
            (self.sigma_intercept, "Peak shape, sigma_b vs TOF index"),
            (self.sigma_mz_slope, "Peak shape, sigma_slope vs mass"),
            (self.sigma_mz_intercept, "Peak shape, sigma_b vs mass"),
            (self.alpha, "alpha, for tofidx Pseudo-Voigt fit"),
            (s_per_bin, "s_per_bin"),
        ]
        lines = [f"{value}\t{text}" for value, text in head]
        lines += ["\t".join([str(round(rt / s_per_bin))] + [repr(float(p)) for p in params])
                  for rt, params in zip(self.rt, self.params, strict=True)]
        lines += [f"{m:.7f}\t{u:.2f}" for m, u in zip(self.u_masses, self.u_ppm, strict=True)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    @classmethod
    def for_file(cls, file: TofFile | str | PathLike, ionisation: str = "EI",
                 mc_file: str | PathLike | None = None) -> MassCalibration:
        """Calibration of an h5 file, searched as the legacy alpinac does.

        In order: ``mc_file`` if given, ``<run>.alpinac/<run>.<ionisation>_mc.txt``,
        ``<run>.<ionisation>_mc.txt`` and ``<run>.mc.txt`` next to the h5 file,
        then the h5 attributes (with a warning).
        """
        file = file if isinstance(file, TofFile) else TofFile(file)
        if mc_file is not None:
            return cls.from_mc_file(mc_file)
        directory, stem = file.path.parent, file.stem
        for candidate in (mc_path(file, ionisation),
                          directory / f"{stem}.{ionisation}_mc.txt",
                          directory / f"{stem}.mc.txt"):
            if candidate.is_file():
                logger.info(f"Mass calibration: {candidate}")
                return cls.from_mc_file(candidate)
        logger.warning(f"No mass calibration file for {file.path.name}: using the h5 calibration "
                       "(run the alpinac mass calibration for accurate masses).")
        return cls.from_h5(file)

    # --- evaluation -------------------------------------------------------

    @property
    def is_constant(self) -> bool:
        return len(self.rt) == 1

    def params_at(self, rt) -> dict:
        """Parameters at retention time(s) ``rt`` (interpolated): {"p1": ..., ...}."""
        if self.is_constant:
            return {f"p{k + 1}": v for k, v in enumerate(self.params[0])}
        rt = np.asarray(rt, dtype=float)
        return {f"p{k + 1}": np.interp(rt, self.rt, self.params[:, k]) for k in range(self.params.shape[1])}

    def mz(self, rt, tof) -> np.ndarray:
        """m/z of tof indices at retention times (broadcast)."""
        rt, tof = np.broadcast_arrays(np.asarray(rt, dtype=float), np.asarray(tof, dtype=float))
        return tof_to_mass(tof, self.params_at(rt), self.mode)

    def tof(self, rt, mz) -> np.ndarray:
        """Tof index of m/z values at retention times (broadcast)."""
        rt, mz = np.broadcast_arrays(np.asarray(rt, dtype=float), np.asarray(mz, dtype=float))
        return mass_to_tof(mz, self.params_at(rt), self.mode)

    def drift_ppm(self, mz, rt, reference_rt: float | None = None) -> np.ndarray:
        """Mass drift: error (ppm) on ``mz`` at ``rt`` if the calibration of
        ``reference_rt`` (default: the first calibration rt) were used (broadcast)."""
        reference_rt = float(self.rt[0]) if reference_rt is None else reference_rt
        mz = np.asarray(mz, dtype=float)
        return (self.mz(reference_rt, self.tof(rt, mz)) - mz) / mz * 1e6

    def sigma(self, tof) -> np.ndarray:
        """Sigma of the mass peaks, tof index units."""
        return self.sigma_intercept + self.sigma_slope * np.asarray(tof, dtype=float)

    def u_ppm_at(self, mz, min_u_mz: float = 0.001, min_u_below: float = 65.0) -> np.ndarray:
        """Calibration uncertainty (ppm) of peaks at ``mz`` (legacy rule, see :func:`peaxart.mass_cal_u_ppm`)."""
        import peaxart

        return peaxart.mass_cal_u_ppm(mz, self.u_masses, self.u_ppm, min_u_mz, min_u_below)
