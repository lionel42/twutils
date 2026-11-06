"""Mass calibration: tof index <-> m/z, and the mass peak shape.

The calibration functions are the ones of TOFWERK (``MassCalibMode``):

====  ===========================================
mode  tof index i of mass m
====  ===========================================
0     i = p1 sqrt(m) + p2
1     i = p1 / sqrt(m) + p2
2     i = p1 m^p3 + p2
3     i = p1 sqrt(m) + p2 + p3 (m - p4)^2        (m/z of tof indices not implemented)
4     i = p1 sqrt(m) + p2 + p3 m^2 + p4 m + p5   (m/z of tof indices not implemented)
5     m = p1 i^2 + p2 i + p3
====  ===========================================

The calibration of the GC-TOF runs drifts with the retention time. It is
fitted at several rts and written in a file with the peak shape (pseudo-Voigt
sigma vs tof index, Lorentzian fraction) and the calibration uncertainty vs m/z:

- ``<run>.alpinac/<run>.seg0_mc.json`` (:func:`twutils.calibrate`, see :data:`FORMAT`),
  one per segment range of the spectra (``seg1-2``: segments 1 and 2 summed; EI/CI
  instruments): rts in s, and how the calibration was made (file, calibrants,
  parameters, versions, result of each rt window);
- ``*_mc.txt`` of the legacy alpinac: read only (:meth:`MassCalibration.from_legacy_file`).

Without a file, the acquisition calibration stored in the h5 file is used, with
default peak shape and uncertainty.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np

from twutils.h5 import TofFile, segments_label, stem

logger = logging.getLogger(__name__)

N_PARAMS = {0: 2, 1: 2, 2: 3, 3: 4, 4: 5, 5: 3}

Ionisation = Literal["EI", "CI"]
"""Ionisation of the spectra: names the legacy calibration files (``<run>.<EI|CI>_mc.txt``) and the
spectrum id of the extraction output. The calibration files of twutils are named by segment range."""

FORMAT = "twutils.mass_calibration"
"""``format`` of the json calibration files; ``version`` is :data:`FORMAT_VERSION`. Content::

    {"format": "twutils.mass_calibration", "version": 1,
     "segments": [start, stop],   # segment range [start, stop[ of the spectra calibrated
     "mode": 2, "rt": [s, ...], "params": [[p1, p2, p3], ...],
     "peak_shape": {"sigma_slope", "sigma_intercept", "sigma_mz_slope", "sigma_mz_intercept", "alpha"},
     "uncertainty": {"mz": [...], "ppm": [...]},
     "metadata": {...}}   # free: source, windows (twutils.calibrate)

NaN values are written as null.
"""
FORMAT_VERSION = 1
DEFAULT_SEGMENTS = (0, 1)
"""Segment range of files with one segment (EI instruments)."""


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
    # modes 3 and 4 have no closed-form inverse: not needed yet
    raise NotImplementedError(f"m/z of tof indices for the mass calibration mode {mode}")


def mc_path(file: TofFile | str | PathLike, segments: tuple[int, int] | None = None) -> Path:
    """Calibration file of a run and segment range: ``<run>.alpinac/<run>.seg0_mc.json``.

    ``segments`` default: (0, 1); for a :class:`TofFile` with several segments, required.
    """
    if isinstance(file, TofFile):
        path, segments = file.path, file.resolve_segments(segments)
    else:
        path, segments = Path(file), segments or DEFAULT_SEGMENTS
    name = stem(path)
    return path.parent / f"{name}.alpinac" / f"{name}.{segments_label(segments)}_mc.json"


def legacy_mc_paths(file: TofFile | str | PathLike, ionisation: Ionisation = "EI") -> list[Path]:
    """Calibration files of the legacy alpinac, in its search order: ``<run>.alpinac/<run>.<ionisation>_mc.txt``,
    ``<run>.<ionisation>_mc.txt`` and ``<run>.mc.txt`` next to the h5 file."""
    path = file.path if isinstance(file, TofFile) else Path(file)
    name, directory = stem(path), path.parent
    return [directory / f"{name}.alpinac" / f"{name}.{ionisation}_mc.txt",
            directory / f"{name}.{ionisation}_mc.txt",
            directory / f"{name}.mc.txt"]


def _jsonable(value):
    """``value`` with numpy types, paths and tuples converted for json; NaN -> None."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _float(value) -> float:
    return float("nan") if value is None else float(value)


@dataclass
class MassCalibration:
    """Mass calibration of a run, optionally varying with the retention time.

    :param mode: TOFWERK calibration mode.
    :param rt: (n,) retention times (s) of the calibration points.
    :param params: (n, n_params) parameters p1, p2, ... at each rt; linearly
        interpolated in between, constant outside.
    :param sigma_slope, sigma_intercept: mass peak sigma = intercept + slope x tof index.
    :param sigma_mz_slope, sigma_mz_intercept: the same in m/z units vs m/z
        (only stored, not used).
    :param alpha: Lorentzian fraction of the pseudo-Voigt mass peaks.
    :param u_masses, u_ppm: calibration uncertainty (ppm) vs m/z.
    :param source: where the calibration was read.
    :param segments: segment range [start, stop[ of the spectra calibrated; None if not
        known (legacy files, h5 attributes).
    :param metadata: how it was made (json files: source file, calibrants,
        parameters, versions, windows), stored with it.
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
    segments: tuple[int, int] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rt = np.atleast_1d(np.asarray(self.rt, dtype=float))
        self.params = np.atleast_2d(np.asarray(self.params, dtype=float))
        self.u_masses = np.asarray(self.u_masses, dtype=float)
        self.u_ppm = np.asarray(self.u_ppm, dtype=float)
        if self.segments is not None:
            self.segments = (int(self.segments[0]), int(self.segments[1]))

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
    def from_legacy_file(cls, path: str | PathLike) -> MassCalibration:
        """Calibration file of the legacy alpinac (``*_mc.txt``, read only).

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
                logger.warning("%s: p3 missing at rt index %s, using 0.5", path.name, line[0])
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

    @classmethod
    def from_json(cls, path: str | PathLike) -> MassCalibration:
        """Calibration file of twutils (``*_mc.json``, see :data:`FORMAT`)."""
        path = Path(path)
        content = json.loads(path.read_text())
        if content.get("format") != FORMAT:
            raise ValueError(f"{path}: not a mass calibration file of twutils (format {content.get('format')!r})")
        if content.get("version", 0) > FORMAT_VERSION:
            raise ValueError(f"{path}: format version {content['version']} is newer than this twutils "
                             f"({FORMAT_VERSION}): update twutils")
        shape, u = content.get("peak_shape", {}), content.get("uncertainty", {})
        kwargs = {k: _float(shape[k]) for k in ("sigma_slope", "sigma_intercept", "sigma_mz_slope",
                                                 "sigma_mz_intercept", "alpha") if k in shape}
        if u.get("mz"):
            kwargs |= {"u_masses": [_float(v) for v in u["mz"]], "u_ppm": [_float(v) for v in u["ppm"]]}
        params = [[_float(v) for v in row] for row in content["params"]]
        segments = content.get("segments")
        return cls(mode=int(content["mode"]), rt=[_float(v) for v in content["rt"]], params=params,
                   source=str(path), segments=tuple(segments) if segments else None,
                   metadata=content.get("metadata", {}), **kwargs)

    def to_json(self, path: str | PathLike) -> Path:
        """Write the calibration file of twutils (read by :meth:`from_json`, see :data:`FORMAT`)."""
        path = Path(path)
        content = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "segments": self.segments,
            "mode": self.mode,
            "rt": self.rt,
            "params": self.params,
            "peak_shape": {"sigma_slope": self.sigma_slope, "sigma_intercept": self.sigma_intercept,
                           "sigma_mz_slope": self.sigma_mz_slope, "sigma_mz_intercept": self.sigma_mz_intercept,
                           "alpha": self.alpha},
            "uncertainty": {"mz": self.u_masses, "ppm": self.u_ppm},
            "metadata": self.metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(content), indent=1, allow_nan=False) + "\n")
        return path

    @classmethod
    def from_file(cls, path: str | PathLike) -> MassCalibration:
        """A calibration file: json of twutils (``*_mc.json``) or legacy (``*_mc.txt``), from its content."""
        path = Path(path)
        is_json = path.read_text().lstrip().startswith("{")
        return cls.from_json(path) if is_json else cls.from_legacy_file(path)

    @classmethod
    def for_file(cls, file: TofFile | str | PathLike, segments: tuple[int, int] | None = None,
                 ionisation: Ionisation = "EI", mc_file: str | PathLike | None = None) -> MassCalibration:
        """Calibration of the segment range ``segments`` of an h5 file (default (0, 1), required
        for files with several segments).

        In order: ``mc_file`` if given, ``<run>.alpinac/<run>.seg0_mc.json`` (:func:`mc_path`),
        the legacy files of ``ionisation`` (:func:`legacy_mc_paths`), then the h5 attributes
        (with a warning).
        """
        file = file if isinstance(file, TofFile) else TofFile(file)
        segments = file.resolve_segments(segments)
        if mc_file is not None:
            cal = cls.from_file(mc_file)
            if cal.segments is not None and cal.segments != segments:
                logger.warning("%s: calibration of the segments %s used for the segments %s", mc_file,
                               cal.segments, segments)
            return cal
        for candidate in (mc_path(file, segments), *legacy_mc_paths(file, ionisation)):
            if candidate.is_file():
                logger.info("Mass calibration: %s", candidate)
                return cls.from_file(candidate)
        logger.warning("No mass calibration file for %s: using the h5 calibration (run twutils calibrate "
                       "for accurate masses).", file.path.name)
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
