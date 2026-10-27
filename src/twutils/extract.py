"""Peak extraction of TOFWERK h5 files with peaxart, output for the alpinac identification."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import peaxart

from twutils.calibration import MassCalibration
from twutils.h5 import TofFile

logger = logging.getLogger(__name__)

SPECTRUM_ID = {"EI": 0, "CI": 1}


@dataclass
class Extraction:
    """Result of :func:`extract`: the peaks and how they were obtained."""

    peaks: pd.DataFrame
    """peaxart peak table (see :func:`peaxart.extract`) with a ``compound`` column."""
    file: TofFile
    calibration: MassCalibration
    rt_start: float
    rt_stop: float
    masses: list[float]
    mz_range: float
    ionisation: str
    segments: tuple[int, int] | None
    config: peaxart.ExtractionConfig
    runtime: float

    def to_alpinac(self, path: str | PathLike | None = None, adduct: str = "",
                   metadata: dict[str, str] | None = None) -> pd.DataFrame:
        """Peak table in the alpinac format; written to ``path`` if given (see :func:`peaxart.to_alpinac`)."""
        return peaxart.to_alpinac(
            self.peaks, path,
            mass_cal_u=self.calibration.u_ppm_at,
            ionisation=self.ionisation,
            adduct=adduct if self.ionisation == "CI" else "",
            spectrum=str(SPECTRUM_ID.get(self.ionisation, 0)),
            metadata={**self.metadata(), **(metadata or {})},
        )

    def metadata(self) -> dict[str, str]:
        default = asdict(peaxart.ExtractionConfig())
        changed = {k: v for k, v in asdict(self.config).items() if default[k] != v}
        masses = (f"{self.masses[0]:g}-{self.masses[-1]:g}"
                  if np.array_equal(np.diff(self.masses), np.ones(len(self.masses) - 1))
                  else " ".join(f"{m:g}" for m in self.masses))
        from twutils import __version__

        return {
            "File extracted": str(self.file.path),
            "rt": f"{self.rt_start:.2f} - {self.rt_stop:.2f} s",
            "masses": f"{masses} (+-{self.mz_range} m/z)",
            "ionisation": self.ionisation + ("" if self.segments is None else f" (segments {self.segments})"),
            "mass calibration": self.calibration.source,
            "peaxart config": str(changed) if changed else "defaults",
            "versions": f"twutils {__version__}, peaxart {peaxart.__version__}",
            "runtime": f"{self.runtime:.1f} s",
        }


def default_masses(file: TofFile, calibration: MassCalibration, rt: float,
                   mass_start: float | None = None, mass_stop: float | None = None) -> list[float]:
    """Unit masses from ``mass_start`` (default: first of the file) to ``mass_stop`` (default: last)."""
    if mass_start is None:
        mass_start = max(1, int(round(float(file.mass_axis[0]))))
    if mass_stop is None:
        mass_stop = math.floor(float(calibration.mz(rt, file.n_points - 1)))
    return [float(m) for m in range(int(round(mass_start)), int(round(mass_stop)) + 1)]


def _tof_range(file: TofFile, calibration: MassCalibration, rts: Sequence[float],
               masses: Sequence[float], mz_range: float) -> tuple[int, int]:
    """Tof indices needed for the domains of all masses over the rts (with 2 points of margin)."""
    rts = np.asarray(rts, dtype=float)
    lo = calibration.tof(rts, min(masses) - mz_range)
    hi = calibration.tof(rts, max(masses) + mz_range)
    return (max(0, int(np.floor(np.nanmin(lo))) - 2),
            min(file.n_points, int(np.ceil(np.nanmax(hi))) + 3))


def extract(
    file: TofFile | str | PathLike,
    rt_start: float | None = None,
    rt_stop: float | None = None,
    masses: Sequence[float] | None = None,
    mass_start: float | None = None,
    mass_stop: float | None = None,
    mz_range: float = 0.5,
    calibration: MassCalibration | str | PathLike | None = None,
    ionisation: str = "EI",
    segments: tuple[int, int] | None = None,
    config: peaxart.ExtractionConfig | None = None,
    n_jobs: int = 1,
    block: float = 120.0,
    margin: float = 10.0,
) -> Extraction:
    """Extract the peaks of an h5 file with peaxart.

    :param file: h5 file.
    :param rt_start, rt_stop: retention time window, s (default: the whole run).
    :param masses: unit masses to extract; default ``mass_start`` to ``mass_stop``
        (default: the masses of the file).
    :param mz_range: half width of the domains around each unit mass, m/z.
    :param calibration: mass calibration, or the path of a ``*_mc.txt`` file;
        default: searched with :meth:`MassCalibration.for_file`.
    :param ionisation: ``"EI"`` or ``"CI"`` (selects the calibration file and the
        spectrum id of the output).
    :param segments: segment range [start, stop[ to read (EI/CI instruments).
    :param config: peaxart parameters (default ``peaxart.ExtractionConfig()``;
        ``peaxart.ExtractionConfig.legacy()`` for the legacy alpinac results).
    :param n_jobs: domains extracted in parallel.
    :param block: windows longer than this (s) are read and extracted in blocks
        of this length, each with ``margin`` s of data on both sides (a full run
        does not fit in memory); a peak belongs to the block of its apex.
    :return: the peaks and the extraction parameters.
    """
    t0 = time.perf_counter()
    file = file if isinstance(file, TofFile) else TofFile(file)
    if calibration is None:
        calibration = MassCalibration.for_file(file, ionisation)
    elif not isinstance(calibration, MassCalibration):
        calibration = MassCalibration.from_mc_file(calibration)
    config = config or peaxart.ExtractionConfig()

    t = file.time_axis
    rt_start = float(t[0]) if rt_start is None else max(float(rt_start), float(t[0]))
    rt_stop = float(t[-1]) if rt_stop is None else min(float(rt_stop), float(t[-1]))
    if rt_stop <= rt_start:
        raise ValueError(f"Empty rt window {rt_start} - {rt_stop} s (run: {t[0]:.1f} - {t[-1]:.1f} s)")
    if masses is None:
        masses = default_masses(file, calibration, (rt_start + rt_stop) / 2, mass_start, mass_stop)
    masses = sorted(float(m) for m in masses)
    if not masses:
        raise ValueError("No mass to extract")

    n_blocks = max(1, math.ceil((rt_stop - rt_start) / block - 1e-9))
    edges = np.linspace(rt_start, rt_stop, n_blocks + 1)
    frames = []
    for k in range(n_blocks):
        b0, b1 = edges[k], edges[k + 1]
        r0, r1 = (b0, b1) if n_blocks == 1 else (b0 - margin, b1 + margin)
        tof0, tof1 = _tof_range(file, calibration, [r0, (r0 + r1) / 2, r1], masses, mz_range)
        intensity, rt = file.read(r0, r1, tof0, tof1, segments)
        if len(rt) < 2:
            continue
        logger.info(f"rt {b0:.1f} - {b1:.1f} s: {intensity.shape[0]} scans x {intensity.shape[1]} points, "
                    f"{len(masses)} masses")
        peaks = peaxart.extract(intensity, rt, calibration.mz, calibration.sigma, calibration.alpha,
                                x=np.arange(tof0, tof1, dtype=float), masses=masses, mz_range=mz_range,
                                config=config, dt=file.s_per_bin, n_jobs=n_jobs)
        if n_blocks > 1:
            last = k == n_blocks - 1
            peaks = peaks[(peaks["rt"] >= b0) & ((peaks["rt"] <= b1) if last else (peaks["rt"] < b1))]
        frames.append(peaks)

    frames = [f for f in frames if len(f)]
    if frames:
        peaks = pd.concat(frames, ignore_index=True).sort_values(["rt", "mz"], ignore_index=True)
    else:
        peaks = pd.DataFrame(columns=["mass"] + peaxart.PEAK_COLUMNS)
    peaks["compound"] = peaxart.group_compounds(peaks, dt=file.s_per_bin, config=config)
    runtime = time.perf_counter() - t0
    logger.info(f"{file.path.name}: {len(peaks)} peaks, {peaks['compound'].max() + 1 if len(peaks) else 0} "
                f"compounds, {runtime:.1f} s")
    return Extraction(peaks=peaks, file=file, calibration=calibration, rt_start=rt_start, rt_stop=rt_stop,
                      masses=masses, mz_range=mz_range, ionisation=ionisation, segments=segments,
                      config=config, runtime=runtime)


def default_output(file: TofFile | str | PathLike, rt_start: float | None = None,
                   rt_stop: float | None = None, output_dir: str | PathLike | None = None) -> Path:
    """Output path as the legacy alpinac: ``<run>.alpinac/<run>.frag.<start>s<stop>s.txt``."""
    path = file.path if isinstance(file, TofFile) else Path(file)
    stem = path.name[:-3] if path.name.endswith(".h5") else path.stem
    directory = path.parent / f"{stem}.alpinac" if output_dir is None else Path(output_dir)
    if rt_start is None and rt_stop is None:
        name = f"{stem}.frag.all_rts.txt"
    else:
        name = f"{stem}.frag.{int(rt_start or 0)}s{int(rt_stop) if rt_stop is not None else 'end'}s.txt"
    return directory / name


def extract_to_file(
    file: TofFile | str | PathLike,
    output: str | PathLike | None = None,
    output_dir: str | PathLike | None = None,
    metadata: dict[str, str] | None = None,
    adduct: str = "",
    peaks_csv: bool = False,
    **kwargs,
) -> Path:
    """Extract the peaks of an h5 file and write the alpinac peak file.

    :param output: output file; default :func:`default_output` (in ``output_dir`` if given).
    :param metadata: added to the header of the file.
    :param adduct: adduct of CI spectra.
    :param peaks_csv: also write the full peaxart table next to it (``.peaks.csv``).
    :param kwargs: arguments of :func:`extract`.
    :return: the path written.
    """
    file = file if isinstance(file, TofFile) else TofFile(file)
    if output is None:
        output = default_output(file, kwargs.get("rt_start"), kwargs.get("rt_stop"), output_dir)
    output = Path(output)
    result = extract(file, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_alpinac(output, adduct=adduct, metadata=metadata)
    if peaks_csv:
        result.peaks.to_csv(output.with_suffix(".peaks.csv"), index=False)
    logger.info(f"Written {output}")
    return output
