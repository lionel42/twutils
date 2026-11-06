"""The mass calibration of the h5 files: where its files are, and the acquisition calibration.

The calibration itself (:class:`masscal.MassCalibration`: tof index <-> m/z, peak shape,
uncertainty; its fit and file format) is in masscal; ``esper calibrate`` fits it.
Here, for an h5 file:

- :func:`mc_path`: ``<run>.alpinac/<run>.seg0_mc.json``, one per segment range of the
  spectra (``seg1-2``: segments 1 and 2 summed; EI/CI instruments);
- :func:`legacy_mc_paths`: the ``*_mc.txt`` files of the legacy alpinac;
- :func:`calibration_for`: the calibration used by the extraction and the chromatograms
  (the first of these files, else the acquisition calibration :func:`h5_calibration`);
- :func:`h5_calibrants`: the calibrants of the acquisition calibration.
"""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import Literal

from masscal import MassCalibration, Mode

from twutils.h5 import TofFile, segments_label, stem

logger = logging.getLogger(__name__)

Ionisation = Literal["EI", "CI"]
"""Ionisation of the spectra: names the legacy calibration files (``<run>.<EI|CI>_mc.txt``) and the
spectrum id of the extraction output. The calibration files of masscal are named by segment range."""

DEFAULT_SEGMENTS = (0, 1)
"""Segment range of files with one segment (EI instruments)."""


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


def h5_calibration(file: TofFile | str | PathLike) -> MassCalibration:
    """Acquisition calibration of the h5 file (constant), default peak shape and uncertainty."""
    file = file if isinstance(file, TofFile) else TofFile(file)
    attrs = file.attrs("FullSpectra")
    mode = Mode.from_tofwerk(int(attrs["MassCalibMode"]))
    params = [float(attrs[f"MassCalibration p{k}"]) for k in range(1, mode.n_params + 1)]
    return MassCalibration(mode=mode, rt=[0.0], params=[params], source=f"{file.path} (h5 attributes)")


def h5_calibrants(file: TofFile | str | PathLike) -> list[tuple[str, float]]:
    """Calibrants of the acquisition calibration (h5 attributes), as (name, m/z).

    TOFWERK: PFTBA ions only, usually without name (then their m/z).
    """
    file = file if isinstance(file, TofFile) else TofFile(file)
    attrs = file.attrs("FullSpectra")
    n = int(attrs.get("MassCalibration nbrPoints", 0))
    out = []
    for k in range(1, n + 1):
        mz = float(attrs[f"MassCalibration m{k}"])
        out.append((str(attrs.get(f"MassCalibration l{k}", "")) or f"{mz:.4f}", mz))
    return out


def calibration_for(file: TofFile | str | PathLike, segments: tuple[int, int] | None = None,
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
        cal = MassCalibration.from_file(mc_file)
        if cal.segments is not None and cal.segments != segments:
            logger.warning("%s: calibration of the segments %s used for the segments %s", mc_file,
                           cal.segments, segments)
        return cal
    for candidate in (mc_path(file, segments), *legacy_mc_paths(file, ionisation)):
        if candidate.is_file():
            logger.info("Mass calibration: %s", candidate)
            return MassCalibration.from_file(candidate)
    logger.warning("No mass calibration file for %s: using the h5 calibration (run esper calibrate "
                   "for accurate masses).", file.path.name)
    return h5_calibration(file)
