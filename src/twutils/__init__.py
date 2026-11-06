"""twutils: utilities for TOFWERK GC-TOF-MS h5 files.

Quick start::

    import twutils

    # peaks of a rt window, written in the format of the alpinac identification
    twutils.extract_to_file("241007.0423.air.1.h5", rt_start=1638, rt_stop=1647)

    # is there data in the run (filament, signal)?
    twutils.check("241007.0423.air.1.h5").status         # RunStatus.OK, RunStatus.FILAMENT_FAILED, ...

    # the mass calibration (fitted by esper calibrate, masscal) -> masscal.MassCalibration
    # (the unit mass chromatograms: esper chromatograms)
    cal = twutils.calibration_for("241007.0423.air.1.h5")  # <run>.alpinac/<run>.seg0_mc.json (or legacy .txt)

    # or step by step
    run = twutils.TofFile("241007.0423.air.1.h5")         # only the written scans: run.n_scans
    intensity, rt = run.read(1638, 1647, tof_start=20000, tof_stop=21000)
    result = twutils.extract(run, rt_start=1638, rt_stop=1647, masses=[127, 142], n_jobs=8)
    result.peaks                                         # peaxart table
    result.to_alpinac("CH3I.frag.txt")
"""

from twutils.calibration import (
    Ionisation,
    calibration_for,
    h5_calibrants,
    h5_calibration,
    legacy_mc_paths,
    mc_path,
)
from twutils.extraction import Extraction, default_output, extract, extract_to_file
from twutils.h5 import TofFile, TofFileError
from twutils.qc import QCConfig, RunQC, RunStatus, check, check_files, to_dataframe

__version__ = "0.1.0.dev0"

__all__ = [
    "Extraction",
    "Ionisation",
    "QCConfig",
    "RunQC",
    "RunStatus",
    "TofFile",
    "TofFileError",
    "calibration_for",
    "check",
    "check_files",
    "default_output",
    "extract",
    "extract_to_file",
    "h5_calibrants",
    "h5_calibration",
    "legacy_mc_paths",
    "mc_path",
    "to_dataframe",
]
