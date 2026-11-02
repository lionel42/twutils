"""twutils: utilities for TOFWERK GC-TOF-MS h5 files.

Quick start::

    import twutils

    # peaks of a rt window, written in the format of the alpinac identification
    twutils.extract_to_file("241007.0423.air.1.h5", rt_start=1638, rt_stop=1647)

    # mass calibration following the drift -> <run>.alpinac/<run>.EI_mc.txt
    twutils.calibrate("241007.0423.air.1.h5", "config_calib.yml").write()

    # or step by step
    run = twutils.TofFile("241007.0423.air.1.h5")
    cal = twutils.MassCalibration.for_file(run)          # <run>.alpinac/<run>.EI_mc.txt
    intensity, rt = run.read(1638, 1647, tof_start=20000, tof_stop=21000)
    result = twutils.extract(run, rt_start=1638, rt_stop=1647, masses=[127, 142], n_jobs=8)
    result.peaks                                         # peaxart table
    result.to_alpinac("CH3I.frag.txt")
"""

from twutils.calibrate import (
    Calibrants,
    CalibrationResult,
    calibrate,
    calibrate_to_file,
)
from twutils.calibration import MassCalibration, mass_to_tof, tof_to_mass
from twutils.extract import Extraction, default_output, extract, extract_to_file
from twutils.h5 import TofFile

__version__ = "0.1.0.dev0"

__all__ = [
    "Calibrants",
    "CalibrationResult",
    "Extraction",
    "MassCalibration",
    "TofFile",
    "calibrate",
    "calibrate_to_file",
    "default_output",
    "extract",
    "extract_to_file",
    "mass_to_tof",
    "tof_to_mass",
]
