"""Mass calibration of a real run, against the legacy alpinac calibration of the same run (~25 s)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import twutils

RUN = Path("/media/Shuttle/TOF-Data/2024/240315.2211.air.1.h5")
LEGACY_MC = RUN.parent / "240315.2211.air.1.alpinac" / "240315.2211.air.1.EI_mc.txt"
CALIBRANTS = Path("/shared/scripts/tof-processing/config_calib.yml")  # of the pipeline

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not (RUN.is_file() and LEGACY_MC.is_file() and CALIBRANTS.is_file()),
                       reason=f"{RUN} or {CALIBRANTS} not available"),
]
ELECTRON_MASS = pytest.importorskip("alpinac.elements", reason="needs the alpinac of gitlabs/alpinac").ELECTRON_MASS


@pytest.fixture(scope="module")
def result():
    return twutils.calibrate(RUN, CALIBRANTS)


def test_real_calibration(result):
    windows = result.windows
    assert windows["ok"].all() and len(windows) == 24
    assert (windows["n_used"] >= 18).all() and (windows["rms_ppm"] < 10).all()
    # most calibrants are used in every window (some air ions only part of the run)
    used = result.peaks.groupby("calibrant")["used"].sum()
    assert (used >= 20).sum() >= 18


def test_real_against_legacy(result):
    """Same calibration as the legacy alpinac, except the electron mass (it used neutral masses).

    Compared away from the jump of the calibration at ~1800 s, where the
    windows of the two calibrations are not at the same rts. From 400 to 800
    s only on average: there N2, O2 and F are 40-80 ppm off the mode 2
    calibration (both codes leave N2 and O2 out), and the calibration depends
    on which of them is kept (up to ~16 ppm at m/z 28, ~5 ppm at m/z 131).
    Below m/z 69 the calibration is extrapolated from few ions: only on average.
    """
    legacy = twutils.MassCalibration.from_mc_file(LEGACY_MC)
    cal = result.calibration
    rt = np.r_[np.linspace(200, 1200, 6), np.linspace(2000, 2700, 4)][:, None]
    mz = np.array([28.0, 44.0, 69.0, 131.0, 219.0, 293.0])
    tof = legacy.tof(rt, mz)
    neutral = cal.mz(rt, tof) + ELECTRON_MASS  # ions of charge 1
    diff = np.abs((neutral - mz) / mz * 1e6)
    clear = ((rt < 400) | (rt > 800)).ravel()
    assert np.median(diff) < 1.5 and np.median(diff[:, :2]) < 2
    assert diff[clear][:, 2:].max() < 4
    # the drift over the run is the same (~70 ppm at m/z 150)
    np.testing.assert_allclose(np.ptp(cal.drift_ppm(150.0, rt)), np.ptp(legacy.drift_ppm(150.0, rt)), atol=5)
    # peak shape (the legacy file has narrower high-mass peaks: there its fits of
    # the Re and PFTBA peaks stopped at the lower sigma bound, the legacy
    # fit_pseudoVoigt of today gives the same peaks as twutils)
    np.testing.assert_allclose(cal.sigma([20000, 40000]), legacy.sigma([20000, 40000]), rtol=0.08)


def test_real_write(result, tmp_path):
    path = result.write(tmp_path / "run.EI_mc.txt")
    cal = twutils.MassCalibration.from_mc_file(path)
    tof = np.array([10000.0, 20000.0, 40000.0])
    np.testing.assert_allclose(cal.mz(1500.0, tof), result.calibration.mz(1500.0, tof), rtol=1e-8)
