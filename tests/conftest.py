"""A small synthetic TOFWERK h5 file: one compound with two mass peaks."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from twutils.calibration import mass_to_tof

N_WRITES, N_BUFS, N_POINTS = 40, 10, 6000
S_PER_BIN = 0.2
CAL = {"p1": 600.0, "p2": -200.0, "p3": 0.5}  # mode 2
RT_APEX = 40.0
PEAKS = {(69.0, 1000.0), (100.0, 400.0)}  # (m/z, height)
SIGMA_B, SIGMA_SLOPE, ALPHA = 0.5, 4e-5, 0.5


def pseudo_voigt(x, mu, sigma, alpha):
    gauss = np.exp(-((x - mu) ** 2) / (2 * sigma) ** 2)  # legacy convention
    lorentz = 1 / (1 + ((x - mu) / sigma) ** 2)
    return (1 - alpha) * gauss + alpha * lorentz


def write_h5(path, n_segments: int = 1, unwritten: int = 0):
    rng = np.random.default_rng(0)
    rt = (np.arange(N_WRITES * N_BUFS) + 1) * S_PER_BIN
    tof = np.arange(N_POINTS, dtype=float)
    data = rng.normal(1.0, 0.3, (len(rt), N_POINTS)).astype(np.float32)
    chromato = np.exp(-((rt - RT_APEX) ** 2) / (2 * 1.5**2))
    for mz, height in PEAKS:
        mu = float(mass_to_tof(mz, CAL, 2))
        data += (height * chromato[:, None] * pseudo_voigt(tof, mu, SIGMA_B + SIGMA_SLOPE * mu, ALPHA)).astype(np.float32)
    data = data.reshape(N_WRITES, N_BUFS, 1, N_POINTS)
    data = np.repeat(data, n_segments, axis=2) / n_segments
    buf_times = rt.reshape(N_WRITES, N_BUFS).copy()
    if unwritten:
        buf_times.ravel()[-unwritten:] = 0
    with h5py.File(path, "w") as f:
        fs = f.create_group("FullSpectra")
        fs.create_dataset("TofData", data=data, chunks=(1, 1, 1, N_POINTS))
        fs.create_dataset("MassAxis", data=((tof - CAL["p2"]) / CAL["p1"]) ** 2)
        fs.attrs["MassCalibMode"] = np.array([2], dtype=np.int32)
        for k, v in CAL.items():
            fs.attrs[f"MassCalibration {k}"] = np.array([v])
        td = f.create_group("TimingData")
        td.create_dataset("BufTimes", data=buf_times)
        td.attrs["BlockPeriod"] = np.array([S_PER_BIN * 1e9])
    return path


def write_mc(path, rts=(0.0, 400.0)):
    """Calibration file of the legacy alpinac, same calibration at every rt."""
    lines = [f"{len(rts)}\tNumber of points, mass cal. vs RT", "2\tNumber of points, mass cal. u vs mass",
             "2\tMass cal. mode", f"{SIGMA_SLOPE}\tPeak shape, sigma_slope vs TOF index",
             f"{SIGMA_B}\tPeak shape, sigma_b vs TOF index", "8e-05\tPeak shape, sigma_slope vs mass",
             "0.0015\tPeak shape, sigma_b vs mass", f"{ALPHA}\talpha, for tofidx Pseudo-Voigt fit",
             f"{S_PER_BIN}\ts_per_bin"]
    lines += [f"{int(rt / S_PER_BIN)}\t{CAL['p1']}\t{CAL['p2']}\t{CAL['p3']}" for rt in rts]
    lines += ["68.9946609\t2.50", "130.9914672\t1.50"]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def h5_file(tmp_path):
    path = write_h5(tmp_path / "241007.0423.air.1.h5")
    (tmp_path / "241007.0423.air.1.alpinac").mkdir()
    write_mc(tmp_path / "241007.0423.air.1.alpinac" / "241007.0423.air.1.EI_mc.txt")
    return path
