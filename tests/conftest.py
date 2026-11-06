"""Small synthetic TOFWERK h5 files.

- :func:`write_h5`: one compound with two mass peaks;
- :func:`write_drift_h5`: calibrants present all along, the calibration drifts;
- :func:`write_run`: the instrument channels of the run check (TPS2), tiny spectra.

As TOFWERK, the buffers not written at the end of the file have zero spectra and
times, and ``NbrWrites`` counts the complete writes.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from peaxart.shapes import LEGACY, pseudo_voigt_height
from peaxart.shapes import pseudo_voigt as pv

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


def write_timing(f: h5py.File, buf_times: np.ndarray, period: float, unwritten: int = 0, attrs: bool = True):
    """``TimingData`` (``unwritten`` last buffers at time 0) and the ``NbrWrites`` / ``NbrBufs`` attributes."""
    n_writes, n_bufs = buf_times.shape
    buf_times = buf_times.copy()
    if unwritten:
        buf_times.ravel()[-unwritten:] = 0
    td = f.create_group("TimingData")
    td.create_dataset("BufTimes", data=buf_times, chunks=(1, 1))
    td.attrs["BlockPeriod"] = np.array([period * 1e9])
    if attrs:  # complete writes only
        f.attrs["NbrWrites"] = np.array([n_writes - int(np.ceil(unwritten / n_bufs))], dtype=np.int32)
        f.attrs["NbrBufs"] = np.array([n_bufs], dtype=np.int32)
        f.attrs["NbrBlocks"] = np.array([1], dtype=np.int32)


def write_h5(path, n_segments: int = 1, unwritten: int = 0, attrs: bool = True):
    rng = np.random.default_rng(0)
    rt = (np.arange(N_WRITES * N_BUFS) + 1) * S_PER_BIN
    tof = np.arange(N_POINTS, dtype=float)
    data = rng.normal(1.0, 0.3, (len(rt), N_POINTS)).astype(np.float32)
    chromato = np.exp(-((rt - RT_APEX) ** 2) / (2 * 1.5**2))
    for mz, height in PEAKS:
        mu = float(mass_to_tof(mz, CAL, 2))
        shape = pseudo_voigt(tof, mu, SIGMA_B + SIGMA_SLOPE * mu, ALPHA)
        data += (height * chromato[:, None] * shape).astype(np.float32)
    if unwritten:
        data[-unwritten:] = 0
    data = data.reshape(N_WRITES, N_BUFS, 1, N_POINTS)
    data = np.repeat(data, n_segments, axis=2) / n_segments
    with h5py.File(path, "w") as f:
        fs = f.create_group("FullSpectra")
        fs.create_dataset("TofData", data=data, chunks=(1, 1, 1, N_POINTS))
        fs.create_dataset("MassAxis", data=((tof - CAL["p2"]) / CAL["p1"]) ** 2)
        fs.attrs["MassCalibMode"] = np.array([2], dtype=np.int32)
        for k, v in CAL.items():
            fs.attrs[f"MassCalibration {k}"] = np.array([v])
        write_timing(f, rt.reshape(N_WRITES, N_BUFS), S_PER_BIN, unwritten, attrs)
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


DRIFT_N_SCANS, DRIFT_N_POINTS = 600, 9400
DRIFT_CALIBRANTS = [("A", 28.0061), ("B", 43.9898), ("C", 68.9947), ("D", 99.9931),
                    ("E", 130.9915), ("F", 168.9883), ("G", 218.9851)]
DRIFT_INTERFERENCE = (68.83, 300.0)  # (m/z, height): a weaker peak next to C


def drift_params(rt):
    """True calibration (mode 2) at rt: the tof indices shift by 0.8 over the run."""
    return {"p1": 600.0, "p2": -200.0 + 0.8 * np.asarray(rt) / (DRIFT_N_SCANS * S_PER_BIN), "p3": 0.5}


def write_drift_h5(path):
    """A run where the calibrants are present all along and the calibration drifts.

    The h5 calibration is the one of the start, off by 0.5 tof index.
    """
    rng = np.random.default_rng(1)
    n_writes = DRIFT_N_SCANS // N_BUFS
    rt = (np.arange(DRIFT_N_SCANS) + 1) * S_PER_BIN
    tof = np.arange(DRIFT_N_POINTS, dtype=float)
    data = rng.normal(1.0, 0.3, (len(rt), DRIFT_N_POINTS)).astype(np.float32)
    peaks = [(mz, 2000.0) for _, mz in DRIFT_CALIBRANTS] + [DRIFT_INTERFERENCE]
    for i, t in enumerate(rt):
        for mz, height in peaks:
            mu = float(mass_to_tof(mz, drift_params(t), 2))
            sigma = SIGMA_B + SIGMA_SLOPE * mu
            area = height / pseudo_voigt_height(1.0, sigma, ALPHA)
            data[i] += pv(tof, area, mu, sigma, ALPHA, LEGACY).astype(np.float32)
    initial = {**drift_params(0.0), "p2": -200.5}
    with h5py.File(path, "w") as f:
        fs = f.create_group("FullSpectra")
        fs.create_dataset("TofData", data=data.reshape(n_writes, N_BUFS, 1, DRIFT_N_POINTS),
                          chunks=(1, 1, 1, DRIFT_N_POINTS))
        fs.create_dataset("MassAxis", data=((tof - initial["p2"]) / initial["p1"]) ** 2)
        fs.attrs["MassCalibMode"] = np.array([2], dtype=np.int32)
        for k, v in initial.items():
            fs.attrs[f"MassCalibration {k}"] = np.array([v])
        fs.attrs["MassCalibration nbrPoints"] = np.array([len(DRIFT_CALIBRANTS)], dtype=np.int32)
        for k, (name, mz) in enumerate(DRIFT_CALIBRANTS, start=1):
            fs.attrs[f"MassCalibration m{k}"] = np.array([mz])
            fs.attrs[f"MassCalibration l{k}"] = name.encode()
        write_timing(f, rt.reshape(n_writes, N_BUFS), S_PER_BIN)
    return path


QC_WRITES, QC_BUFS, QC_DT = 100, 10, 1.0  # 1000 s run
QC_CHANNELS = ["Ionizer status monitor []", "FILAMENT_SELECT target []", "FILAMENT_EMISSION monitor [mA]",
               "FILAMENT_EMISSION target [mA]", "FILAMENT_CURRENT target [A]", "FILAMENT_CURRENT monitor [A]"]


def write_run(path, emission=None, target=None, sum_max: float = 1e8, unwritten: int = 0,
              n_writes: int = QC_WRITES, tps2: bool = True, attrs: bool = False):
    """A run with the TPS2 filament channels: ``emission`` / ``target`` functions of rt (default 0.5 mA).
    Without ``attrs``, the written scans are found from ``BufTimes``."""
    n = n_writes * QC_BUFS
    rt = np.arange(n) * QC_DT
    target_values = np.full(n, 0.5) if target is None else target(rt)
    emission_values = target_values.copy() if emission is None else emission(rt)
    tps = np.zeros((n, len(QC_CHANNELS)))
    tps[:, 0] = 5
    tps[:, 1] = 1
    tps[:, 2] = emission_values
    tps[:, 3] = target_values
    tps[:, 5] = np.where(emission_values > 0.01, 1.1, 0.003)
    if unwritten:
        tps[-unwritten:] = 0
    with h5py.File(path, "w") as f:
        f.create_dataset("FullSpectra/TofData", data=np.zeros((n_writes, QC_BUFS, 1, 4), dtype=np.float32))
        f.create_dataset("FullSpectra/SumSpectrum", data=np.array([0.0, sum_max, 1.0, 2.0]))
        write_timing(f, rt.reshape(n_writes, QC_BUFS), QC_DT, unwritten, attrs)
        if tps2:
            f.create_dataset("TPS2/TwData", data=tps.reshape(n_writes, QC_BUFS, len(QC_CHANNELS)),
                             chunks=(1, 1, len(QC_CHANNELS)))
            f.create_dataset("TPS2/TwInfo", data=np.array([c.encode() for c in QC_CHANNELS], dtype="S256"))
    return path


@pytest.fixture(scope="session")
def drift_file(tmp_path_factory):
    return write_drift_h5(tmp_path_factory.mktemp("drift") / "241007.0423.air.1.h5")


@pytest.fixture
def h5_file(tmp_path):
    path = write_h5(tmp_path / "241007.0423.air.1.h5")
    (tmp_path / "241007.0423.air.1.alpinac").mkdir()
    write_mc(tmp_path / "241007.0423.air.1.alpinac" / "241007.0423.air.1.EI_mc.txt")
    return path
