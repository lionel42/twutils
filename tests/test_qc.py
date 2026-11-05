"""Run check: filament emission, signal, duration, unreadable files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import twutils
from twutils.cli import main
from twutils.qc import QCConfig, check

N_WRITES, N_BUFS, DT = 100, 10, 1.0  # 1000 s run
CHANNELS = ["Ionizer status monitor []", "FILAMENT_SELECT target []", "FILAMENT_EMISSION monitor [mA]",
            "FILAMENT_EMISSION target [mA]", "FILAMENT_CURRENT target [A]", "FILAMENT_CURRENT monitor [A]"]


def write_run(path: Path, emission=None, target=None, sum_max: float = 1e8, unwritten: int = 0,
              n_writes: int = N_WRITES, tps2: bool = True, attrs: bool = False) -> Path:
    """Small run with the TPS2 channels: ``emission`` / ``target`` functions of rt (default 0.5 mA)."""
    n = n_writes * N_BUFS
    rt = np.arange(n) * DT
    target_values = np.full(n, 0.5) if target is None else target(rt)
    emission_values = target_values.copy() if emission is None else emission(rt)
    current = np.where(emission_values > 0.01, 1.1, 0.003)
    tps = np.zeros((n, len(CHANNELS)))
    tps[:, 0] = 5
    tps[:, 1] = 1
    tps[:, 2] = emission_values
    tps[:, 3] = target_values
    tps[:, 5] = current
    buf_times = rt.copy()
    if unwritten:
        buf_times[-unwritten:] = 0
        tps[-unwritten:] = 0
    with h5py.File(path, "w") as f:
        f.create_dataset("FullSpectra/TofData", data=np.zeros((n_writes, N_BUFS, 1, 4), dtype=np.float32))
        f.create_dataset("FullSpectra/SumSpectrum", data=np.array([0.0, sum_max, 1.0, 2.0]))
        f.create_dataset("TimingData/BufTimes", data=buf_times.reshape(n_writes, N_BUFS))
        f["TimingData"].attrs["BlockPeriod"] = np.array([DT * 1e9])
        if attrs:  # as written by TOFWERK: complete writes only
            f.attrs["NbrWrites"] = np.array([n_writes - int(np.ceil(unwritten / N_BUFS))], dtype=np.int32)
            f.attrs["NbrBufs"] = np.array([N_BUFS], dtype=np.int32)
            f.attrs["NbrBlocks"] = np.array([1], dtype=np.int32)
        if tps2:
            f.create_dataset("TPS2/TwData", data=tps.reshape(n_writes, N_BUFS, len(CHANNELS)))
            f.create_dataset("TPS2/TwInfo", data=np.array([c.encode() for c in CHANNELS], dtype="S256"))
    return path


def lowered_target(rt):
    """As the acquisition does: 0.05 mA between 10 and 200 s and 380 and 900 s, 0.5 mA otherwise."""
    return np.where(((rt > 10) & (rt < 200)) | ((rt > 380) & (rt < 900)), 0.05, 0.5)


def test_ok(tmp_path):
    qc = check(write_run(tmp_path / "241007.0423.air.1.h5", unwritten=7))
    assert qc.ok and qc.status == "ok"
    assert qc.n_scans == N_WRITES * N_BUFS - 7  # unwritten buffers left out
    assert qc.emission_ratio == pytest.approx(1.0)
    assert qc.name == "241007.0423.air.1"
    assert "ok" in str(qc)


def test_lowered_target_is_ok(tmp_path):
    qc = check(write_run(tmp_path / "a.h5", target=lowered_target))
    assert qc.ok
    assert qc.fraction_no_emission == 0


def test_filament_off(tmp_path):
    qc = check(write_run(tmp_path / "a.h5", emission=lambda rt: np.zeros_like(rt), sum_max=53))
    assert qc.status == "filament_off" and not qc.ok
    assert qc.emission_ratio == 0


def test_times_from_attributes(tmp_path):
    qc = check(write_run(tmp_path / "a.h5", unwritten=7, attrs=True))
    assert qc.ok and qc.n_scans == (N_WRITES - 1) * N_BUFS  # the incomplete last write left out
    assert qc.duration == pytest.approx(((N_WRITES - 1) * N_BUFS - 1) * DT)


def test_emission_switched_off(tmp_path):
    qc = check(write_run(tmp_path / "a.h5", target=lambda rt: np.zeros_like(rt), sum_max=53))
    assert qc.status == "switched_off" and not qc.ok
    assert "target 0" in qc.message


def test_switched_off_during_run(tmp_path):
    # as 240522.1607.std.7: the acquisition sets the target to 0, not a failure of the filament
    qc = check(write_run(tmp_path / "a.h5", target=lambda rt: np.where(rt < 600, 0.5, 0.0)))
    assert qc.status == "switched_off" and qc.rt_failure == pytest.approx(600)


def test_monitor_lag_after_step_is_ok(tmp_path):
    # the monitor follows a step of the target 0.05 -> 0.5 mA in ~1 s (5 scans at 6 Hz)
    def emission(rt):
        return np.where((rt >= 200) & (rt < 201), 0.05, lowered_target(rt))

    assert check(write_run(tmp_path / "a.h5", target=lowered_target, emission=emission)).ok
    assert check(write_run(tmp_path / "b.h5", target=lowered_target, emission=emission),
                 QCConfig(min_gap=0)).status == "filament_unstable"


def test_filament_failed(tmp_path):
    # broken during a lowered-target period, as 261031.1617.tank.9
    path = write_run(tmp_path / "a.h5", target=lowered_target, unwritten=5,
                     emission=lambda rt: np.where(rt < 196.5, lowered_target(rt), 0.0))
    qc = check(path)
    assert qc.status == "filament_failed"
    assert qc.rt_failure == pytest.approx(197.0)
    assert "197.0" in qc.message


def test_filament_unstable(tmp_path):
    qc = check(write_run(tmp_path / "a.h5", emission=lambda rt: np.where((rt > 300) & (rt < 350), 0.0, 0.5)))
    assert qc.status == "filament_unstable"
    # rt 301-349 s, sampled at the first scan of each write (every 10 s here)
    assert qc.fraction_no_emission == pytest.approx(0.045, abs=0.01)


def test_switched_on_after_start_is_ok(tmp_path):
    # as 240425.1141.tank.3: target 0 during the first ~30 s of the acquisition
    def target(rt):
        return np.where(rt < 32, 0.0, 0.5)

    assert check(write_run(tmp_path / "a.h5", target=target, emission=target)).ok
    # then warm-up: target 0.05 mA from 12 s, emission reached at 32 s
    target2 = lambda rt: np.where(rt < 12, 0.0, 0.05)  # noqa: E731
    emission2 = lambda rt: np.where(rt < 32, 0.0, 0.05)  # noqa: E731
    assert check(write_run(tmp_path / "b.h5", target=target2, emission=emission2)).ok
    late = lambda rt: np.where(rt < 120, 0.0, 0.05)  # noqa: E731
    assert check(write_run(tmp_path / "c.h5", target=target2, emission=late)).status == "filament_unstable"


def test_no_signal(tmp_path):
    assert check(write_run(tmp_path / "a.h5", sum_max=50)).status == "no_signal"


def test_short(tmp_path):
    qc = check(write_run(tmp_path / "a.h5", n_writes=4))
    assert qc.status == "short"
    assert check(write_run(tmp_path / "b.h5", n_writes=4), QCConfig(min_duration=10)).ok


def test_without_tps2(tmp_path):
    assert check(write_run(tmp_path / "a.h5", tps2=False)).ok
    assert check(write_run(tmp_path / "b.h5", tps2=False, sum_max=50)).status == "no_signal"


def test_unreadable(tmp_path):
    bad = tmp_path / "a.h5"
    bad.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 88)  # truncated
    assert check(bad).status == "unreadable"
    with h5py.File(tmp_path / "b.h5", "w") as f:
        f["x"] = 1
    assert check(tmp_path / "b.h5").status == "unreadable"


def test_cli(tmp_path, capsys):
    ok = write_run(tmp_path / "ok.h5")
    off = write_run(tmp_path / "off.h5", emission=lambda rt: np.zeros_like(rt))
    assert main(["check", str(ok)]) == 0
    csv = tmp_path / "qc.csv"
    assert main(["check", str(tmp_path / "*.h5"), "--not-ok", "--csv", str(csv)]) == 1
    out = capsys.readouterr().out
    assert "filament_off" in out and "ok.h5" not in out.split("\n")[-2]
    import pandas as pd

    table = pd.read_csv(csv, index_col="name")
    assert table.loc["off", "status"] == "filament_off" and table.loc["ok", "status"] == "ok"
    assert off.is_file()


REAL = Path("/media/Shuttle/TOF-Data/2026")


@pytest.mark.real_data
@pytest.mark.skipif(not (REAL / "261031.1447.std.3.h5").is_file(), reason="needs /media/Shuttle")
def test_real_broken_filament():
    """The filament broke on 2026-10-31 at 196.6 s of 261031.1617.tank.9."""
    assert twutils.check(REAL / "261031.1447.std.3.h5").ok
    failed = twutils.check(REAL / "261031.1617.tank.9.h5")
    assert failed.status == "filament_failed" and failed.rt_failure == pytest.approx(196.6, abs=0.2)
    assert twutils.check(REAL / "261031.1747.std.3.h5").status == "filament_off"
