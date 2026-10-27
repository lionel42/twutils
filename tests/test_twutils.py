from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import CAL, N_POINTS, PEAKS, RT_APEX, S_PER_BIN, write_h5

import twutils
from twutils.calibration import mass_to_tof, tof_to_mass
from twutils.cli import main


@pytest.mark.parametrize("mode, params", [
    (0, {"p1": 2800.0, "p2": -3700.0}),
    (1, {"p1": -2800.0, "p2": 30000.0}),
    (2, {"p1": 2829.0, "p2": -3726.0, "p3": 0.4999}),
    (3, {"p1": 2829.0, "p2": -3726.0, "p3": 1e-4, "p4": 100.0}),
    (4, {"p1": 2829.0, "p2": -3726.0, "p3": 1e-6, "p4": 1e-3, "p5": 0.5}),
    (5, {"p1": 1e-7, "p2": 1e-3, "p3": -2.0}),
])
def test_calibration_modes_round_trip(mode, params):
    mz = np.array([12.0, 69.0, 131.0, 300.0])
    np.testing.assert_allclose(tof_to_mass(mass_to_tof(mz, params, mode), params, mode), mz, rtol=1e-6)


def test_read(h5_file):
    run = twutils.TofFile(h5_file)
    assert (run.n_scans, run.n_points, run.n_segments) == (400, N_POINTS, 1)
    assert run.s_per_bin == pytest.approx(S_PER_BIN)
    assert run.name_info["sample_type"] == "air"
    intensity, rt = run.read(10.0, 20.0, 100, 200)
    assert intensity.dtype == np.float32
    assert intensity.shape == (len(rt), 100)
    assert rt[0] >= 10.0 and rt[-1] <= 20.0 and len(rt) == 51
    full, rt_all = run.read()
    assert full.shape == (400, N_POINTS)
    i0 = int(np.searchsorted(rt_all, 10.0))
    np.testing.assert_array_equal(full[i0:i0 + len(rt), 100:200], intensity)


def test_read_segments(tmp_path):
    run = twutils.TofFile(write_h5(tmp_path / "a.h5", n_segments=2))
    with pytest.raises(ValueError, match="segments"):
        run.read(0, 10)
    both, _ = run.read(0, 10, segments=(0, 2))
    one, _ = run.read(0, 10, segments=(0, 1))
    np.testing.assert_allclose(both, 2 * one, rtol=1e-6)


def test_unwritten_scans_extrapolated(tmp_path):
    run = twutils.TofFile(write_h5(tmp_path / "a.h5", unwritten=7))
    np.testing.assert_allclose(np.diff(run.time_axis), S_PER_BIN)


def test_calibration_search(h5_file):
    cal = twutils.MassCalibration.for_file(h5_file)
    assert cal.source.endswith("EI_mc.txt")
    assert len(cal.rt) == 2 and cal.alpha == 0.5
    tof = mass_to_tof(69.0, CAL, 2)
    assert float(cal.mz(100.0, tof)) == pytest.approx(69.0)
    assert float(cal.tof(100.0, 69.0)) == pytest.approx(tof)
    np.testing.assert_allclose(cal.u_ppm_at([68.9946609, 130.9914672]), [2.5, 1.5])

    (h5_file.parent / "241007.0423.air.1.alpinac" / "241007.0423.air.1.EI_mc.txt").unlink()
    h5_cal = twutils.MassCalibration.for_file(h5_file)
    assert "h5 attributes" in h5_cal.source
    assert float(h5_cal.mz(0, tof)) == pytest.approx(69.0)


def test_rt_dependent_calibration():
    cal = twutils.MassCalibration(mode=2, rt=[0.0, 100.0], params=[[600, -200, 0.5], [600, -100, 0.5]])
    assert cal.params_at(50.0)["p2"] == pytest.approx(-150.0)
    assert cal.params_at(500.0)["p2"] == pytest.approx(-100.0)
    rt = np.array([[0.0], [50.0]])
    assert cal.mz(rt, np.arange(5)).shape == (2, 5)


def test_extract(h5_file):
    result = twutils.extract(h5_file, rt_start=30, rt_stop=50, masses=[69, 100])
    peaks = result.peaks
    assert len(peaks) == 2
    for mz, _ in PEAKS:
        row = peaks.iloc[np.argmin(np.abs(peaks["mz"] - mz))]
        assert row["mz"] == pytest.approx(mz, abs=1e-3)
        assert row["rt"] == pytest.approx(RT_APEX, abs=0.2)
    assert "compound" in peaks


def test_extract_blocks(h5_file):
    """A peak is found once when the window is processed in blocks (block edge near the apex)."""
    one = twutils.extract(h5_file, rt_start=10, rt_stop=70, masses=[69, 100]).peaks
    blocks = twutils.extract(h5_file, rt_start=10, rt_stop=70, masses=[69, 100], block=29.0, margin=8.0).peaks
    assert len(blocks) == len(one) == 2
    np.testing.assert_allclose(blocks["area"], one["area"], rtol=1e-3)


def test_extract_to_file(h5_file):
    out = twutils.extract_to_file(h5_file, rt_start=30, rt_stop=50, masses=[69, 100], metadata={"sample": "x"},
                                  peaks_csv=True)
    assert out == h5_file.parent / "241007.0423.air.1.alpinac" / "241007.0423.air.1.frag.30s50s.txt"
    text = out.read_text()
    assert "# sample: x" in text and "EI_mc.txt" in text
    table = pd.read_csv(out, sep="\t", comment="#")
    assert list(table.columns)[:3] == ["RT", "mass", "mass_u_ppm"]
    assert len(table) == 2 and set(table["Ionisation"]) == {"EI"}
    assert out.with_suffix(".peaks.csv").is_file()


def test_cli(h5_file, tmp_path, capsys):
    out = tmp_path / "out.txt"
    assert main(["extract", str(h5_file), "--rt", "40", "10", "--mass", "69", "100", "-o", str(out),
                 "--legacy", "--config", "rt_window=60", "--metadata", "operator=me"]) == 0
    text = out.read_text()
    assert "# operator: me" in text and "rt_window" in text
    assert main(["info", str(h5_file)]) == 0
    assert "mode 2" in capsys.readouterr().out
