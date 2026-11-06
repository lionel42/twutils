from __future__ import annotations

import h5py
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
    # Not implmented yet:
    #(3, {"p1": 2829.0, "p2": -3726.0, "p3": 1e-4, "p4": 100.0}),
    #(4, {"p1": 2829.0, "p2": -3726.0, "p3": 1e-6, "p4": 1e-3, "p5": 0.5}),
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


@pytest.mark.parametrize("attrs", [True, False])
def test_unwritten_scans_left_out(tmp_path, attrs):
    """The buffers not written (zero spectra and times) are not scans of the run."""
    run = twutils.TofFile(write_h5(tmp_path / "a.h5", unwritten=7, attrs=attrs))
    # NbrWrites counts the complete writes only; without it, up to the last BufTimes
    assert run.n_scans == (390 if attrs else 393) and run.n_buffers == 400
    assert len(run.time_axis) == run.n_scans
    np.testing.assert_allclose(np.diff(run.time_axis), S_PER_BIN)
    np.testing.assert_allclose(run.regular_time_axis + run.time_axis[0], run.time_axis)
    intensity, rt = run.read()
    assert len(rt) == len(intensity) == run.n_scans and (intensity.sum(axis=1) > 0).all()
    assert run.rt_range()[1] == pytest.approx(run.time_axis[-1])
    assert len(run.read(70, 1000)[1]) == run.n_scans - int(np.searchsorted(run.time_axis, 70))


def test_buf_times_not_increasing(tmp_path, caplog):
    path = write_h5(tmp_path / "a.h5")
    with h5py.File(path, "r+") as f:
        f["TimingData/BufTimes"][5, 3] = 0  # missed buffer
    run = twutils.TofFile(path)
    np.testing.assert_allclose(np.diff(run.time_axis), S_PER_BIN)
    assert "do not increase" in caplog.text


def test_iter_blocks(h5_file):
    run = twutils.TofFile(h5_file)
    full, rt_all = run.read(tof_start=100, tof_stop=200)
    with run.open():
        blocks = list(run.iter_blocks(block=25, tof_start=100, tof_stop=200, scans=slice(13, 377)))
    assert all(b.start == 13 or b.start % run.n_bufs == 0 for b, _, _ in blocks)  # whole writes
    assert [b.stop - b.start for b, _, _ in blocks][1:-1] == [30] * (len(blocks) - 2)
    np.testing.assert_array_equal(np.concatenate([d for _, _, d in blocks]), full[13:377])
    np.testing.assert_array_equal(np.concatenate([t for _, t, _ in blocks]), rt_all[13:377])
    with pytest.raises(ValueError, match="step"):
        run.read_scans(slice(0, 10, 2))


def test_open_and_pickle(h5_file):
    import pickle

    run = twutils.TofFile(h5_file)
    with run.open():
        with run.open():  # nested
            run.read(10, 20)
        assert run._h5 is not None
        copy = pickle.loads(pickle.dumps(run))
        assert copy._h5 is None and copy.n_scans == run.n_scans
    assert run._h5 is None


def test_bad_files(tmp_path):
    truncated = tmp_path / "a.h5"
    truncated.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 88)
    with pytest.raises(twutils.TofFileError, match="cannot read"):
        twutils.TofFile(truncated)
    with h5py.File(tmp_path / "b.h5", "w") as f:
        f["x"] = 1
    with pytest.raises(twutils.TofFileError, match="no FullSpectra/TofData"):
        twutils.TofFile(tmp_path / "b.h5")
    with pytest.raises(FileNotFoundError):
        twutils.TofFile(tmp_path / "missing.h5")


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


def test_calibration_files(h5_file, tmp_path):
    """The legacy *_mc.txt is read; the json file of twutils is preferred when both exist."""
    legacy = twutils.MassCalibration.for_file(h5_file)
    assert legacy.source.endswith("EI_mc.txt") and legacy.metadata == {}
    np.testing.assert_allclose(legacy.rt, [0.0, 400.0])
    assert np.isnan(twutils.MassCalibration(mode=2, rt=[0], params=[[1, 2, 0.5]]).sigma_mz_slope)

    cal = twutils.MassCalibration(mode=2, rt=[12.345, 400.1], params=[[600, -200, 0.5], [600, -199.9, 0.5]],
                                  metadata={"note": "x"})
    path = cal.to_json(twutils.mc_path(h5_file))  # next to the legacy file
    found = twutils.MassCalibration.for_file(h5_file)
    assert found.source == str(path) and found.metadata == {"note": "x"}
    np.testing.assert_array_equal(found.rt, cal.rt)
    assert np.isnan(found.sigma_mz_slope)  # written as null
    assert twutils.MassCalibration.from_file(cal.to_json(tmp_path / "x.txt")).rt[0] == 12.345  # by content

    path.write_text('{"format": "twutils.mass_calibration", "version": 99}')
    with pytest.raises(ValueError, match="newer"):
        twutils.MassCalibration.from_json(path)


def test_calibration_of_segments(tmp_path, caplog):
    """Files with several segments: one calibration file per segment range."""
    run = twutils.TofFile(write_h5(tmp_path / "241007.0423.air.1.h5", n_segments=2))
    with pytest.raises(ValueError, match="2 segments"):
        twutils.MassCalibration.for_file(run)
    for k in range(2):
        twutils.MassCalibration(mode=2, rt=[0.0], params=[[600, -200 + k, 0.5]], segments=(k, k + 1)).to_json(
            twutils.mc_path(run, (k, k + 1)))
    for k in range(2):
        cal = twutils.MassCalibration.for_file(run, (k, k + 1))
        assert cal.segments == (k, k + 1) and cal.params[0, 1] == -200 + k
        assert cal.source.endswith(f"seg{k}_mc.json")
    # no file for the segments 0 and 1 summed: the h5 calibration
    assert "h5 attributes" in twutils.MassCalibration.for_file(run, (0, 2)).source
    # a calibration of other segments: warning
    twutils.MassCalibration.for_file(run, (0, 1), mc_file=twutils.mc_path(run, (1, 2)))
    assert "segments (1, 2) used for the segments (0, 1)" in caplog.text


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


def test_extract_masses_outside(h5_file, caplog):
    result = twutils.extract(h5_file, rt_start=30, rt_stop=50, masses=[69, 100, 500])
    assert result.masses == [69.0, 100.0] and "outside the spectra left out: 500" in caplog.text
    with pytest.raises(ValueError, match="No mass to extract"):
        twutils.extract(h5_file, rt_start=30, rt_stop=50, masses=[500])


def test_default_output(h5_file):
    names = [twutils.default_output(h5_file, *rts).name for rts in [(30, 50), (30, None), (None, 50), (None, None)]]
    assert names == ["241007.0423.air.1.frag.30s50s.txt", "241007.0423.air.1.frag.30s_end.txt",
                     "241007.0423.air.1.frag.0s50s.txt", "241007.0423.air.1.frag.all_rts.txt"]
    directory = h5_file.parent / "241007.0423.air.1.alpinac"
    assert twutils.mc_path(h5_file) == directory / "241007.0423.air.1.seg0_mc.json"
    assert twutils.mc_path(h5_file, (1, 3)) == directory / "241007.0423.air.1.seg1-2_mc.json"


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


def test_cli_continues_after_bad_file(h5_file, tmp_path, capsys, caplog):
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not an h5 file")
    assert main(["info", str(bad), str(h5_file)]) == 1
    assert "mode 2" in capsys.readouterr().out  # the good file after the bad one
    assert "Info failed: bad.h5" in caplog.text and "cannot read" in caplog.text and "Traceback" not in caplog.text
    with pytest.raises(SystemExit, match="--rt cannot"):
        main(["extract", str(h5_file), "--rt", "40", "10", "--rt-start", "30"])


def test_cli_config_values():
    from twutils.cli import _parse_value

    assert _parse_value("x", "1.5", "float | None") == 1.5 and _parse_value("x", "none", "float | None") is None
    assert _parse_value("x", "3", "int") == 3 and _parse_value("x", "yes", "bool") is True
    with pytest.raises(SystemExit, match="invalid value"):
        _parse_value("x", "maybe", "bool")


def test_cli(h5_file, tmp_path, capsys):
    out = tmp_path / "out.txt"
    assert main(["extract", str(h5_file), "--rt", "40", "10", "--mass", "69", "100", "-o", str(out),
                 "--legacy", "--config", "rt_window=60", "--metadata", "operator=me"]) == 0
    text = out.read_text()
    assert "# operator: me" in text and "rt_window" in text
    assert main(["info", str(h5_file)]) == 0
    assert "mode 2" in capsys.readouterr().out
    two = write_h5(tmp_path / "two.h5", n_segments=2)
    assert main(["info", str(two)]) == 0  # each segment
    out = capsys.readouterr().out
    assert "mass cal seg0" in out and "mass cal seg1" in out
