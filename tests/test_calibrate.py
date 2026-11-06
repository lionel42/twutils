from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    ALPHA,
    DRIFT_CALIBRANTS,
    SIGMA_B,
    SIGMA_SLOPE,
    drift_params,
)

import twutils
from twutils.calibrate import (
    calibration_windows,
    fit_calibration,
    initial_params,
    resolve_calibrants,
)
from twutils.calibration import mass_to_tof
from twutils.cli import main

MZ = np.array([mz for _, mz in DRIFT_CALIBRANTS])
WINDOWS = {"averaging": 30.0, "spacing": 15.0}


def test_resolve_calibrants():
    pytest.importorskip("alpinac.formula", reason="needs the alpinac of gitlabs/alpinac")
    (cf3, cf3_mz), (re, re_mz), (_, o2_mz), number, named = resolve_calibrants(
        ["CF3", "Re+2", "O[18O]", 100.5, ("x", 12.0)])
    assert (cf3, re) == ("CF3", "Re+2")
    assert cf3_mz == pytest.approx(68.99466091, abs=1e-7)  # TOFWERK value
    assert re_mz == pytest.approx(93.4773265)
    assert o2_mz == pytest.approx(15.9949146223 + 17.9991604 - 0.00054858)
    assert number == ("100.5000", 100.5) and named == ("x", 12.0)
    # legacy: neutral masses, except explicit charges
    neutral = dict(resolve_calibrants(["CF3", "Re+2"], ionised=False))
    assert neutral == {"CF3": pytest.approx(68.99520948, abs=1e-7), "Re+2": pytest.approx(re_mz)}


def test_calibrants_from_yaml(tmp_path):
    path = tmp_path / "config_calib.yml"
    path.write_text("""
"default":
  "mode": 2
  "p1": 2829.1
  "p2": -3731.25
  "p3": 0.49994
# continuous
"0.0":
  - F
  - "[13C]O"
  - "Re+2"
"1200.5":
  - C4F9
""")
    cal = twutils.Calibrants.from_yaml(path)
    assert cal.continuous == ["F", "[13C]O", "Re+2"]
    assert cal.spikes == {1200.5: ["C4F9"]}
    assert cal.initial.mode == 2 and cal.initial.params_at(0)["p3"] == 0.49994
    path.write_text('"0.0": [CF3, 28.0061]\n')
    cal = twutils.Calibrants.from_yaml(path)
    assert cal.continuous == ["CF3", 28.0061] and cal.spikes == {} and cal.initial is None
    for key in ["0", "0.0", "'0'"]:
        path.write_text(f'{key}: [CF3]\n1200: [C4F9]\n')
        cal = twutils.Calibrants.from_yaml(path)
        assert cal.continuous == ["CF3"] and cal.spikes == {1200.0: ["C4F9"]}


def test_calibration_windows():
    t = np.arange(1, 1001) * 0.5
    w = calibration_windows(t, averaging=60, spacing=20)
    assert w["rt_start"].iloc[0] == t[0] and w["rt_stop"].iloc[-1] == pytest.approx(t[-1])
    np.testing.assert_allclose(np.diff(w["rt"]), np.diff(w["rt"])[0])
    assert 19 < np.diff(w["rt"])[0] < 21
    assert (w["scan_stop"] - w["scan_start"]).between(120, 121).all()
    assert len(calibration_windows(t, averaging=1000, spacing=20)) == 1


@pytest.mark.parametrize("mode, params", [
    (0, [2800.0, -3700.0]),
    (1, [-2800.0, 30000.0]),
    (2, [2829.0, -3726.0, 0.4999]),
    (5, [1e-7, 1e-3, -2.0]),
])
def test_fit_calibration(mode, params):
    mz = np.array([19.0, 28.0, 44.0, 69.0, 100.0, 131.0, 169.0, 219.0, 293.0])
    tof = mass_to_tof(mz, {f"p{k + 1}": v for k, v in enumerate(params)}, mode)
    bad = tof.copy()
    bad[4] = float(mass_to_tof(mz[4] * (1 + 100e-6), {f"p{k + 1}": v for k, v in enumerate(params)}, mode))
    p0 = initial_params(mode, tof, mz)
    _, ppm, used = fit_calibration(bad, mz, mode, p0, max_ppm=20.0)
    assert not used[4] and used.sum() == len(mz) - 1
    assert np.abs(ppm[used]).max() < 0.01
    assert ppm[4] == pytest.approx(100, abs=1)


@pytest.fixture(scope="module")
def drift_result(drift_file):
    return twutils.calibrate(drift_file, **WINDOWS)


def test_calibrate_drift(drift_result):
    """The drift of the synthetic run is followed (the h5 calibration is off by 0.5 tof index)."""
    result, cal = drift_result, drift_result.calibration
    assert result.windows["ok"].all() and len(result.windows) == 7
    assert (result.windows["n_used"] == len(DRIFT_CALIBRANTS)).all()
    rt = cal.rt[:, None]
    true_tof = mass_to_tof(MZ[None, :], drift_params(rt), 2)
    assert np.abs((cal.mz(rt, true_tof) - MZ) / MZ * 1e6).max() < 0.5
    # drift between the first and last window, as the truth
    first, last = cal.rt[0], cal.rt[-1]
    true_drift = (mass_to_tof(MZ, drift_params(last), 2) - mass_to_tof(MZ, drift_params(first), 2))
    np.testing.assert_allclose(cal.tof(last, MZ) - cal.tof(first, MZ), true_drift, atol=0.01)
    assert np.abs(cal.drift_ppm(150.0, cal.rt)).max() > 50
    # the calibrant C is selected, not the weaker peak next to it
    c = result.peaks[(result.peaks["calibrant"] == "C") & result.peaks["selected"]]
    assert len(c) == 7 and (np.abs(c["tof"] - mass_to_tof(68.9947, drift_params(c["rt"]), 2)) < 0.05).all()
    assert ((result.peaks["calibrant"] == "C") & ~result.peaks["selected"]).sum() == 7
    # peak shape and uncertainty
    assert cal.alpha == pytest.approx(ALPHA, abs=0.03)
    np.testing.assert_allclose(cal.sigma([2000, 8000]), SIGMA_B + SIGMA_SLOPE * np.array([2000, 8000]), rtol=0.03)
    np.testing.assert_allclose(cal.u_masses, MZ)
    assert (cal.u_ppm < 1).all()


def test_calibrate_write(drift_result, drift_file):
    path = drift_result.write(peaks_csv=True)
    assert path == drift_file.parent / "241007.0423.air.1.alpinac" / "241007.0423.air.1.seg0_mc.json"
    assert path.with_suffix(".peaks.csv").is_file()
    cal = twutils.MassCalibration.for_file(drift_file)
    assert cal.source == str(path)
    # exact: rts in s, full precision
    written = drift_result.calibration
    np.testing.assert_array_equal(cal.rt, written.rt)
    np.testing.assert_array_equal(cal.params, written.params)
    np.testing.assert_array_equal(cal.u_ppm, written.u_ppm)
    assert (cal.sigma_mz_slope, cal.alpha) == (written.sigma_mz_slope, written.alpha)
    # how it was made
    source = cal.metadata["source"]
    assert source["file"] == str(drift_file.absolute()) and source["config"]["averaging"] == WINDOWS["averaging"]
    assert [c["mz"] for c in source["calibrants"]] == list(MZ) and "twutils" in source["versions"]
    assert len(cal.metadata["windows"]) == len(drift_result.windows) and cal.segments == (0, 1)
    assert "NaN" not in path.read_text()  # valid json
    # the masses are right at the end of the run (the h5 calibration is ~100 ppm off)
    rt_end = cal.rt[-1]
    true_tof = mass_to_tof(MZ, drift_params(rt_end), 2)
    assert np.abs((cal.mz(rt_end, true_tof) - MZ) / MZ * 1e6).max() < 1
    h5 = twutils.MassCalibration.from_h5(drift_file)
    assert np.abs((h5.mz(rt_end, true_tof) - MZ) / MZ * 1e6).max() > 50
    path.unlink()


def test_calibrate_spikes(drift_file):
    result = twutils.calibrate(drift_file, calibrants=[], spikes={60.0: DRIFT_CALIBRANTS})
    assert list(result.windows["rt"]) == [60.0] and result.windows["ok"].all()
    true_tof = mass_to_tof(MZ, drift_params(60.0), 2)
    assert np.abs((result.calibration.mz(60.0, true_tof) - MZ) / MZ * 1e6).max() < 0.5


def test_calibrate_spike_keeps_its_rt(drift_file):
    """A spike is fitted with the closest window, at its own rt."""
    spike = ("spike", DRIFT_CALIBRANTS[3][1])
    result = twutils.calibrate(drift_file, calibrants=DRIFT_CALIBRANTS, spikes={61.3: [spike]}, **WINDOWS)
    peaks = result.peaks[result.peaks["calibrant"] == "spike"]
    assert len(peaks) and (peaks["rt"] == 61.3).all()


def test_calibrate_arguments(drift_file, drift_result):
    with pytest.raises(NotImplementedError):
        twutils.calibrate(drift_file, mode=3)
    with pytest.raises(TypeError):
        twutils.calibrate(drift_file, averagin=30)  # typo: not a CalibrationConfig field
    assert drift_result.config == twutils.CalibrationConfig(**WINDOWS)
    pytest.importorskip("alpinac.formula", reason="needs the alpinac of gitlabs/alpinac")
    # a single formula, not a yaml file
    with pytest.raises((RuntimeError, ValueError), match="calibra"):
        twutils.calibrate(drift_file, calibrants="CF3", **WINDOWS)


def test_calibrate_several_segments(tmp_path):
    from conftest import write_h5

    run = write_h5(tmp_path / "a.h5", n_segments=2)
    with pytest.raises(ValueError, match="2 segments"):  # which segments: not guessed
        twutils.calibrate(run, calibrants=[69.0, 100.0])


def test_calibrate_too_few_calibrants(drift_file):
    with pytest.raises(RuntimeError, match="failed in all windows"):
        twutils.calibrate(drift_file, calibrants=DRIFT_CALIBRANTS[:3], **WINDOWS)


def test_cli_calibrate(drift_file, tmp_path, capsys):
    out = tmp_path / "run.EI_mc.json"
    args = ["calibrate", str(drift_file), "--averaging", "60", "--spacing", "30", "-o", str(out)]
    assert main(args + ["--calibrant", *[f"{mz}" for mz in MZ]]) == 0
    assert "3/3 windows" in capsys.readouterr().out
    assert len(twutils.MassCalibration.from_file(out).rt) == 3
    assert main(["calibrate", str(drift_file), "--averaging", "60", "--spacing", "30", "--output-dir",
                 str(tmp_path / "cal"), "--calibrant", *[f"{mz}" for mz in MZ]]) == 0
    assert (tmp_path / "cal" / "241007.0423.air.1.seg0_mc.json").is_file()


def test_calibrate_plots(drift_result, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from twutils.plots import PLOTS

    figures = drift_result.plot(write=False)
    assert list(figures) == list(PLOTS)
    path = drift_result.write(tmp_path / "run.EI_mc.json")
    pngs = drift_result.plot(path)
    assert all(png.is_file() and png.name == f"run.EI_mc.{name}.png" for name, png in pngs.items())


def test_cli_plot_calibration(drift_result, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    path = drift_result.write(tmp_path / "run.EI_mc.json")
    # without the peaks: the plots of the calibration only
    assert main(["plot-calibration", str(path)]) == 0
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["run.EI_mc.drift.png", "run.EI_mc.parameters.png"]
    drift_result.write(path, peaks_csv=True)
    assert main(["plot-calibration", str(path), "--plot", "residuals", "peak_shape"]) == 0
    assert "run.EI_mc.peak_shape.png" in capsys.readouterr().out
    assert len(list(tmp_path.glob("*.png"))) == 4
