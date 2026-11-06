# twutils

Utilities for TOFWERK GC-TOF-MS h5 files:

- `check`: is there data in a run? Filament emission (from the instrument
  housekeeping data), signal, duration, readable file;
- `TofFile`: reads the spectra (`FullSpectra/TofData`) of a retention time
  window and a tof index range, or of a whole run in blocks, the time axis, the
  instrument channels (`TPS2`) and the acquisition log. Only the written scans;
  a truncated or corrupt file raises `TofFileError`;
- `MassCalibration`: tof index <-> m/z (TOFWERK modes 0, 1, 2 and 5; m/z -> tof
  only for 3 and 4), varying with the
  retention time, and the mass peak shape. Read from the `*_mc.json` file of
  `twutils calibrate`, the `*_mc.txt` file of the legacy alpinac (read only),
  or the h5 attributes;
- `calibrate`: the mass calibration of a run, following its drift with the
  retention time (method of the legacy `alpinac.mass_calibration`, peaks
  detected and fitted with peaxart), written in a `*_mc.json` file;
- `extract`: peak extraction with [peaxart](../peaxart), written in the peak file
  format of the alpinac identification (the `.frag.txt` files of the legacy
  `alpinac_extract`).

## Install

```bash
pip install -e ../peaxart
pip install -e .
```

The m/z of calibrant formulas (`twutils calibrate`) come from the
[alpinac](../alpinac) package (`pip install -e ../alpinac`, or `.[formulas]`).
It has the same package name as the legacy alpinac of halosearch: in an
environment with the legacy one (e.g. `/shared/pyenv`, used by the
tof-processing pipeline), use `PYTHONPATH=../alpinac/src` or give the
calibrants as m/z values.

## Command line

```bash
# mass calibration with the drift -> 241007.0423.air.1.alpinac/241007.0423.air.1.seg0_mc.json
twutils -v calibrate 241007.0423.air.1.h5 -c /shared/scripts/tof-processing/config_calib.yml

# calibrants of the h5 file (PFTBA only on our instruments: m/z < 69 extrapolated), other windows
twutils calibrate "/media/Shuttle/TOF-Data/2024/2403*.air.*.h5" --averaging 240 --spacing 60
twutils calibrate run.h5 --calibrant CF3 C2F5 C3F5 N2 O2 Ar --spike 1200 C4F9 --peaks-csv

# plots of the calibration (next to the _mc.json file: <run>.seg0_mc.<plot>.png)
twutils calibrate run.h5 -c config_calib.yml --peaks-csv --plots
twutils plot-calibration run.alpinac/run.seg0_mc.json       # from the outputs (or give the h5 file)
twutils plot-calibration run.h5 --plot drift residuals --show

# rt window, all the masses of the file -> 241007.0423.air.1.alpinac/241007.0423.air.1.frag.1638s1647s.txt
twutils -v extract 241007.0423.air.1.h5 --rt-start 1638 --rt-stop 1647 -j 8

# rt 1642 +- 4.5 s, some masses, own output
twutils extract 241007.0423.air.1.h5 --rt 1642 4.5 --mass 127 142 -o CH3I.frag.txt

# whole run of several files (read in blocks of 120 s)
twutils extract "/media/Shuttle/TOF-Data/2024/2410*.air.*.h5" --mass-start 20 --mass-stop 300 -j 16

# results of the legacy alpinac extraction, other peaxart parameters
twutils extract run.h5 --rt 1642 4.5 --legacy
twutils extract run.h5 --rt 1642 4.5 --rt-sigma 1.5 --config peak_linear_ratio=0.8

# dimensions, timing, mass calibration used
twutils info run.h5

# is there data? one line per run, exit code 1 if a run is not ok
twutils check "/media/Shuttle/TOF-Data/2026/2610*.h5" --not-ok --csv qc.csv
```

`twutils extract -h` lists all options. Also `python -m twutils`.

## Python

```python
import peaxart
import twutils

# one call: extract and write the alpinac file
twutils.extract_to_file("run.h5", rt_start=1638, rt_stop=1647, n_jobs=8)

# step by step
result = twutils.extract("run.h5", rt_start=1638, rt_stop=1647, masses=range(20, 301),
                         config=peaxart.ExtractionConfig(), n_jobs=8)
result.peaks                       # peaxart table + compound column
result.to_alpinac("run.frag.txt", metadata={"sample": "tank 5"})

# mass calibration
result = twutils.calibrate("run.h5", "config_calib.yml")   # or a list, or twutils.Calibrants
result.windows                     # per rt window: calibrants used, rms ppm, p1, p2, ...
result.peaks                       # every calibrant peak: tof, sigma, alpha, ppm, used
result.calibration.drift_ppm(150, [600, 1800, 3000])  # drift since the first window
result.write(peaks_csv=True)       # <run>.alpinac/<run>.seg0_mc.json, used by extract (+ .peaks.csv)
result.plot()                      # <run>.alpinac/<run>.seg0_mc.<plot>.png; plot(write=False): figures
twutils.plots.plot_outputs("run.alpinac/run.seg0_mc.json")  # the same from the files
twutils.extract("run.h5", rt_start=1638, rt_stop=1647, calibration=result.calibration)

# reading and calibration only
run = twutils.TofFile("run.h5")
cal = twutils.MassCalibration.for_file(run)        # or .from_file(...), .from_h5(run)
intensity, rt = run.read(1638, 1647, tof_start=20000, tof_stop=21000)
mz = cal.mz(rt[:, None], range(20000, 21000))      # (n_scans, n_points)

# a whole run in blocks of whole writes, the file kept open
with run.open():
    for scans, rt, data in run.iter_blocks(block=600, tof_start=20000, tof_stop=21000):
        ...
```

See `scripts/extract_rt_window.py` for a script to edit.

## Details

- **Written scans**: TOFWERK allocates `FullSpectra/TofData` for the planned
  duration; the buffers not written (acquisition stopped early, incomplete last
  write) contain zeros. `TofFile` only uses the written scans (`run.n_scans`,
  from the `NbrWrites` complete writes, else the last non-zero `BufTimes`), so
  the default rt ranges of `extract` and `calibrate` stop at the last one. The
  time axis is `TimingData/BufTimes` (checked to increase), or
  `run.regular_time_axis` from the scan period without reading it (`BufTimes`
  is stored in one chunk per scan: ~2 s to read on a USB disk).
- **Bad files**: the commands go on with the next file when one fails. A
  truncated or corrupt file is logged in one line; any other error with its
  traceback. The exit code is 1 if a file failed.
- **Parameters**: `twutils.calibrate(run, config=CalibrationConfig(...))`, or the fields
  directly (`twutils.calibrate(run, averaging=60, mode=0)`); `twutils.check(run, QCConfig(...))`;
  `twutils.extract(run, config=peaxart.ExtractionConfig(...))`. The defaults of the command line
  are those of these classes.
- **Mass calibration with the drift** (`twutils calibrate`, legacy method):
  the run is cut in rt windows (360 s every 120 s); in each window the spectra
  are summed around each calibrant (+-0.2 m/z), the peaks detected (peaxart)
  and fitted with pseudo-Voigt functions of free sigma and alpha, and the
  calibration fitted to the calibrant peaks, leaving out those above 20 ppm.
  The peak shape (sigma vs tof, alpha) and the uncertainty per m/z come from
  the calibrant peaks used. Calibrants: formulas (`CF3`, `Re+2`, `[13C]O`) or
  m/z, or the yaml file of the tof-processing pipeline (`-c`,
  `twutils.Calibrants.from_yaml`: `"0.0"` continuous, other keys spike rts,
  `default` the initial calibration); default the calibrants of the h5 file
  (PFTBA on our instruments). The air ions (N2, O2, Ar, CO2) are needed for
  accurate masses below m/z 69. Differences with the
  legacy code: ion masses (the legacy code used neutral masses, ~8 ppm at m/z
  69, `--neutral` to reproduce it), outliers also removed by leverage, the run
  is read once (~25 s for a 50 min run). Spikes are fitted with the closest
  window, at their own rt. Modes 0, 1, 2 and 5. On 240315.2211.air.1 it agrees with
  the legacy calibration to ~1 ppm (median) once the electron mass is accounted for.
- **Calibration plots** (`twutils.plots`, needs matplotlib: `.[plots]`), as the
  legacy `make_final_plots`: `parameters` (p1, p2, ... and the residuals vs rt),
  `drift` (vs rt, of the calibration and of each calibrant), `residuals` (vs
  m/z, with the uncertainty), `peak_shape` (sigma and alpha vs m/z). The last
  two need the calibrant peaks: from a result, or the `.peaks.csv` file
  (`--peaks-csv`) when plotting from the files. Calibrants left out are hollow.
- **Run check** (`twutils check`, `twutils.check`): the filament channels are read by name from
  `TPS2/TwData` / `TPS2/TwInfo` (`FILAMENT_EMISSION monitor` / `target`, `FILAMENT_CURRENT monitor`).
  The acquisition lowers the emission target from 0.5 to 0.05 mA during parts of the run (~10-200 s,
  ~380-1100 s), so the monitor is compared with the target of each scan: below half of it, no emission.
  Stretches without emission shorter than 5 s are ignored (the monitor follows a step of the target in
  ~1 s), and so is the warm-up at the start (the filament is switched on at ~10 s and reaches its
  target ~20 s later; not ignored if longer than 60 s). A stretch until the end of the run is never
  ignored. Status: `unreadable` (truncated / corrupt file), `short` (< 600 s written), `filament_off`
  (never any emission), `filament_failed` (lost for good, with the rt), `switched_off` (target set to 0
  by the acquisition when the emission stopped: not a failure, the data before are fine), `no_signal`
  (max of `FullSpectra/SumSpectrum`
  < 1e4: ~1e8 in a normal run, ~50 with the filament off), `filament_unstable` (lost for a while), `ok`.
  Speed: `TPS2/TwData` and `TimingData/BufTimes` are stored in one chunk per scan, slow to read from
  a USB disk (~8 s per file). So the times come from the scan period (`run.regular_time_axis`), and the
  channels are read at the first scan of each write (~3 s) and at the last scan, at full resolution only
  around a failure: ~1.8 s per file on the Shuttle disk, less once cached. Files without `TPS2` are only
  checked on the signal.
- **Mass calibration file**: `<run>.alpinac/<run>.seg0_mc.json`, written by
  `twutils calibrate` (`twutils.calibration.FORMAT`), one per segment range of the
  spectra (`seg0`, `seg1`, `seg1-2` for segments 1 and 2 summed; files with one
  segment: `seg0`): the segment range, mode, rts (s), parameters,
  peak shape and uncertainty, and how it was made in `metadata` (h5 file,
  calibrants, `CalibrationConfig`, initial calibration, versions, date, and the
  result of each rt window); NaN written as null. The `*_mc.txt` files of the
  legacy alpinac are still read (not written any more: the legacy pipeline keeps
  its own calibration until it is retired).
- **Mass calibration search**: `--mass-cal` file, `<run>.alpinac/<run>.seg0_mc.json`
  (of the segments read), then the legacy files of the ionisation: `<run>.alpinac/<run>.<EI|CI>_mc.txt`, `<run>.<EI|CI>_mc.txt`,
  `<run>.mc.txt`, then the h5 attributes (with a warning: the acquisition
  calibration is not accurate enough for the identification, and the peak
  shape and calibration uncertainty are then defaults). The calibration
  parameters are interpolated at the rt of each scan.
- **Domains**: each unit mass +- `mz_range` (0.5) over the rt window, as the legacy code.
  Masses whose domain is outside the spectra are left out (with a warning if they were given).
- **Long windows** (default > 120 s, `block`) are read and extracted in blocks
  with 10 s (`margin`) of data on each side; a peak belongs to the block of its
  apex. A whole run is ~3.5 GB of float32. A window shorter than `block` is
  extracted exactly as given.
- **Files with several segments** (EI/CI instruments): the segments to read or
  calibrate are required (`--segments 1`, `--segments 1:3` sums 1 and 2); each
  segment range has its own calibration file. For the extraction, also the
  ionisation (`--ionisation CI --adduct ...`): the spectrum id of the output is
  0 for EI, 1 for CI. `twutils info` shows the calibration of each segment.
- **Output**: same columns as the legacy extraction; see the peaxart README for
  the differences of the values (`peak_width`, `peak_alpha`, `area_u`). The
  header lists the file, window, calibration file, non-default peaxart
  parameters and versions. Unlike the legacy `alpinac_extract`, several input
  files give one output file each (no merged file).
- **peaxart parameters**: `peaxart.ExtractionConfig()` by default (per-peak
  tests on top of the legacy method), `--legacy` / `ExtractionConfig.legacy()`
  reproduces the legacy alpinac.

## Development

```bash
python -m pytest                   # tests/test_real_data.py needs /media/Shuttle (skipped otherwise)
python -m pytest -m "not real_data"
```
