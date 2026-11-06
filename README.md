# twutils

Utilities for TOFWERK GC-TOF-MS h5 files:

- `check`: is there data in a run? Filament emission (from the instrument
  housekeeping data), signal, duration, readable file;
- `TofFile`: reads the spectra (`FullSpectra/TofData`) of a retention time
  window and a tof index range, or of a whole run in blocks, the time axis, the
  instrument channels (`TPS2`) and the acquisition log. Only the written scans;
  a truncated or corrupt file raises `TofFileError`;
- `calibration_for`: the mass calibration of a run (a `masscal.MassCalibration`:
  tof index <-> m/z varying with the retention time, mass peak shape), from its
  `*_mc.json` file, the `*_mc.txt` file of the legacy alpinac, or the h5 attributes.
  The calibration is fitted by `esper calibrate` ([masscal](../masscal));
- `extract`: peak extraction with [peaxart](../peaxart), written in the peak file
  format of the alpinac identification (the `.frag.txt` files of the legacy
  `alpinac_extract`);

The unit mass chromatograms are made by `esper chromatograms` (esper, peaxart).

## Install

```bash
pip install -e ../peaxart -e ../masscal
pip install -e .
```

## Command line

```bash
# (the mass calibration: esper calibrate run.h5, see esper)

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

# reading and calibration only
run = twutils.TofFile("run.h5")
cal = twutils.calibration_for(run)                 # masscal.MassCalibration; or twutils.h5_calibration(run)
run.start_time                                     # acquisition start (rt 0), UTC to the ms
intensity, rt = run.read(1638, 1647, tof_start=20000, tof_stop=21000)
mz = cal.mz(rt[:, None], range(20000, 21000))      # (n_scans, n_points)

# a whole run in blocks of whole writes, the file kept open
with run.open():
    for scans, rt, data in run.iter_blocks(block=600, tof_start=20000, tof_stop=21000):
        ...

# spectra summed over scan ranges at some tof indices (read once, in blocks)
sums = run.sum_spectra([(0, 1800), (600, 2400)], columns=[11230, 11231, 11232])
```

See `scripts/extract_rt_window.py` for a script to edit.

## Details

- **Written scans**: TOFWERK allocates `FullSpectra/TofData` for the planned
  duration; the buffers not written (acquisition stopped early, incomplete last
  write) contain zeros. `TofFile` only uses the written scans (`run.n_scans`,
  from the `NbrWrites` complete writes, else the last non-zero `BufTimes`), so
  the default rt ranges of `extract` and of the calibration stop at the last one. The
  time axis is `TimingData/BufTimes` (checked to increase), or
  `run.regular_time_axis` from the scan period without reading it (`BufTimes`
  is stored in one chunk per scan: ~2 s to read on a USB disk).
- **Bad files**: the commands go on with the next file when one fails. A
  truncated or corrupt file is logged in one line; any other error with its
  traceback. The exit code is 1 if a file failed.
- **Parameters**: `twutils.check(run, QCConfig(...))`;
  `twutils.extract(run, config=peaxart.ExtractionConfig(...))`. The defaults of the command line
  are those of these classes.
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
- **Mass calibration file**: `<run>.alpinac/<run>.seg0_mc.json` (`twutils.mc_path`), written by
  `esper calibrate` (format `masscal.FORMAT`), one per segment range of the
  spectra (`seg0`, `seg1`, `seg1-2` for segments 1 and 2 summed; files with one
  segment: `seg0`). The `*_mc.txt` files of the legacy alpinac are still read.
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
- **Files with several segments** (EI/CI instruments): the segments to read
  are required (`--segments 1`, `--segments 1:3` sums 1 and 2); each
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
python -m pytest
```
