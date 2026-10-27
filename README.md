# twutils

Utilities for TOFWERK GC-TOF-MS h5 files:

- `TofFile`: reads the spectra (`FullSpectra/TofData`) of a retention time
  window and a tof index range, and the time axis;
- `MassCalibration`: tof index <-> m/z (TOFWERK modes 0-5), varying with the
  retention time, and the mass peak shape. Read from the `*_mc.txt` file of the
  alpinac mass calibration, or from the h5 attributes;
- `extract`: peak extraction with [peaxart](../peaxart), written in the peak file
  format of the alpinac identification (the `.frag.txt` files of the legacy
  `alpinac_extract`).

## Install

```bash
pip install -e ../peaxart
pip install -e .
```

## Command line

```bash
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
cal = twutils.MassCalibration.for_file(run)        # or .from_mc_file(...), .from_h5(run)
intensity, rt = run.read(1638, 1647, tof_start=20000, tof_stop=21000)
mz = cal.mz(rt[:, None], range(20000, 21000))      # (n_scans, n_points)
```

See `scripts/extract_rt_window.py` for a script to edit.

## Details

- **Mass calibration**: searched as in the legacy alpinac:
  `--mass-cal` file, `<run>.alpinac/<run>.<EI|CI>_mc.txt`, `<run>.<EI|CI>_mc.txt`,
  `<run>.mc.txt`, then the h5 attributes (with a warning: the acquisition
  calibration is not accurate enough for the identification, and the peak
  shape and calibration uncertainty are then defaults). The calibration
  parameters are interpolated at the rt of each scan.
- **Domains**: each unit mass +- `mz_range` (0.5) over the rt window, as the legacy code.
- **Long windows** (default > 120 s, `block`) are read and extracted in blocks
  with 10 s (`margin`) of data on each side; a peak belongs to the block of its
  apex. A whole run is ~3.5 GB of float32. A window shorter than `block` is
  extracted exactly as given.
- **EI/CI instruments**: give the segments to read (`--segments 0:1`) and the
  ionisation (`--ionisation CI --adduct ...`); the spectrum id of the output is
  0 for EI, 1 for CI.
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
