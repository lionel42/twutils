"""Command line: ``twutils extract``, ``twutils calibrate``, ``twutils plot-calibration`` and ``twutils info``."""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

logger = logging.getLogger("twutils.cli")


def _files(patterns: list[str]) -> list[Path]:
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matches:
            logger.warning(f"No file matches '{pattern}'")
        files += [Path(m) for m in matches]
    return files


def _key_value(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got '{text}'")
    return key.strip(), value.strip()


def _segments(text: str) -> tuple[int, int]:
    start, sep, stop = text.partition(":")
    try:
        return (int(start), int(stop)) if sep else (int(start), int(start) + 1)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected START:STOP or INDEX, got '{text}'") from None


def add_extract_parser(sub) -> None:
    p = sub.add_parser("extract", help="Extract the peaks of h5 files for the alpinac identification.",
                       description="Extract the peaks of h5 files with peaxart and write the alpinac "
                                   "peak file (<run>.alpinac/<run>.frag.<start>s<stop>s.txt by default).")
    p.add_argument("files", nargs="+", help="h5 files (glob patterns allowed).")
    p.add_argument("--rt-start", type=float, help="Start of the retention time window, s (default: start of the run).")
    p.add_argument("--rt-stop", type=float, help="Stop of the retention time window, s (default: end of the run).")
    p.add_argument("--rt", type=float, nargs=2, metavar=("RT", "HALF_WIDTH"),
                   help="Window RT +- HALF_WIDTH s (instead of --rt-start/--rt-stop).")
    p.add_argument("--mass", type=float, nargs="+", help="Unit masses to extract (default: --mass-start to --mass-stop).")
    p.add_argument("--mass-start", type=float, help="First unit mass (default: first of the file).")
    p.add_argument("--mass-stop", type=float, help="Last unit mass (default: last of the file).")
    p.add_argument("--mz-range", type=float, default=0.5, help="Half width of the domain around each unit mass (default 0.5).")
    p.add_argument("--mass-cal", type=Path, help="Mass calibration file (*_mc.txt). Default: searched next to the h5 file.")
    p.add_argument("--ionisation", choices=["EI", "CI"], default="EI", help="Ionisation (default EI).")
    p.add_argument("--segments", type=_segments, help="Segments to read, START:STOP (EI/CI instruments).")
    p.add_argument("--adduct", default="", help="Adduct of CI spectra.")
    p.add_argument("--legacy", action="store_true", help="Reproduce the legacy alpinac extraction (peaxart.ExtractionConfig.legacy()).")
    p.add_argument("--rt-sigma", type=float, help="Expected chromatographic sigma, s (default: legacy model of rt).")
    p.add_argument("--config", type=_key_value, action="append", default=[], metavar="KEY=VALUE",
                   help="Other peaxart.ExtractionConfig parameter (repeatable), e.g. --config peak_linear_ratio=0.8.")
    p.add_argument("-j", "--jobs", type=int, default=1, help="Domains extracted in parallel.")
    p.add_argument("--block", type=float, default=120.0, help="Long windows are processed in blocks of this many s (default 120).")
    p.add_argument("-o", "--output", type=Path, help="Output file (one input file only).")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: <run>.alpinac next to the h5 file).")
    p.add_argument("--metadata", type=_key_value, action="append", default=[], metavar="KEY=VALUE",
                   help="Added to the header of the output file (repeatable).")
    p.add_argument("--peaks-csv", action="store_true", help="Also write the full peaxart peak table (.peaks.csv).")
    p.set_defaults(func=run_extract)


def _config(args):
    import dataclasses

    import peaxart

    fields = {f.name: f for f in dataclasses.fields(peaxart.ExtractionConfig)}
    changes = {}
    for key, text in args.config:
        if key not in fields:
            raise SystemExit(f"Unknown peaxart.ExtractionConfig parameter '{key}'")
        default = getattr(peaxart.ExtractionConfig(), key)
        if text.lower() == "none":
            changes[key] = None
        elif isinstance(default, bool):
            changes[key] = text.lower() in ("1", "true", "yes")
        elif isinstance(default, int):
            changes[key] = int(text)
        elif isinstance(default, float) or default is None:
            changes[key] = float(text)
        else:
            changes[key] = text
    if args.rt_sigma is not None:
        changes["rt_sigma"] = args.rt_sigma
    return peaxart.ExtractionConfig.legacy(**changes) if args.legacy else peaxart.ExtractionConfig(**changes)


def run_extract(args) -> int:
    from twutils.extract import extract_to_file

    files = _files(args.files)
    if not files:
        return 1
    if args.output is not None and len(files) > 1:
        raise SystemExit("--output can only be used with one file (use --output-dir).")
    rt_start, rt_stop = args.rt_start, args.rt_stop
    if args.rt is not None:
        rt_start, rt_stop = args.rt[0] - args.rt[1], args.rt[0] + args.rt[1]
    config = _config(args)
    failed = 0
    for path in files:
        try:
            out = extract_to_file(
                path, output=args.output, output_dir=args.output_dir, metadata=dict(args.metadata),
                adduct=args.adduct, peaks_csv=args.peaks_csv,
                rt_start=rt_start, rt_stop=rt_stop, masses=args.mass, mass_start=args.mass_start,
                mass_stop=args.mass_stop, mz_range=args.mz_range,
                calibration=args.mass_cal, ionisation=args.ionisation, segments=args.segments,
                config=config, n_jobs=args.jobs, block=args.block,
            )
            print(out)
        except Exception:
            logger.exception(f"Extraction of '{path}' failed")
            failed += 1
    return 1 if failed else 0


def _calibrants(values: list[str] | None) -> list | None:
    """Formulas or m/z values."""
    if values is None:
        return None
    out = []
    for v in values:
        try:
            out.append(float(v))
        except ValueError:
            out.append(v)
    return out


def add_calibrate_parser(sub) -> None:
    p = sub.add_parser("calibrate", help="Mass calibration of h5 files, following the drift with the retention time.",
                       description="Fit the mass calibration in rt windows on calibrant ions and write the "
                                   "alpinac calibration file (<run>.alpinac/<run>.EI_mc.txt by default), "
                                   "used by 'twutils extract'.")
    p.add_argument("files", nargs="+", help="h5 files (glob patterns allowed).")
    p.add_argument("-c", "--calibrants-file", type=Path,
                   help="Yaml file of the calibrants (and initial calibration), as "
                        "/shared/scripts/tof-processing/config_calib.yml.")
    p.add_argument("--calibrant", nargs="+", metavar="FORMULA",
                   help="Calibrants present during the run: formulas (CF3, Re+2, [13C]O) or m/z values "
                        "(instead of those of --calibrants-file). Default: the calibrants of the h5 file.")
    p.add_argument("--spike", nargs="+", action="append", default=[], metavar=("RT", "FORMULA"),
                   help="Calibrants injected at RT s (repeatable): --spike 1200 CF3 C2F5.")
    p.add_argument("--mode", type=int, default=2, choices=range(6), help="TOFWERK calibration mode (default 2).")
    p.add_argument("--averaging", type=float, default=360.0, help="Duration of the rt windows, s (default 360).")
    p.add_argument("--spacing", type=float, default=120.0, help="Interval between the rt windows, s (default 120).")
    p.add_argument("--rt-start", type=float, help="Start of the calibrated rt range, s.")
    p.add_argument("--rt-stop", type=float, help="Stop of the calibrated rt range, s.")
    p.add_argument("--mz-range", type=float, default=0.2, help="Half width of the calibrant domains, m/z (default 0.2).")
    p.add_argument("--max-ppm", type=float, default=20.0, help="Calibrants with larger residuals are left out (default 20).")
    p.add_argument("--initial", type=Path, help="Calibration file to find the calibrants (default: the h5 calibration).")
    p.add_argument("--neutral", action="store_true", help="Neutral masses of the formulas without charge (legacy alpinac).")
    p.add_argument("--ionisation", choices=["EI", "CI"], default="EI", help="Name of the output file (default EI).")
    p.add_argument("--segments", type=_segments, help="Segments to read, START:STOP (EI/CI instruments).")
    p.add_argument("-o", "--output", type=Path, help="Output file (one input file only).")
    p.add_argument("--peaks-csv", action="store_true", help="Also write the calibrant peaks (.peaks.csv).")
    p.add_argument("--plots", action="store_true",
                   help="Also write the calibration plots (.<plot>.png next to the output, needs matplotlib).")
    p.set_defaults(func=run_calibrate)


def run_calibrate(args) -> int:
    from twutils.calibrate import Calibrants, calibrate_to_file

    files = _files(args.files)
    if not files:
        return 1
    if args.output is not None and len(files) > 1:
        raise SystemExit("--output can only be used with one file.")
    try:
        spikes = {float(rt): _calibrants(cals) for rt, *cals in args.spike}
    except ValueError:
        raise SystemExit("--spike: expected RT FORMULA [FORMULA ...]") from None
    calibrants = _calibrants(args.calibrant)
    if args.calibrants_file is not None:
        config = Calibrants.from_yaml(args.calibrants_file)
        calibrants = config if calibrants is None else Calibrants(calibrants, config.spikes, config.initial)
    failed = 0
    for path in files:
        try:
            result = calibrate_to_file(
                path, output=args.output, ionisation=args.ionisation, peaks_csv=args.peaks_csv,
                calibrants=calibrants, spikes=spikes, mode=args.mode, averaging=args.averaging,
                spacing=args.spacing, rt_start=args.rt_start, rt_stop=args.rt_stop, mz_range=args.mz_range,
                max_ppm=args.max_ppm, initial=args.initial, ionised=not args.neutral, segments=args.segments,
            )
            print(result.summary())
            if args.plots:
                for png in result.plot(args.output, args.ionisation).values():
                    print(f"  {png}")
        except Exception:
            logger.exception(f"Calibration of '{path}' failed")
            failed += 1
    return 1 if failed else 0


def add_plot_parser(sub) -> None:
    p = sub.add_parser("plot-calibration", help="Plots of mass calibrations written by 'twutils calibrate'.",
                       description="Plot a *_mc.txt file and its calibrant peaks (.peaks.csv next to it, from "
                                   "'twutils calibrate --peaks-csv'; without them only the parameters and the "
                                   "drift). The plots are written next to the calibration file.")
    p.add_argument("files", nargs="+", help="*_mc.txt files, or h5 files (their <run>.alpinac/<run>.EI_mc.txt).")
    p.add_argument("--peaks", type=Path, help="Calibrant peaks file (one input file only).")
    p.add_argument("--plot", nargs="+", dest="plots", metavar="NAME",
                   help="Plots to make: parameters drift residuals peak_shape (default all).")
    p.add_argument("--ionisation", choices=["EI", "CI"], default="EI", help="Calibration file of the h5 files.")
    p.add_argument("--show", action="store_true", help="Show the plots.")
    p.set_defaults(func=run_plot)


def run_plot(args) -> int:
    from twutils.calibration import mc_path
    from twutils.h5 import TofFile
    from twutils.plots import PLOTS, plot_outputs

    files = _files(args.files)
    if not files:
        return 1
    if args.peaks is not None and len(files) > 1:
        raise SystemExit("--peaks can only be used with one file.")
    failed = 0
    for path in files:
        mc_file = mc_path(TofFile(path), args.ionisation) if path.suffix == ".h5" else path
        try:
            pngs = plot_outputs(mc_file, args.peaks, args.plots or PLOTS, show=args.show)
            print(mc_file)
            for png in pngs.values():
                print(f"  {png}")
        except Exception:
            logger.exception(f"Plots of '{mc_file}' failed")
            failed += 1
    return 1 if failed else 0


def add_info_parser(sub) -> None:
    p = sub.add_parser("info", help="Dimensions, timing and mass calibration of h5 files.")
    p.add_argument("files", nargs="+", help="h5 files (glob patterns allowed).")
    p.add_argument("--ionisation", choices=["EI", "CI"], default="EI")
    p.set_defaults(func=run_info)


def run_info(args) -> int:
    from twutils.calibration import MassCalibration
    from twutils.h5 import TofFile

    for path in _files(args.files):
        run = TofFile(path)
        cal = MassCalibration.for_file(run, args.ionisation)
        t = run.time_axis
        print(run.path)
        print(f"  scans     : {run.n_scans} ({run.n_writes} writes x {run.n_bufs} bufs), "
              f"{run.s_per_bin:.4f} s/scan, rt {t[0]:.1f} - {t[-1]:.1f} s")
        print(f"  spectra   : {run.n_points} points, {run.n_segments} segment(s), "
              f"m/z {float(cal.mz(t[-1], 0)) if run.n_points else 0:.1f} - {float(cal.mz(t[-1], run.n_points - 1)):.1f}")
        print(f"  mass cal  : mode {cal.mode}, {len(cal.rt)} rt point(s), {cal.source}")
        print(f"  peak shape: sigma = {cal.sigma_intercept:.4g} + {cal.sigma_slope:.4g} x tof, alpha {cal.alpha:.3f}")
        if run.name_info:
            print(f"  run       : {run.name_info['datetime']:%Y-%m-%d %H:%M}, "
                  f"{run.name_info['sample_type']}, port {run.name_info['port']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="twutils", description="Utilities for TOFWERK h5 files.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v: info, -vv: debug.")
    sub = parser.add_subparsers(dest="command", required=True)
    add_extract_parser(sub)
    add_calibrate_parser(sub)
    add_plot_parser(sub)
    add_info_parser(sub)
    args = parser.parse_args(argv)
    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s: %(message)s", level=logging.WARNING)
    logging.getLogger("twutils").setLevel(level)
    logging.getLogger("peaxart").setLevel(level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
