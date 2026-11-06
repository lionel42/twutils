"""Command line: ``twutils extract``, ``twutils info`` and ``twutils check``.

The mass calibration and the unit mass chromatograms are made by esper (``esper calibrate``,
``esper chromatograms``)."""

from __future__ import annotations

import argparse
import glob
import inspect
import logging
import sys
from pathlib import Path
from typing import get_args

from twutils.calibration import Ionisation

logger = logging.getLogger("twutils.cli")
IONISATIONS = list(get_args(Ionisation))


def _default(func, name: str):
    """Default value of the parameter ``name`` of ``func``: the CLI defaults are those of the API."""
    return inspect.signature(func).parameters[name].default


def _for_each(files: list[Path], action: str, func) -> int:
    """``func(path)`` for each file; a failure is logged and the next file processed.

    A bad file (:class:`twutils.h5.TofFileError`: truncated, corrupt, missing data) is
    logged in one line; any other error with its traceback (a bug, or an unexpected file).
    :return: exit code, 1 if a file failed.
    """
    from twutils.h5 import TofFileError

    failed = 0
    for path in files:
        try:
            func(path)
        except TofFileError as e:  # its message starts with the file name
            logger.error("%s failed: %s", action, e)
            failed += 1
        except FileNotFoundError as e:
            logger.error("%s of '%s' failed: no such file (%s)", action, path, e)
            failed += 1
        except Exception:
            logger.exception("%s of '%s' failed", action, path)
            failed += 1
    if failed and len(files) > 1:
        logger.error("%s: %s of %s files failed", action, failed, len(files))
    return 1 if failed else 0


def _files(patterns: list[str]) -> list[Path]:
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matches:
            logger.warning("No file matches '%s'", pattern)
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


# --- arguments shared by several commands -------------------------------------------


def _add_files(p, what: str = "h5 files") -> None:
    p.add_argument("files", nargs="+", help=f"{what} (glob patterns allowed).")


def _add_rt_range(p, what: str) -> None:
    """``--rt-start``, ``--rt-stop``: the rt range ``what`` (default the whole run)."""
    p.add_argument("--rt-start", type=float, help=f"Start of the {what}, s (default: start of the run).")
    p.add_argument("--rt-stop", type=float, help=f"Stop of the {what}, s (default: end of the run).")


def _add_segments(p, what: str) -> None:
    """``--segments``: segment range [start, stop[ (``what`` it selects)."""
    p.add_argument("--segments", type=_segments, metavar="INDEX|START:STOP",
                   help=f"Segments {what}: INDEX, or START:STOP summed (STOP excluded). Required for files with "
                        "several segments (EI/CI instruments); default 0 for files with one.")


def _add_ionisation(p, what: str) -> None:
    p.add_argument("--ionisation", choices=IONISATIONS, default="EI",
                   help=f"Ionisation (default %(default)s): {what}.")


def _add_outputs(p, peaks_csv: str) -> None:
    """``-o``, ``--output-dir``, ``--peaks-csv``."""
    p.add_argument("-o", "--output", type=Path, help="Output file (one input file only).")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: <run>.alpinac next to the h5 file).")
    p.add_argument("--peaks-csv", action="store_true", help=f"Also write {peaks_csv} (.peaks.csv).")


LEGACY_CALIBRATION = "name of the legacy calibration file <run>.<EI|CI>_mc.txt, used if there is no json file"


def add_extract_parser(sub) -> None:
    from twutils.extraction import extract

    p = sub.add_parser("extract", help="Extract the peaks of h5 files for the alpinac identification.",
                       description="Extract the peaks of h5 files with peaxart and write the alpinac "
                                   "peak file (<run>.alpinac/<run>.frag.<start>s<stop>s.txt by default).")
    _add_files(p)
    _add_rt_range(p, "retention time window")
    p.add_argument("--rt", type=float, nargs=2, metavar=("RT", "HALF_WIDTH"),
                   help="Window RT +- HALF_WIDTH s (instead of --rt-start/--rt-stop, not with them).")
    p.add_argument("--mass", type=float, nargs="+",
                   help="Unit masses to extract (default: --mass-start to --mass-stop).")
    p.add_argument("--mass-start", type=float, help="First unit mass (default: first of the file).")
    p.add_argument("--mass-stop", type=float, help="Last unit mass (default: last of the file).")
    p.add_argument("--mz-range", type=float, default=_default(extract, "mz_range"),
                   help="Half width of the domain around each unit mass (default %(default)s).")
    p.add_argument("--mass-cal", type=Path,
                   help="Mass calibration file (*_mc.json, or legacy *_mc.txt). Default: searched next to the h5 file.")
    _add_ionisation(p, f"spectrum id of the output (EI 0, CI 1), and {LEGACY_CALIBRATION}")
    _add_segments(p, "read (and their calibration file)")
    p.add_argument("--adduct", default="", help="Adduct of CI spectra.")
    p.add_argument("--legacy", action="store_true",
                   help="Reproduce the legacy alpinac extraction (peaxart.ExtractionConfig.legacy()).")
    p.add_argument("--rt-sigma", type=float, help="Expected chromatographic sigma, s (default: legacy model of rt).")
    p.add_argument("--config", type=_key_value, action="append", default=[], metavar="KEY=VALUE",
                   help="Other peaxart.ExtractionConfig parameter (repeatable), e.g. --config peak_linear_ratio=0.8.")
    p.add_argument("-j", "--jobs", type=int, default=1, help="Domains extracted in parallel.")
    p.add_argument("--block", type=float, default=_default(extract, "block"),
                   help="Long windows are processed in blocks of this many s (default %(default)s).")
    p.add_argument("--metadata", type=_key_value, action="append", default=[], metavar="KEY=VALUE",
                   help="Added to the header of the output file (repeatable).")
    _add_outputs(p, "the full peaxart peak table")
    p.set_defaults(func=run_extract)


def _parse_value(key: str, text: str, annotation) -> object:
    """``text`` as the type of a dataclass field (annotation such as ``'float | None'``)."""
    kinds = {k.strip() for k in str(annotation).split("|")}
    if text.lower() == "none" and "None" in kinds:
        return None
    try:
        if "bool" in kinds:
            if text.lower() not in ("1", "0", "true", "false", "yes", "no"):
                raise ValueError(text)
            return text.lower() in ("1", "true", "yes")
        if "int" in kinds:
            return int(text)
        if "float" in kinds:
            return float(text)
    except ValueError:
        raise SystemExit(f"--config {key}: invalid value '{text}' ({annotation})") from None
    return text


def _config(args):
    import dataclasses

    import peaxart

    fields = {f.name: f for f in dataclasses.fields(peaxart.ExtractionConfig)}
    changes = {}
    for key, text in args.config:
        if key not in fields:
            raise SystemExit(f"Unknown peaxart.ExtractionConfig parameter '{key}'")
        changes[key] = _parse_value(key, text, fields[key].type)
    if args.rt_sigma is not None:
        changes["rt_sigma"] = args.rt_sigma
    return peaxart.ExtractionConfig.legacy(**changes) if args.legacy else peaxart.ExtractionConfig(**changes)


def run_extract(args) -> int:
    from twutils.extraction import extract_to_file

    files = _files(args.files)
    if not files:
        return 1
    if args.output is not None and len(files) > 1:
        raise SystemExit("--output can only be used with one file (use --output-dir).")
    rt_start, rt_stop = args.rt_start, args.rt_stop
    if args.rt is not None:
        if rt_start is not None or rt_stop is not None:
            raise SystemExit("--rt cannot be used with --rt-start / --rt-stop.")
        rt_start, rt_stop = args.rt[0] - args.rt[1], args.rt[0] + args.rt[1]
    config = _config(args)

    def run(path):
        print(extract_to_file(
            path, output=args.output, output_dir=args.output_dir, metadata=dict(args.metadata),
            adduct=args.adduct, peaks_csv=args.peaks_csv,
            rt_start=rt_start, rt_stop=rt_stop, masses=args.mass, mass_start=args.mass_start,
            mass_stop=args.mass_stop, mz_range=args.mz_range,
            calibration=args.mass_cal, ionisation=args.ionisation, segments=args.segments,
            config=config, n_jobs=args.jobs, block=args.block,
        ))

    return _for_each(files, "Extraction", run)


def add_info_parser(sub) -> None:
    p = sub.add_parser("info", help="Dimensions, timing and mass calibration of h5 files.")
    _add_files(p)
    _add_segments(p, "of the mass calibration shown (default here: each segment)")
    _add_ionisation(p, LEGACY_CALIBRATION)
    p.set_defaults(func=run_info)


def run_info(args) -> int:
    from twutils.calibration import calibration_for
    from twutils.h5 import TofFile, segments_label

    files = _files(args.files)
    if not files:
        return 1

    def info(path):
        run = TofFile(path)
        print(run.path)
        print(f"  scans     : {run.n_scans} written of {run.n_buffers} ({run.n_writes} writes x {run.n_bufs} "
              f"bufs), {run.s_per_bin:.4f} s/scan, {run.duration:.1f} s")
        print(f"  spectra   : {run.n_points} points, {run.n_segments} segment(s)")
        rt = float(run.regular_time_axis[0]) if run.n_scans else 0.0  # m/z range at the start of the run
        ranges = [args.segments] if args.segments else [(k, k + 1) for k in range(run.n_segments)]
        for segments in ranges:
            cal = calibration_for(run, segments, args.ionisation)
            mz0, mz1 = (float(v) for v in cal.mz(rt, [0, run.n_points - 1]))
            label = f" {segments_label(segments)}" if run.n_segments > 1 or args.segments else ""
            print(f"  mass cal{label:5} : mode {cal.mode}, {len(cal.rt)} rt point(s), m/z {mz0:.1f} - {mz1:.1f}, "
                  f"{cal.source}")
            print(f"  peak shape: sigma = {cal.sigma_intercept:.4g} + {cal.sigma_slope:.4g} x tof, "
                  f"alpha {cal.alpha:.3f}")
        if run.name_info:
            print(f"  run       : {run.name_info['datetime']:%Y-%m-%d %H:%M}, "
                  f"{run.name_info['sample_type']}, port {run.name_info['port']}")

    return _for_each(files, "Info", info)


def add_check_parser(sub) -> None:
    from twutils.qc import QCConfig

    p = sub.add_parser("check", help="Check that runs have data (filament emission, signal, duration).",
                       description="One line per run: ok, unreadable, short, filament_off, filament_failed (with "
                                   "the rt), switched_off (target set to 0 by the acquisition), no_signal or "
                                   "filament_unstable. Exit code 1 if a run is not ok.")
    _add_files(p)
    default = QCConfig()
    p.add_argument("--emission-fraction", type=float, default=default.emission_fraction,
                   help="Emission monitor below this fraction of the target: no emission (default %(default)s).")
    p.add_argument("--min-sum-spectrum", type=float, default=default.min_sum_spectrum,
                   help="Maximum of FullSpectra/SumSpectrum below this: no signal (default %(default)s).")
    p.add_argument("--min-duration", type=float, default=default.min_duration,
                   help="Shorter runs are 'short', s (default %(default)s).")
    p.add_argument("--min-gap", type=float, default=default.min_gap,
                   help="Shorter stretches without emission are ignored, s (default %(default)s).")
    p.add_argument("--max-startup", type=float, default=default.max_startup,
                   help="No emission before this rt (filament warm-up) is not a problem, s (default %(default)s).")
    p.add_argument("--not-ok", action="store_true", help="Only print the runs that are not ok.")
    p.add_argument("--csv", type=Path, help="Also write the results to this csv file.")
    p.set_defaults(func=run_check)


def run_check(args) -> int:
    from twutils.qc import QCConfig, check, to_dataframe

    files = _files(args.files)
    if not files:
        return 1
    config = QCConfig(emission_fraction=args.emission_fraction, min_sum_spectrum=args.min_sum_spectrum,
                      min_duration=args.min_duration, min_gap=args.min_gap,
                      max_startup=args.max_startup)
    results = []

    def run(path):  # check() gives 'unreadable' for bad files; other errors are bugs, logged
        qc = check(path, config)
        results.append(qc)
        if not (args.not_ok and qc.ok):
            print(qc, flush=True)

    code = _for_each(files, "Check", run)
    if args.csv is not None and results:
        to_dataframe(results).to_csv(args.csv)
    return 0 if code == 0 and all(qc.ok for qc in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="twutils", description="Utilities for TOFWERK h5 files.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v: info, -vv: debug.")
    sub = parser.add_subparsers(dest="command", required=True)
    add_extract_parser(sub)
    add_info_parser(sub)
    add_check_parser(sub)
    args = parser.parse_args(argv)
    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s: %(message)s", level=logging.WARNING)
    logging.getLogger("twutils").setLevel(level)
    logging.getLogger("peaxart").setLevel(level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
