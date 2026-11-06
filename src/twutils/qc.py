"""Quality check of a run: is there data at all (filament, signal, file)?

The instrument state is stored by TOFWERK in ``TPS2/TwData`` (writes, buffers,
channels), with the channel names in ``TPS2/TwInfo``. A broken filament shows
as an emission monitor at 0 while its target stays at 0.5 mA, and the spectra
then only contain noise (``FullSpectra/SumSpectrum`` max ~50 instead of ~1e8).

    qc = twutils.check("261031.1617.tank.9.h5")
    qc.status        # 'filament_failed'
    qc.rt_failure    # 196.6 s
    print(qc)        # one line summary
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from os import PathLike
from pathlib import Path

import numpy as np

from twutils.h5 import H5_ERRORS, TofFile, TofFileError, stem

logger = logging.getLogger(__name__)

EMISSION_MONITOR = "FILAMENT_EMISSION monitor"
EMISSION_TARGET = "FILAMENT_EMISSION target"
CURRENT_MONITOR = "FILAMENT_CURRENT monitor"

class RunStatus(StrEnum):
    """Status of a run, from the worst to the best.

    Members are strings: ``qc.status == "ok"`` works, and they are written as their value.
    """

    UNREADABLE = "unreadable"                #: the file cannot be read (truncated, corrupt)
    SHORT = "short"                          #: fewer scans than a run (acquisition stopped early)
    FILAMENT_OFF = "filament_off"            #: no emission during the whole run
    FILAMENT_FAILED = "filament_failed"      #: emission at the start, lost before the end
    SWITCHED_OFF = "switched_off"            #: emission target set to 0 by the acquisition (not a failure)
    NO_SIGNAL = "no_signal"                  #: emission ok but no signal in the spectra
    FILAMENT_UNSTABLE = "filament_unstable"  #: emission lost for a while, back at the end
    OK = "ok"


STATUSES = list(RunStatus)


@dataclass
class QCConfig:
    """Thresholds of the run check."""

    #: emission monitor below this fraction of its target: no emission
    emission_fraction: float = 0.5
    #: below this maximum of ``FullSpectra/SumSpectrum``: no signal
    min_sum_spectrum: float = 1e4
    #: below this duration (s of written scans): the run is too short to be processed
    min_duration: float = 600.0
    #: shorter stretches without emission are ignored, s: the monitor follows a step of the target in ~1 s
    min_gap: float = 5.0
    #: the filament is switched on after the start of the acquisition and warms up (~30 s): no emission
    #: before this rt is not a problem, s
    max_startup: float = 60.0

@dataclass
class RunQC:
    """Result of the check of one run."""

    path: Path
    status: RunStatus
    message: str = ""
    #: retention time at which the emission was lost for good, s (``filament_failed``, ``switched_off``)
    rt_failure: float | None = None
    #: fraction of the written scans without emission
    fraction_no_emission: float | None = None
    emission_target_max: float | None = None  # mA (the target is lowered during parts of a run)
    emission_ratio: float | None = None    # median of monitor / target over the scans with emission
    current_median: float | None = None    # A, median over the scans with emission
    sum_spectrum_max: float | None = None
    duration: float | None = None          # s, of the written scans
    n_scans: int | None = None             # written scans
    log: list[str] = field(default_factory=list)  # ``AcquisitionLog/Log`` texts

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.OK

    @property
    def name(self) -> str:
        return stem(self.path)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        d["status"] = self.status.value
        d["name"] = self.name
        d["log"] = " | ".join(self.log)
        return d

    def __str__(self) -> str:
        details = []
        if self.emission_ratio is not None:
            details.append(f"emission {100 * self.emission_ratio:.0f} % of target")
        if self.current_median is not None:
            details.append(f"current {_fmt(self.current_median)} A")
        if self.sum_spectrum_max is not None:
            details.append(f"sum spectrum max {self.sum_spectrum_max:.3g}")
        if self.duration is not None:
            details.append(f"{self.duration:.0f} s")
        text = f"{self.name:<24} {self.status:<17}"
        if self.message:
            text += f" {self.message}"
        if details:
            text += f" ({', '.join(details)})"
        return text


def _fmt(value: float | None) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:.3g}"


def check(path: str | PathLike, config: QCConfig | None = None) -> RunQC:
    """Check that a run has data: readable file, filament emission, signal, duration.

    Never raises for a bad file (``unreadable``); other errors (bugs) are raised.
    """
    config = config or QCConfig()
    path = Path(path)
    try:
        run = TofFile(path)
        with run.open():
            return _check(run, config)
    except (TofFileError, *H5_ERRORS) as e:  # truncated / corrupt file, file being written
        logger.debug("Check of %s failed", path, exc_info=True)
        return RunQC(path, RunStatus.UNREADABLE, str(e).removeprefix(f"{path.name}: "))


def below_target(emission: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    """Scans without emission: the monitor below ``fraction`` of the target, or the target at 0.

    The acquisition lowers the target during parts of the run (0.5 -> 0.05 mA): the
    monitor is compared with the target of each scan."""
    return (emission < fraction * target) | (target <= 0)


def ignore_short_gaps(no_emission: np.ndarray, rt: np.ndarray, min_gap: float) -> np.ndarray:
    """``no_emission`` without the stretches shorter than ``min_gap`` s (lag of the monitor after a
    step of the target). A stretch that lasts until the end of the run is kept: it is not a lag."""
    if not no_emission.any() or min_gap <= 0:
        return no_emission
    edges = np.diff(np.concatenate([[0], no_emission.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    kept = no_emission.copy()
    for start, stop in zip(starts, stops, strict=True):
        # lower bound of the duration (the scans may be sampled): a single scan counts as 0 s
        if stop < len(no_emission) and rt[stop - 1] - rt[start] < min_gap:
            kept[start:stop] = False
    return kept


def lost_emission(no_emission: np.ndarray, target: np.ndarray, rt: np.ndarray, max_startup: float) -> np.ndarray:
    """Scans without emission that are a problem: not wanted (target 0), not the warm-up at the start
    (the acquisition switches the filament on after ~10 s, the emission reaches its target ~20 s later)."""
    lost = no_emission & (target > 0)
    first_on = np.flatnonzero(~no_emission)
    if len(first_on) and rt[first_on[0]] <= max_startup:
        lost[: first_on[0]] = False
    return lost


def _rt_failure(run: TofFile, rt: np.ndarray, last_ok: int, config: QCConfig) -> tuple[float, bool]:
    """rt of the first scan without emission after the scan ``last_ok`` (last sampled scan with
    emission), read at full resolution up to the next sampled scan (without emission); and whether the
    target is 0 there."""
    stop = min(run.n_scans, last_ok + run.n_bufs + 1)
    index, channels = run.read_tps2([EMISSION_MONITOR, EMISSION_TARGET], slice(last_ok, stop))
    target = channels[EMISSION_TARGET]
    on = np.flatnonzero(~below_target(channels[EMISSION_MONITOR], target, config.emission_fraction))
    k = min(on[-1] + 1 if len(on) else 0, len(index) - 1)
    return float(rt[index[k]]), bool(target[k] <= 0)


def _check(run: TofFile, config: QCConfig) -> RunQC:
    # Times from the scan period: BufTimes are stored in one chunk per scan, slow to read
    rt = run.regular_time_axis
    qc = RunQC(run.path, RunStatus.OK, n_scans=run.n_scans, duration=run.duration, log=run.acquisition_log())
    sum_spectrum = run.sum_spectrum()
    if sum_spectrum is not None and sum_spectrum.size:
        qc.sum_spectrum_max = float(np.max(sum_spectrum))

    # the first scan of each write is enough (every ~3 s), the failure time is refined at full resolution
    names = [EMISSION_MONITOR, EMISSION_TARGET, CURRENT_MONITOR]
    sampled, channels = run.read_tps2(names, every_write=True)
    if len(sampled) and sampled[-1] < run.n_scans - 1:  # and the last scan: a failure in the last write
        last, values = run.read_tps2(names, slice(run.n_scans - 1, run.n_scans))
        sampled = np.concatenate([sampled, last])
        channels = {k: np.concatenate([v, values[k]]) for k, v in channels.items()}
    no_emission = None
    if EMISSION_MONITOR in channels and EMISSION_TARGET in channels and len(sampled):
        rt_sampled = rt[sampled]
        emission, target = channels[EMISSION_MONITOR], channels[EMISSION_TARGET]
        qc.emission_target_max = float(target.max())
        no_emission = ignore_short_gaps(below_target(emission, target, config.emission_fraction), rt_sampled,
                                        config.min_gap)
        qc.fraction_no_emission = float(no_emission.mean())
        with_emission = ~no_emission & (target > 0)
        qc.emission_ratio = float(np.median(emission[with_emission] / target[with_emission])) \
            if with_emission.any() else 0.0
        if CURRENT_MONITOR in channels:
            current = channels[CURRENT_MONITOR]
            qc.current_median = float(np.median(current[with_emission] if with_emission.any() else current))
    else:
        logger.info("%s: no filament channels in TPS2, only the signal is checked", run.path.name)

    if run.n_scans == 0 or qc.duration < config.min_duration:
        qc.status, qc.message = RunStatus.SHORT, f"only {qc.n_scans} scans written ({qc.duration:.0f} s)"
    elif no_emission is not None and no_emission.all():
        if qc.emission_target_max <= 0:
            qc.status, qc.message = RunStatus.SWITCHED_OFF, "emission switched off during the whole run (target 0)"
        else:
            qc.status, qc.message = RunStatus.FILAMENT_OFF, "no filament emission during the whole run"
    elif no_emission is not None and no_emission[-1]:
        # lost for good: first scan of the last stretch without emission
        last_ok = int(sampled[np.flatnonzero(~no_emission)[-1]])
        qc.rt_failure, target_off = _rt_failure(run, rt, last_ok, config)
        if target_off:  # the acquisition set the target to 0 when the emission stopped: wanted
            qc.status = RunStatus.SWITCHED_OFF
            qc.message = f"emission switched off at rt {qc.rt_failure:.1f} s (target 0)"
        else:
            qc.status = RunStatus.FILAMENT_FAILED
            qc.message = f"filament emission lost at rt {qc.rt_failure:.1f} s"
    elif qc.sum_spectrum_max is not None and qc.sum_spectrum_max < config.min_sum_spectrum:
        qc.status, qc.message = RunStatus.NO_SIGNAL, "no signal in the spectra (detector?)"
    elif no_emission is not None and (lost := lost_emission(no_emission, target, rt_sampled,
                                                            config.max_startup)).any():
        qc.status = RunStatus.FILAMENT_UNSTABLE
        qc.message = f"no filament emission during {100 * lost.mean():.1f} % of the run"
    return qc


def check_files(paths, config: QCConfig | None = None) -> Iterator[RunQC]:
    """:func:`check` of several files, one result at a time."""
    for path in paths:
        yield check(path, config)


def to_dataframe(results: list[RunQC]):
    """Table of results (one row per run)."""
    import pandas as pd

    return pd.DataFrame([r.to_dict() for r in results]).set_index("name")


__all__ = ["QCConfig", "RunQC", "RunStatus", "STATUSES", "check", "check_files", "to_dataframe"]
