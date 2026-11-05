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
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

EMISSION_MONITOR = "FILAMENT_EMISSION monitor"
EMISSION_TARGET = "FILAMENT_EMISSION target"
CURRENT_MONITOR = "FILAMENT_CURRENT monitor"

# Status of a run, from the worst to the best
UNREADABLE = "unreadable"            # the file cannot be read
SHORT = "short"                      # fewer scans than a run (acquisition stopped early)
FILAMENT_OFF = "filament_off"        # no emission during the whole run
FILAMENT_FAILED = "filament_failed"  # emission at the start, lost before the end
SWITCHED_OFF = "switched_off"        # emission target set to 0 by the acquisition (not a failure)
NO_SIGNAL = "no_signal"              # emission ok but no signal in the spectra
FILAMENT_UNSTABLE = "filament_unstable"  # emission lost for a while, back at the end
OK = "ok"
STATUSES = [UNREADABLE, SHORT, FILAMENT_OFF, FILAMENT_FAILED, SWITCHED_OFF, NO_SIGNAL, FILAMENT_UNSTABLE, OK]


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
    status: str
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
        return self.status == OK

    @property
    def name(self) -> str:
        name = self.path.name
        return name[:-3] if name.endswith(".h5") else name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
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


def _channel_index(f: h5py.File, names: list[str]) -> dict[str, int]:
    """Index in ``TPS2/TwData`` of the channels whose name starts with ``names`` (missing ones left out)."""
    if "TPS2/TwData" not in f or "TPS2/TwInfo" not in f:
        return {}
    info = [x.decode(errors="ignore") if isinstance(x, bytes) else str(x) for x in f["TPS2/TwInfo"][()]]
    found = {}
    for name in names:
        index = [i for i, n in enumerate(info) if n.startswith(name)]
        if index:
            found[name] = index[0]
    return found


def _read_channels(f: h5py.File, found: dict[str, int], selection=()) -> dict[str, np.ndarray]:
    """Values (n_scans,) of the channels ``found`` in ``TPS2/TwData[selection]`` (all channels read at once:
    h5py fancy indexing on the last axis is ~10x slower)."""
    data = f["TPS2/TwData"][selection]
    values = data.reshape(-1, data.shape[-1])
    return {name: values[:, i].astype(float) for name, i in found.items()}


def check(path: str | PathLike, config: QCConfig | None = None) -> RunQC:
    """Check that a run has data: readable file, filament emission, signal, duration."""
    config = config or QCConfig()
    path = Path(path)
    try:
        f = h5py.File(path, "r")
    except Exception as e:  # truncated file, file being written, not an h5 file
        return RunQC(path, UNREADABLE, f"cannot open the file: {e}")
    with f:
        try:
            if "FullSpectra/TofData" not in f or "TimingData/BufTimes" not in f:
                return RunQC(path, UNREADABLE, "no FullSpectra/TofData or TimingData/BufTimes")
            return _check(f, path, config)
        except Exception as e:
            logger.debug(f"Check of {path} failed", exc_info=True)
            return RunQC(path, UNREADABLE, f"cannot read the file: {e}")


def _scan_times(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """(written, rt): mask of the written scans in the file buffers, and their retention times, s.

    From the attributes (``NbrWrites`` complete writes, ``BlockPeriod``) when possible: the chunks of
    ``TimingData/BufTimes`` are single values, slow to read (~1.7 s per file, more than the rest). The
    times agree with ``BufTimes`` within ~10 ms. Buffers of an incomplete last write are left out.
    """
    n_total = int(np.prod(f["TimingData/BufTimes"].shape))
    try:
        n_writes = int(np.ravel(f.attrs["NbrWrites"])[0])
        n_bufs = int(np.ravel(f.attrs["NbrBufs"])[0])
        period = float(np.ravel(f["TimingData"].attrs["BlockPeriod"])[0]) * 1e-9
        n_blocks = int(np.ravel(f.attrs.get("NbrBlocks", [1]))[0])
    except (KeyError, IndexError, ValueError):
        n_writes = None
    if n_writes is not None and 0 <= n_writes * n_bufs <= n_total and period > 0:
        n = n_writes * n_bufs
        written = np.zeros(n_total, dtype=bool)
        written[:n] = True
        return written, np.arange(n) * period * n_blocks
    buf_times = f["TimingData/BufTimes"][()].astype(float).ravel()
    # Buffers not written (incomplete last write) have time 0, and all channels at 0
    written = buf_times > 0
    if len(written) > 1 and written[1:].any():
        written[0] = True  # the first buffer may be at t = 0
    return written, buf_times[written]


def _ignore_short_gaps(no_emission: np.ndarray, rt: np.ndarray, min_gap: float) -> np.ndarray:
    """``no_emission`` without the stretches shorter than ``min_gap`` s (lag of the monitor after a step)."""
    if not no_emission.any() or min_gap <= 0:
        return no_emission
    edges = np.diff(np.concatenate([[0], no_emission.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    kept = no_emission.copy()
    for start, stop in zip(starts, stops):
        # lower bound of the duration (the scans may be sampled): a single scan counts as 0 s
        if rt[stop - 1] - rt[start] < min_gap:
            kept[start:stop] = False
    return kept


def _no_emission(emission: np.ndarray, target: np.ndarray, rt: np.ndarray, config: QCConfig) -> np.ndarray:
    """Scans without emission. The acquisition lowers the target during parts of the run (0.5 -> 0.05 mA):
    the monitor is compared with the target."""
    no_emission = (emission < config.emission_fraction * target) | (target <= 0)
    return _ignore_short_gaps(no_emission, rt, config.min_gap)


def _lost(no_emission: np.ndarray, target: np.ndarray, rt: np.ndarray, config: QCConfig) -> np.ndarray:
    """Scans without emission that are a problem: not wanted (target 0), not the warm-up at the start
    (the acquisition switches the filament on after ~10 s, the emission reaches its target ~20 s later)."""
    lost = no_emission & (target > 0)
    first_on = np.flatnonzero(~no_emission)
    if len(first_on) and rt[first_on[0]] <= config.max_startup:
        lost[: first_on[0]] = False
    return lost


def _rt_failure(f, found, written, rt, n_bufs: int, last_ok: int, config: QCConfig) -> float:
    """rt of the first scan without emission after ``last_ok`` (write of the last sampled scan with
    emission), read at full resolution in this write and the next one."""
    channels = _read_channels(f, found, np.s_[last_ok:last_ok + 2])
    start = last_ok * n_bufs
    stop = min(start + len(channels[EMISSION_MONITOR]), int(written.sum()))
    emission = channels[EMISSION_MONITOR][: stop - start]
    target = channels[EMISSION_TARGET][: stop - start]
    no_emission = (emission < config.emission_fraction * target) | (target <= 0)
    ok = np.flatnonzero(~no_emission)
    index = start + (ok[-1] + 1 if len(ok) else 0)
    return float(rt[min(index, len(rt) - 1)])


def _check(f: h5py.File, path: Path, config: QCConfig) -> RunQC:
    written, rt = _scan_times(f)
    qc = RunQC(path, OK, n_scans=int(written.sum()))
    qc.duration = float(rt[-1] - rt[0]) if len(rt) > 1 else 0.0
    if "AcquisitionLog/Log" in f:
        qc.log = [row["logtext"].decode(errors="ignore") for row in f["AcquisitionLog/Log"][()]]
    if "FullSpectra/SumSpectrum" in f:
        qc.sum_spectrum_max = float(np.max(f["FullSpectra/SumSpectrum"][()]))

    found = _channel_index(f, [EMISSION_MONITOR, EMISSION_TARGET, CURRENT_MONITOR])
    no_emission = None
    if EMISSION_MONITOR in found and EMISSION_TARGET in found:
        # TwData is stored in one chunk per scan: on a slow disk reading them all takes several s. The
        # first scan of each write is enough (every ~3 s), the failure time is refined below.
        shape = f["TPS2/TwData"].shape
        n_bufs = shape[1] if len(shape) == 3 else 1
        n_written = int(written.sum())
        n_samples = -(-n_written // n_bufs)  # writes with at least their first scan written
        sampled = np.arange(n_samples) * n_bufs
        if len(shape) == 3:
            channels = _read_channels(f, found, np.s_[:n_samples, 0, :])
        else:
            channels = {k: v[written] for k, v in _read_channels(f, found).items()}
        rt_sampled = rt[sampled]
        emission, target = channels[EMISSION_MONITOR], channels[EMISSION_TARGET]
        qc.emission_target_max = float(target.max()) if len(target) else None
        no_emission = _no_emission(emission, target, rt_sampled, config)
        qc.fraction_no_emission = float(no_emission.mean()) if len(no_emission) else None
        with_emission = ~no_emission
        ratio = emission[with_emission] / target[with_emission]
        qc.emission_ratio = float(np.median(ratio)) if len(ratio) else 0.0
        if CURRENT_MONITOR in channels and len(target):
            current = channels[CURRENT_MONITOR]
            qc.current_median = float(np.median(current[with_emission] if with_emission.any() else current))
    else:
        logger.info(f"{path.name}: no filament channels in TPS2, only the signal is checked")

    if len(rt) == 0 or qc.duration < config.min_duration:
        qc.status, qc.message = SHORT, f"only {qc.n_scans} scans written ({qc.duration:.0f} s)"
    elif no_emission is not None and no_emission.all():
        if qc.emission_target_max is not None and qc.emission_target_max <= 0:
            qc.status, qc.message = SWITCHED_OFF, "emission switched off during the whole run (target 0)"
        else:
            qc.status, qc.message = FILAMENT_OFF, "no filament emission during the whole run"
    elif no_emission is not None and no_emission[-1]:
        # lost for good: first scan of the last stretch without emission
        last_ok = np.flatnonzero(~no_emission)[-1]
        qc.rt_failure = _rt_failure(f, found, written, rt, n_bufs, last_ok, config) if len(shape) == 3 \
            else float(rt_sampled[last_ok + 1])
        if target[-1] <= 0:  # the acquisition set the target to 0: wanted, the data before are fine
            qc.status = SWITCHED_OFF
            qc.message = f"emission switched off at rt {qc.rt_failure:.1f} s (target 0)"
        else:
            qc.status = FILAMENT_FAILED
            qc.message = f"filament emission lost at rt {qc.rt_failure:.1f} s"
    elif qc.sum_spectrum_max is not None and qc.sum_spectrum_max < config.min_sum_spectrum:
        qc.status, qc.message = NO_SIGNAL, "no signal in the spectra (detector?)"
    elif no_emission is not None and (lost := _lost(no_emission, target, rt_sampled, config)).any():
        qc.status = FILAMENT_UNSTABLE
        qc.message = f"no filament emission during {100 * lost.mean():.1f} % of the run"
    return qc


def check_files(paths, config: QCConfig | None = None) -> list[RunQC]:
    """:func:`check` of several files."""
    return [check(p, config) for p in paths]


def to_dataframe(results: list[RunQC]):
    """Table of results (one row per run)."""
    import pandas as pd

    return pd.DataFrame([r.to_dict() for r in results]).set_index("name")


__all__ = ["QCConfig", "RunQC", "STATUSES", "check", "check_files", "to_dataframe"]
