"""Reading of TOFWERK h5 files (``FullSpectra/TofData``)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import cached_property
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# yymmdd.hhmm.type.port, e.g. 241007.0423.air.1
_NAME_PATTERN = re.compile(r"(\d{6})\.(\d{4})\.([a-zA-Z0-9_-]+)\.(\d{1,2})")


def _scalar(value):
    if isinstance(value, np.ndarray):
        value = value.flat[0]
    if isinstance(value, bytes):
        value = value.decode()
    return value


class TofFile:
    """A TOFWERK h5 file.

    The spectra are stored in ``FullSpectra/TofData`` with dimensions
    (writes, buffers, segments, tof index): writes x buffers is the time axis,
    the segments are the ionisation modes of EI/CI instruments (one for EI).
    The file is only opened while reading.
    """

    def __init__(self, path: str | PathLike):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as f:
            tofdata = f["FullSpectra/TofData"]
            self.n_writes, self.n_bufs, self.n_segments, self.n_points = tofdata.shape
            self.s_per_bin = float(_scalar(f["TimingData"].attrs["BlockPeriod"])) * 1e-9

    def __repr__(self) -> str:
        return (f"TofFile('{self.path}', {self.n_scans} scans x {self.n_points} points, "
                f"{self.n_segments} segment(s), {self.s_per_bin:.4f} s/scan)")

    @property
    def n_scans(self) -> int:
        return self.n_writes * self.n_bufs

    @property
    def stem(self) -> str:
        """File name without ``.h5``."""
        return self.path.name[:-3] if self.path.name.endswith(".h5") else self.path.stem

    @cached_property
    def name_info(self) -> dict:
        """Date, sample type and port from a name like ``241007.0423.air.1.h5`` (empty if no match)."""
        m = _NAME_PATTERN.match(self.path.name)
        if m is None:
            return {}
        day, hhmm, sample_type, port = m.groups()
        return {"datetime": datetime.strptime(day + hhmm, "%y%m%d%H%M"),
                "sample_type": sample_type, "port": int(port)}

    @cached_property
    def time_axis(self) -> np.ndarray:
        """Retention time of each scan, s (``TimingData/BufTimes``).

        Scans not written at the end of the file (time 0) are extrapolated
        with the scan period.
        """
        with h5py.File(self.path, "r") as f:
            t = f["TimingData/BufTimes"][()].astype(float).ravel()
        written = np.flatnonzero(t != 0)
        last = written[-1] if len(written) else 0
        if last < len(t) - 1:
            t[last + 1:] = t[last] + self.s_per_bin * np.arange(1, len(t) - last)
        return t

    @cached_property
    def mass_axis(self) -> np.ndarray:
        """``FullSpectra/MassAxis``: m/z of the tof indices from the acquisition calibration."""
        with h5py.File(self.path, "r") as f:
            return f["FullSpectra/MassAxis"][()]

    def attrs(self, group: str = "FullSpectra") -> dict:
        """Attributes of a group (single values unpacked)."""
        with h5py.File(self.path, "r") as f:
            return {k: _scalar(v) for k, v in f[group].attrs.items()}

    def scan_range(self, rt_start: float | None = None, rt_stop: float | None = None) -> slice:
        """Scans with ``rt_start <= rt <= rt_stop``."""
        t = self.time_axis
        i0 = 0 if rt_start is None else int(np.searchsorted(t, rt_start, side="left"))
        i1 = len(t) if rt_stop is None else int(np.searchsorted(t, rt_stop, side="right"))
        return slice(i0, max(i0, i1))

    def read(
        self,
        rt_start: float | None = None,
        rt_stop: float | None = None,
        tof_start: int = 0,
        tof_stop: int | None = None,
        segments: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spectra of a retention time window.

        :param rt_start, rt_stop: retention time window, s (inclusive).
        :param tof_start, tof_stop: tof index range [start, stop[.
        :param segments: segment range [start, stop[, summed. Required for
            files with several segments (EI/CI instruments).
        :return: intensity (n_scans, n_points) float32 and rt (n_scans,) s.
        """
        if segments is None:
            if self.n_segments != 1:
                raise ValueError(f"{self.path.name} has {self.n_segments} segments: give the segments to read.")
            segments = (0, 1)
        s0, s1 = segments
        if not 0 <= s0 < s1 <= self.n_segments:
            raise ValueError(f"Invalid segments {segments}, the file has {self.n_segments}.")
        tof_stop = self.n_points if tof_stop is None else min(tof_stop, self.n_points)
        tof_start = max(0, tof_start)
        scans = self.scan_range(rt_start, rt_stop)
        n = scans.stop - scans.start
        if n == 0 or tof_stop <= tof_start:
            return np.zeros((n, max(0, tof_stop - tof_start)), np.float32), self.time_axis[scans]
        w0 = scans.start // self.n_bufs
        w1 = (scans.stop - 1) // self.n_bufs + 1
        with h5py.File(self.path, "r") as f:
            data = f["FullSpectra/TofData"][w0:w1, :, s0:s1, tof_start:tof_stop]
        data = data.sum(axis=2, dtype=data.dtype) if s1 - s0 > 1 else data[:, :, 0]
        data = data.reshape(-1, tof_stop - tof_start)
        offset = scans.start - w0 * self.n_bufs
        return data[offset:offset + n], self.time_axis[scans]
