"""Reading of TOFWERK h5 files.

Everything that depends on how TOFWERK writes the files is here: the spectra
(``FullSpectra/TofData``), the timing (``TimingData``), the instrument channels
(``TPS2``) and the acquisition log. The other modules go through :class:`TofFile`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# yymmdd.hhmm.type.port, e.g. 241007.0423.air.1
_NAME_PATTERN = re.compile(r"(\d{6})\.(\d{4})\.([a-zA-Z0-9_-]+)\.(\d{1,2})")

#: errors of h5py / HDF5 on a truncated or corrupt file (raised when opening it, or later when reading it)
H5_ERRORS = (OSError, RuntimeError, KeyError)


class TofFileError(Exception):
    """The file is not a readable TOFWERK h5 file (truncated, corrupt, missing datasets)."""


def stem(path: str | PathLike) -> str:
    """Run name: the file name without ``.h5`` (``241007.0423.air.1``)."""
    name = Path(path).name
    return name[:-3] if name.endswith(".h5") else Path(name).stem


def segments_label(segments: tuple[int, int]) -> str:
    """Name of a segment range [start, stop[ in file names: ``seg0``, ``seg1-2`` (segments 1 and 2 summed)."""
    s0, s1 = segments
    return f"seg{s0}" if s1 == s0 + 1 else f"seg{s0}-{s1 - 1}"


def _scalar(value):
    if isinstance(value, np.ndarray):
        value = value.flat[0]
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    return value


class TofFile:
    """A TOFWERK h5 file.

    The spectra are stored in ``FullSpectra/TofData`` with dimensions
    (writes, buffers, segments, tof index): writes x buffers is the time axis
    (one scan per buffer), the segments are the ionisation modes of EI/CI
    instruments (one for EI).

    Only the written scans are used (:attr:`n_scans`): the dataset is allocated
    for the planned duration, and the buffers not written (acquisition stopped,
    incomplete last write) contain zeros.

    The file is opened for each read, or once for several reads with :meth:`open`::

        with run.open():
            for scans, rt, data in run.iter_blocks(tof_start=1000, tof_stop=2000):
                ...

    :raises TofFileError: the file cannot be opened or has no spectra.
    """

    def __init__(self, path: str | PathLike):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._h5: h5py.File | None = None
        try:
            with h5py.File(self.path, "r") as f:
                if "FullSpectra/TofData" not in f or "TimingData" not in f:
                    raise TofFileError(f"{self.path.name}: no FullSpectra/TofData or TimingData")
                tofdata = f["FullSpectra/TofData"]
                if tofdata.ndim != 4:
                    raise TofFileError(f"{self.path.name}: FullSpectra/TofData has {tofdata.ndim} dimensions, not 4")
                self.n_writes, self.n_bufs, self.n_segments, self.n_points = tofdata.shape
                self.s_per_bin = float(_scalar(f["TimingData"].attrs["BlockPeriod"])) * 1e-9
                self.n_scans = self._written_scans(f)
        except TofFileError:
            raise
        except H5_ERRORS as e:
            raise TofFileError(f"{self.path.name}: cannot read the file ({e})") from e

    def _written_scans(self, f: h5py.File) -> int:
        """Number of written scans: complete writes (``NbrWrites`` x ``NbrBufs``), else up to the last
        non-zero ``BufTimes`` (slower: one chunk per scan)."""
        try:
            n_writes, n_bufs = int(_scalar(f.attrs["NbrWrites"])), int(_scalar(f.attrs["NbrBufs"]))
        except (KeyError, ValueError, TypeError):
            n_writes = n_bufs = None
        if n_writes is not None and n_bufs == self.n_bufs and 0 <= n_writes <= self.n_writes:
            return n_writes * n_bufs
        if "TimingData/BufTimes" not in f:
            raise TofFileError(f"{self.path.name}: no NbrWrites attribute and no TimingData/BufTimes")
        written = np.flatnonzero(f["TimingData/BufTimes"][()].ravel() > 0)
        return int(written[-1]) + 1 if len(written) else 0  # the first scan is at t = 0

    def __repr__(self) -> str:
        return (f"TofFile('{self.path}', {self.n_scans} scans x {self.n_points} points, "
                f"{self.n_segments} segment(s), {self.s_per_bin:.4f} s/scan)")

    def __getstate__(self):  # picklable (parallel jobs) without the open h5 file
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    # --- access to the h5 file ------------------------------------------------

    @contextmanager
    def open(self) -> Generator[TofFile]:
        """Keep the file open for the reads inside the ``with`` block (nested blocks allowed)."""
        if self._h5 is not None:
            yield self
            return
        self._h5 = h5py.File(self.path, "r")
        try:
            yield self
        finally:
            self._h5.close()
            self._h5 = None

    @contextmanager
    def _file(self) -> Generator[h5py.File]:
        if self._h5 is not None:
            yield self._h5
        else:
            with h5py.File(self.path, "r") as f:
                yield f

    # --- run information ----------------------------------------------------

    @property
    def stem(self) -> str:
        """File name without ``.h5``."""
        return stem(self.path)

    @property
    def n_buffers(self) -> int:
        """Scans allocated in the file (written or not)."""
        return self.n_writes * self.n_bufs

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
        """Retention time of each written scan, s (``TimingData/BufTimes``).

        Read from the file: slow on a slow disk (one chunk per scan, ~2 s per
        run on a USB disk). If the times do not increase (missed buffer), the
        regular times of :attr:`regular_time_axis` are used, with a warning.
        """
        with self._file() as f:
            if "TimingData/BufTimes" not in f:
                return self.regular_time_axis
            t = f["TimingData/BufTimes"][()].astype(float).ravel()[: self.n_scans]
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            bad = np.flatnonzero(np.diff(t) <= 0)
            logger.warning("%s: BufTimes do not increase at scan(s) %s%s: using the scan period", self.path.name,
                           bad[:5].tolist(), "..." if len(bad) > 5 else "")
            return t[0] + self.regular_time_axis
        return t

    @property
    def regular_time_axis(self) -> np.ndarray:
        """Retention times from the scan period (``BlockPeriod``), without reading ``BufTimes``:
        within ~10 ms of them over a run."""
        return np.arange(self.n_scans) * self.s_per_bin

    @property
    def duration(self) -> float:
        """Time between the first and the last written scans, s (from the scan period)."""
        return max(0, self.n_scans - 1) * self.s_per_bin

    @cached_property
    def mass_axis(self) -> np.ndarray:
        """``FullSpectra/MassAxis``: m/z of the tof indices from the acquisition calibration."""
        with self._file() as f:
            return f["FullSpectra/MassAxis"][()]

    def attrs(self, group: str = "FullSpectra") -> dict:
        """Attributes of a group (single values unpacked)."""
        with self._file() as f:
            return {k: _scalar(v) for k, v in f[group].attrs.items()}

    def sum_spectrum(self) -> np.ndarray | None:
        """``FullSpectra/SumSpectrum`` (sum of all spectra written by the acquisition), None if absent."""
        with self._file() as f:
            return f["FullSpectra/SumSpectrum"][()] if "FullSpectra/SumSpectrum" in f else None

    def acquisition_log(self) -> list[str]:
        """Texts of ``AcquisitionLog/Log`` (empty if absent or not readable)."""
        with self._file() as f:
            if "AcquisitionLog/Log" not in f:
                return []
            log = f["AcquisitionLog/Log"][()]
        if log.dtype.names is None or "logtext" not in log.dtype.names:
            return []
        return [_scalar(text) if isinstance(text, bytes) else str(text) for text in log["logtext"]]

    # --- instrument channels (TPS2) --------------------------------------------

    def tps2_names(self) -> list[str]:
        """Names of the ``TPS2/TwData`` channels (``TPS2/TwInfo``), empty if absent."""
        with self._file() as f:
            if "TPS2/TwData" not in f or "TPS2/TwInfo" not in f:
                return []
            return [_scalar(x) if isinstance(x, bytes) else str(x) for x in f["TPS2/TwInfo"][()]]

    def read_tps2(self, names: list[str], scans: slice | None = None,
                  every_write: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Values of the ``TPS2`` channels whose name starts with ``names`` (missing ones left out).

        ``TPS2/TwData`` is stored in one chunk per scan: reading a whole run
        takes several s on a slow disk. With ``every_write``, only the first
        scan of each write is read (every ~3 s, ~20x faster).

        :param scans: written scans to read (default all).
        :return: the scan indices read and {name: values}.
        """
        start, stop, _ = (scans or slice(None)).indices(self.n_scans)
        stop = max(start, stop)
        info = self.tps2_names()
        found = {}
        for name in names:
            index = [i for i, n in enumerate(info) if n.startswith(name)]
            if index:
                found[name] = index[0]
        if not found:
            return np.arange(0), {}
        with self._file() as f:
            dataset = f["TPS2/TwData"]
            if dataset.ndim != 3 or dataset.shape[1] != self.n_bufs:
                values = dataset[()].reshape(-1, dataset.shape[-1])[start:stop]
                index = np.arange(start, stop)
                if every_write:
                    index, values = index[:: self.n_bufs], values[:: self.n_bufs]
            elif every_write:
                w0, w1 = -(-start // self.n_bufs), -(-stop // self.n_bufs)  # writes starting in [start, stop[
                values = dataset[w0:w1, 0, :]
                index = np.arange(w0, w1) * self.n_bufs
            else:
                # all channels at once: h5py fancy indexing on the last axis is ~10x slower
                w0, w1 = start // self.n_bufs, -(-stop // self.n_bufs)
                values = dataset[w0:w1].reshape(-1, dataset.shape[-1])
                values = values[start - w0 * self.n_bufs: stop - w0 * self.n_bufs]
                index = np.arange(start, stop)
        return index, {name: values[:, i].astype(float) for name, i in found.items()}

    # --- spectra ----------------------------------------------------------------

    def rt_range(self, rt_start: float | None = None, rt_stop: float | None = None) -> tuple[float, float]:
        """``rt_start``, ``rt_stop`` clipped to the written scans (default: the whole run).

        :raises ValueError: empty range.
        """
        t = self.time_axis
        if len(t) == 0:
            raise ValueError(f"{self.path.name}: no scan written")
        start = float(t[0]) if rt_start is None else max(float(rt_start), float(t[0]))
        stop = float(t[-1]) if rt_stop is None else min(float(rt_stop), float(t[-1]))
        if stop <= start:
            raise ValueError(f"Empty rt window {start:g} - {stop:g} s (run: {t[0]:.1f} - {t[-1]:.1f} s)")
        return start, stop

    def scan_range(self, rt_start: float | None = None, rt_stop: float | None = None) -> slice:
        """Written scans with ``rt_start <= rt <= rt_stop``."""
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
        return self.read_scans(self.scan_range(rt_start, rt_stop), tof_start, tof_stop, segments)

    def resolve_segments(self, segments: tuple[int, int] | None = None) -> tuple[int, int]:
        """Segment range [start, stop[ to read: ``segments``, or (0, 1) for a file with one segment.

        :raises ValueError: no ``segments`` for a file with several, or out of range.
        """
        if segments is None:
            if self.n_segments != 1:
                raise ValueError(f"{self.path.name} has {self.n_segments} segments: give the segments to read.")
            return 0, 1
        s0, s1 = (int(v) for v in segments)
        if not 0 <= s0 < s1 <= self.n_segments:
            raise ValueError(f"Invalid segments {segments}, the file has {self.n_segments}.")
        return s0, s1

    def read_scans(
        self,
        scans: slice,
        tof_start: int = 0,
        tof_stop: int | None = None,
        segments: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spectra of the written scans ``scans`` (a slice of :attr:`time_axis`), as :meth:`read`."""
        if scans.step not in (None, 1):
            raise ValueError("read_scans: slices with a step are not supported")
        s0, s1 = self.resolve_segments(segments)
        tof_stop = self.n_points if tof_stop is None else min(tof_stop, self.n_points)
        tof_start = max(0, tof_start)
        start, stop, _ = scans.indices(self.n_scans)
        n = max(0, stop - start)
        rt = self.time_axis[start:start + n]
        if n == 0 or tof_stop <= tof_start:
            return np.zeros((n, max(0, tof_stop - tof_start)), np.float32), rt
        w0 = start // self.n_bufs
        w1 = (stop - 1) // self.n_bufs + 1
        with self._file() as f:
            data = f["FullSpectra/TofData"][w0:w1, :, s0:s1, tof_start:tof_stop]
        data = data.sum(axis=2, dtype=data.dtype) if s1 - s0 > 1 else data[:, :, 0]
        data = data.reshape(-1, tof_stop - tof_start)
        offset = start - w0 * self.n_bufs
        return data[offset:offset + n], rt

    def iter_blocks(
        self,
        block: int = 600,
        tof_start: int = 0,
        tof_stop: int | None = None,
        segments: tuple[int, int] | None = None,
        scans: slice | None = None,
    ) -> Iterator[tuple[slice, np.ndarray, np.ndarray]]:
        """Read the written scans ``scans`` (default all) in consecutive blocks of about ``block`` scans.

        The blocks end at the end of a write (rounded up to whole writes: the
        file is stored in one chunk per spectrum and write), so no scan is read
        twice. Use inside :meth:`open` to keep the file open.

        :return: iterator of (scans of the block, rt, intensity) as :meth:`read_scans`.
        """
        start, stop, _ = (scans or slice(None)).indices(self.n_scans)
        size = max(1, -(-block // self.n_bufs)) * self.n_bufs
        with self.open():
            b0 = start
            while b0 < stop:
                b1 = min(stop, (b0 // self.n_bufs) * self.n_bufs + size)
                data, rt = self.read_scans(slice(b0, b1), tof_start, tof_stop, segments)
                yield slice(b0, b1), rt, data
                b0 = b1
