"""Extract the peaks of a retention time window of an h5 file (edit the values below).

    python scripts/extract_rt_window.py

Same as::

    twutils -v extract /media/Shuttle/TOF-Data/2020/200602.1550.std.7.h5 --rt 1645 4.5 -j 8
"""

import logging

import peaxart as px

import twutils

FILE = "/media/Shuttle/TOF-Data/2020/200602.1550.std.7.h5"
RT = 1645.0          # s
HALF_WIDTH = 4.5     # s
MASSES = None        # e.g. [101, 103]; None: all the masses of the file
N_JOBS = 8

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logging.getLogger("twutils").setLevel(logging.INFO)

    result = twutils.extract(
        FILE,
        rt_start=RT - HALF_WIDTH,
        rt_stop=RT + HALF_WIDTH,
        masses=MASSES,
        mz_range=0.5,
        config=px.ExtractionConfig(),  # .legacy() for the results of the legacy alpinac
        n_jobs=N_JOBS,
    )
    print(result.peaks[["mass", "rt", "mz", "area", "sn", "compound"]].to_string())

    out = twutils.default_output(FILE, RT - HALF_WIDTH, RT + HALF_WIDTH)
    out.parent.mkdir(exist_ok=True)
    result.to_alpinac(out, metadata={"operator": "coli"})
    print(f"Written {out}")
