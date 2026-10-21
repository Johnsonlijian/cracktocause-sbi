"""Bookkeeping for simulation campaigns run by independent worker processes.

Simulation i is stored in OUT_DIR/sims/sim_XXXXXX.npz. A worker claims an index by creating
sim_XXXXXX.npz.lock exclusively, so any number of workers can be started or stopped at any time and
fill the gaps in index order; results are written to a hidden temporary file and renamed, so readers
never see a partially written file. Locks left by killed workers are removed with
`clear_stale_locks` while no worker runs.
"""
import glob
import os
import socket
import time

import numpy as np


def claim(path):
    """Lock simulation `path`; returns the lock file name, or None if another worker holds it."""
    lock = path + '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f'{socket.gethostname()} {os.getpid()} {time.time():.0f}\n'.encode())
    os.close(fd)
    return lock


def release(lock):
    if lock and os.path.exists(lock):
        os.remove(lock)


def save_atomic(path, **arrays):
    """np.savez_compressed to a hidden temporary file in the same directory, then rename."""
    d, name = os.path.split(path)
    tmp = os.path.join(d, '.tmp_' + name)                 # keeps the .npz suffix; not matched by sim_*.npz
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def pending(out_dir, start, n_sims):
    """Indices in [start, start + n_sims) without a result, in ascending order (re-evaluated lazily)."""
    for i in range(start, start + n_sims):
        if not os.path.exists(os.path.join(out_dir, 'sims', f'sim_{i:06d}.npz')):
            yield i


def clear_stale_locks(out_dir):
    """Remove all lock and temporary files (call only while no worker of this campaign runs)."""
    n = 0
    for f in glob.glob(os.path.join(out_dir, 'sims', '*.lock')) + glob.glob(os.path.join(out_dir, 'sims', '.tmp_*')):
        os.remove(f)
        n += 1
    return n
