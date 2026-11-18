"""One dataset-hash function, used everywhere.

The first validation run rejected every candidate, including known-good ones,
because the freezer hashed arrays in sorted key order and the contract hashed
them in literal order. Two correct implementations that disagree are worse
than one, since the failure looks like a fail-closed contract working
correctly. Clause 1 is only meaningful if producer and checker compute the
same bytes, so there is exactly one place that decides.
"""

import hashlib

import numpy as np


def dataset_sha256(**arrays) -> str:
    """Hash named arrays in sorted-key order."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()
