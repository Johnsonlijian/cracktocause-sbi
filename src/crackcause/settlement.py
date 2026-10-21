"""Settlement profiles imposed on the base of the wall."""
import numpy as np


def hinge_profile(x, x_p, w, x_ref=30.0):
    """Smoothed hinge g(x)/S: linear settlement towards the free edge x = 0, zero beyond the pivot x_p,
    slope change spread over the width w; normalized to -1 at x_ref."""
    f = lambda z: w * np.logaddexp(0.0, (x_p - z) / w)  # noqa: E731
    return -f(x) / f(x_ref)
