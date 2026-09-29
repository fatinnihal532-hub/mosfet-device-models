"""Parameter extraction from simulated or measured transfer curves."""
import numpy as np


def vth_gm_max(vg, i):
    """Threshold by linear extrapolation at the point of maximum transconductance (low Vds)."""
    gm = np.gradient(i, vg)
    k = int(np.argmax(gm))
    return vg[k] - i[k] / gm[k]


def vth_const_current(vg, i, w_over_l, i0=1e-7):
    """Constant-current threshold: Vg where Id = i0 * W/L."""
    target = i0 * w_over_l
    return float(np.interp(np.log(target), np.log(i), vg))


def subthreshold_swing(vg, i, floor=1e-13):
    """Minimum local swing in mV/dec over the transfer curve (points with i > floor)."""
    m = i > floor
    x, y = vg[m], np.log10(i[m])
    slope = np.gradient(y, x)
    return 1e3 / np.max(slope)


def dibl(vth_low, vth_high, vds_low, vds_high):
    """mV/V."""
    return 1e3 * (vth_low - vth_high) / (vds_high - vds_low)
