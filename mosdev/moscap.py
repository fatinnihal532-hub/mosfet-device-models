"""MOS capacitor: exact 1-D surface-potential model, plus an independent finite-difference Poisson solver.

Convention: p-type substrate, x into the silicon, psi measured from the neutral bulk (psi_s > 0 is
depletion/inversion, psi_s < 0 is accumulation). Qs is the charge per area in the silicon (negative
in depletion and inversion), and the gate carries -Qs.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import solve_banded
from .constants import Q, VT, NI, EPS_SI, EPS_OX, phi_f


class MOSCap:
    def __init__(self, na=1e17, tox_nm=2.0, phi_ms=None):
        """na in cm^-3, tox in nm. phi_ms is the gate-semiconductor work-function difference (V).
        Default is an n+ poly gate: phi_ms = -(Eg/2 + phi_F)."""
        self.na = na
        self.cox = EPS_OX / (tox_nm * 1e-7)              # F/cm^2
        self.phi_f = phi_f(na)
        self.phi_ms = -(0.56 + self.phi_f) if phi_ms is None else phi_ms
        self.vfb = self.phi_ms                            # no oxide charge
        self.ld = np.sqrt(EPS_SI * VT / (Q * na))         # extrinsic Debye length, cm
        self.r = (NI / na) ** 2

    # exact charge sheet ---------------------------------------------------------------
    def _f(self, psi):
        b = psi / VT
        arg = np.exp(-b) + b - 1.0 + self.r * (np.exp(b) - b - 1.0)
        return np.sqrt(np.maximum(arg, 0.0))

    def qs(self, psi):
        """Silicon charge per area (C/cm^2), exact under the depletion-free-carrier statistics."""
        psi = np.asarray(psi, dtype=float)
        return -np.sign(psi) * np.sqrt(2.0) * VT * EPS_SI / self.ld * self._f(psi)

    def vg(self, psi):
        """Gate voltage that gives surface potential psi."""
        return self.vfb + psi - self.qs(psi) / self.cox

    def psi_s(self, vg):
        """Invert vg(psi) by bracketing (monotonic)."""
        return brentq(lambda p: self.vg(p) - vg, -1.2, 2.0, xtol=1e-14)

    # analytic reference points --------------------------------------------------------
    def wd_max(self):
        return np.sqrt(2 * EPS_SI * 2 * self.phi_f / (Q * self.na))

    def vth_textbook(self):
        """Threshold from the depletion approximation: Vfb + 2phiF + sqrt(2 q eps Na 2phiF)/Cox."""
        return self.vfb + 2 * self.phi_f + np.sqrt(2 * Q * EPS_SI * self.na * 2 * self.phi_f) / self.cox

    def vth_exact(self):
        """Gate voltage at psi_s = 2 phi_F on the exact charge sheet."""
        return float(self.vg(2 * self.phi_f))

    # capacitance ----------------------------------------------------------------------
    def cs_lf(self, psi, h=1e-5):
        return -(self.qs(psi + h) - self.qs(psi - h)) / (2 * h)

    def cv(self, vg):
        """Return (C_lf, C_hf) per area in F/cm^2 for an array of gate voltages."""
        vg = np.atleast_1d(vg)
        psi = np.array([self.psi_s(v) for v in vg])
        cs = self.cs_lf(psi)
        c_lf = self.cox * cs / (self.cox + cs)
        psi_c = np.minimum(psi, 2 * self.phi_f)           # depletion width freezes at Wd,max
        cs_hf = np.where(psi < 2 * self.phi_f, cs, EPS_SI / self.wd_max())
        cs_hf = np.where(psi < 0, cs, cs_hf)
        c_hf = self.cox * cs_hf / (self.cox + cs_hf)
        return c_lf, c_hf

    def cmin_hf(self):
        return self.cox * (EPS_SI / self.wd_max()) / (self.cox + EPS_SI / self.wd_max())

    # independent numerical Poisson solve ---------------------------------------------
    def qs_numeric(self, psi_s, n=1600, depth_ld=None):
        """Solve psi'' = -rho/eps with Newton + tridiagonal solves on a geometric grid.
        Returns the silicon charge per area by integrating rho."""
        p0 = self.na
        n0 = NI ** 2 / self.na
        depth = depth_ld if depth_ld else max(12 * self.ld, 4 * self.wd_max())
        h0 = self.ld * 1e-3
        ratio = 1.0
        # geometric grid from h0 to depth
        lo, hi = 1.0, 1.2
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if h0 * (mid ** n - 1) / (mid - 1) > depth:
                hi = mid
            else:
                lo = mid
        ratio = 0.5 * (lo + hi)
        x = np.concatenate([[0.0], np.cumsum(h0 * ratio ** np.arange(n))])
        psi = psi_s * np.exp(-x / max(self.ld, self.wd_max()))    # initial guess
        psi[0] = psi_s
        psi[-1] = 0.0
        h = np.diff(x)
        for _ in range(200):
            b = psi / VT
            rho = Q * (p0 * np.exp(-b) - n0 * np.exp(b) - (p0 - n0))
            drho = Q * (-p0 * np.exp(-b) - n0 * np.exp(b)) / VT
            # interior nodes 1..N-1; F = psi'' + rho/eps
            hm, hp = h[:-1], h[1:]
            lower = 2.0 / (hm * (hm + hp))
            upper = 2.0 / (hp * (hm + hp))
            diag = -(lower + upper) + drho[1:-1] / EPS_SI
            res = lower * psi[:-2] - (lower + upper) * psi[1:-1] + upper * psi[2:] + rho[1:-1] / EPS_SI
            ab = np.zeros((3, len(diag)))
            ab[0, 1:] = upper[:-1]
            ab[1] = diag
            ab[2, :-1] = lower[1:]
            d = solve_banded((1, 1), ab, -res)
            step = np.clip(d, -0.2, 0.2)
            psi[1:-1] += step
            if np.max(np.abs(d)) < 1e-12:
                break
        b = psi / VT
        rho = Q * (p0 * np.exp(-b) - n0 * np.exp(b) - (p0 - n0))
        return float(np.trapezoid(rho, x)) if hasattr(np, "trapezoid") else float(np.trapz(rho, x))
