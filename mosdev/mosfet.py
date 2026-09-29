"""Charge-based compact MOSFET model (EKV-style interpolation) with a scale-length short-channel correction.

Long channel: forward/reverse normalised currents i = ln^2(1 + exp(v/2)) give one continuous expression from
weak to strong inversion. Short channel: a 2-D scale-length term lowers Vth with shorter L and higher Vds
(the source of Vth roll-off and DIBL), and the subthreshold slope factor grows with it.
"""
import numpy as np
from .constants import Q, VT, EPS_SI, EPS_OX
from .moscap import MOSCap


def _lnsq(x):
    """ln^2(1 + e^x), stable for large |x|."""
    return np.where(x > 30, x * x, np.log1p(np.exp(np.minimum(x, 30.0))) ** 2)


class MOSFET:
    def __init__(self, w_um=1.0, l_nm=100.0, na=5e17, tox_nm=1.5, mu0=250.0, theta=0.6,
                 vsat=8e6, nd_sd=1e20, vth_target=0.35, short_channel=True, arch="bulk", tsi_nm=6.0, n_body=0.05):
        """mu0 in cm^2/Vs, theta in 1/V (mobility degradation), vsat in cm/s.
        The gate work function is set so the long-channel threshold equals vth_target (a metal gate).
        arch: "bulk" (planar, depletion depth sets the scale), "soi" (single-gate fully depleted body of
        thickness tsi) or "dg" (double gate / FinFET-like). For the fully depleted cases the body factor
        is n = 1 + n_body (undoped floating body: an assumption, not a fit)."""
        assert arch in ("bulk", "soi", "dg")
        self.arch, self.tsi = arch, tsi_nm * 1e-7
        c0 = MOSCap(na, tox_nm)
        self.cap = MOSCap(na, tox_nm, phi_ms=vth_target - 2 * c0.phi_f
                          - np.sqrt(2 * Q * EPS_SI * na * 2 * c0.phi_f) / c0.cox)
        self.w = w_um * 1e-4
        self.l = l_nm * 1e-7
        self.na, self.tox = na, tox_nm * 1e-7
        self.mu0, self.theta, self.vsat = mu0, theta, vsat
        self.short = short_channel
        cap = self.cap
        self.vth_long = vth_target
        self.cox = cap.cox
        self.cdep = EPS_SI / cap.wd_max()
        self.n_long = 1.0 + (self.cdep / self.cox if arch == "bulk" else n_body)
        self.vbi = VT * np.log(nd_sd * na / (1e10) ** 2)
        if arch == "bulk":
            self.lam = np.sqrt(EPS_SI / EPS_OX * self.tox * cap.wd_max())      # scale length, cm
        elif arch == "soi":
            self.lam = np.sqrt(EPS_SI / EPS_OX * self.tox * self.tsi)
        else:
            self.lam = np.sqrt(EPS_SI / (2 * EPS_OX) * self.tox * self.tsi)

    # --- short-channel pieces ---------------------------------------------------------
    def theta_l(self):
        """Geometric factor 1/(2cosh(L/2lambda) - 2) from the 2-D scale-length solution."""
        return 1.0 / (2.0 * np.cosh(self.l / (2.0 * self.lam)) - 2.0)

    def vth(self, vds=0.0):
        """Threshold voltage including roll-off and DIBL."""
        if not self.short:
            return self.vth_long
        psi = 2 * self.cap.phi_f
        return self.vth_long - (2.0 * (self.vbi - psi) + abs(vds)) * self.theta_l()

    def n_eff(self):
        return self.n_long * (1.0 + (self.theta_l() if self.short else 0.0))

    # --- drain current ----------------------------------------------------------------
    def id(self, vgs, vds):
        """Drain current in A for an n-channel device (vds >= 0 forward, < 0 reversed)."""
        vgs = np.asarray(vgs, dtype=float)
        vds = np.asarray(vds, dtype=float)
        sign = np.sign(vds)
        vd = np.abs(vds)
        n = self.n_eff()
        vth = self.vth(vd)
        vp = (vgs - vth) / n
        vov = np.maximum(vgs - vth, 0.0)
        mu = self.mu0 / (1.0 + self.theta * vov)
        beta = mu * self.cox * self.w / self.l
        # velocity saturation: the drain voltage is capped at Vdsat' = Vov*EcL/(Vov + EcL) (smoothly), and the
        # mobility is divided by 1 + Vds/EcL, with Ec = 2 vsat / mu
        ec_l = 2.0 * self.vsat / mu * self.l
        vp_pos = np.maximum(vp, 0.0)                        # saturation voltage without velocity limit
        vdsat = vp_pos * ec_l / (vp_pos + ec_l) + 4 * VT
        vde = vd / (1.0 + (vd / vdsat) ** 6) ** (1.0 / 6.0)
        i_f = _lnsq(vp / (2 * VT))
        i_r = _lnsq((vp - vde) / (2 * VT))
        i0 = 2.0 * n * beta * VT ** 2
        idd = i0 * (i_f - i_r) / (1.0 + vde / ec_l)
        return sign * idd
