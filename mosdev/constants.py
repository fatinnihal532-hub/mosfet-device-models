"""Physical constants and silicon parameters (300 K, CGS-style units: cm, F/cm, C/cm^2)."""
import numpy as np

Q = 1.602176634e-19          # C
KB = 1.380649e-23            # J/K
EPS0 = 8.8541878128e-14      # F/cm
T = 300.0
VT = KB * T / Q              # thermal voltage, V (0.02585)
NI = 1.0e10                  # intrinsic density of Si at 300 K, cm^-3
EPS_SI = 11.7 * EPS0
EPS_OX = 3.9 * EPS0


def phi_f(na):
    """Bulk Fermi potential of p-type silicon, V."""
    return VT * np.log(na / NI)
