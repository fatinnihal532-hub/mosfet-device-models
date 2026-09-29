"""Independent checks. Exit code 1 if any fails."""
import sys
import numpy as np
from mosdev import MOSCap, MOSFET, extract, VT
from mosdev.constants import Q, EPS_SI

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    fails += not ok


cap = MOSCap(1e17, 2.0)
check("Vth: exact charge sheet at psi=2phiF equals the depletion-approximation formula",
      abs(cap.vth_exact() - cap.vth_textbook()) < 1e-6, f"({cap.vth_exact():.4f} V)")
worst = max(abs(cap.qs_numeric(p) / float(cap.qs(p)) - 1) for p in np.linspace(-0.2, 0.95, 24) if abs(p) > 1e-6)
check("Surface charge: analytic vs finite-difference Poisson solve, 24 points", worst < 2e-3, f"(worst {worst:.1e})")
lf, hf = cap.cv(np.array([cap.vfb - 3.0, cap.vfb, cap.vth_exact() + 1.5]))
check("C-V limits: LF accumulation and inversion near Cox", abs(lf[0] / cap.cox - 1) < 0.03 and abs(lf[2] / cap.cox - 1) < 0.04)
check("C-V: HF inversion capacitance equals Cmin", abs(hf[2] / cap.cmin_hf() - 1) < 1e-6, f"({cap.cmin_hf()*1e6:.3f} uF/cm2 vs Cox {cap.cox*1e6:.3f})")
m = MOSFET(1.0, 10000.0, theta=0.0, vsat=1e12, short_channel=False)
ref = m.mu0 * m.cox * m.w / m.l / (2 * m.n_long) * 0.25
check("Long channel: saturation current follows the square law", abs(float(m.id(m.vth_long + 0.5, 1.2)) / ref - 1) < 0.03)
vg = np.linspace(0, 0.6, 3000)
ss = extract.subthreshold_swing(vg, m.id(vg, 1.0), 1e-16)
check("Weak inversion: swing equals n*kT/q*ln10", abs(ss / (m.n_long * VT * np.log(10) * 1e3) - 1) < 0.005, f"({ss:.2f} mV/dec)")
for a, L, lim in (("bulk", 100.0, 80.0), ("soi", 30.0, 100.0), ("dg", 20.0, 100.0)):
    d = MOSFET(1.0, L, arch=a)
    v = np.linspace(0, 1.0, 2001)
    s = extract.subthreshold_swing(v, d.id(v, 1.0), 1e-16)
    check(f"{a} at L={L:.0f} nm keeps SS below {lim:.0f} mV/dec (electrostatic control)", s < lim, f"({s:.1f})")
d = MOSFET(1.0, 30.0, arch="bulk")
check("Bulk planar at 30 nm has lost control (Vth < 0.1 V at high Vds)", d.vth(1.0) < 0.1, f"({d.vth(1.0):.2f} V)")
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
