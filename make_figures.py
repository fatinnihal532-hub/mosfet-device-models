"""Regenerate docs/*.svg and results/*.csv."""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mosdev import MOSCap, MOSFET, extract

plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fixed", "font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, GOLD, RED, TEAL, GREY = "#1F3864", "#C9971C", "#B3261E", "#2E7D6B", "#777777"


def compact_svg(path):
    s = open(path).read()
    s = re.sub(r"<metadata>.*?</metadata>\s*", "", s, flags=re.S)
    s = re.sub(r"-?\d+\.\d{2,}", lambda m: f"{float(m.group()):.1f}".rstrip("0").rstrip("."), s)
    s = re.sub(r"\n\s+", "\n", s)
    open(path, "w").write(s)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"docs/{name}.svg")
    compact_svg(f"docs/{name}.svg")
    plt.close(fig)


os.makedirs("docs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. MOS capacitor: C-V and Poisson cross-check
cap = MOSCap(1e17, 2.0)
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
vg = np.linspace(cap.vfb - 2.0, cap.vth_exact() + 2.0, 260)
lf, hf = cap.cv(vg)
a.plot(vg, lf / cap.cox, color=NAVY, lw=2, label="low frequency")
a.plot(vg, hf / cap.cox, color=GOLD, lw=2, label="high frequency")
a.axvline(cap.vfb, color=GREY, ls=":", lw=1)
a.axvline(cap.vth_exact(), color=GREY, ls=":", lw=1)
a.text(cap.vfb, 0.5, "Vfb ", color=GREY, fontsize=8, ha="right")
a.text(cap.vth_exact(), 0.5, " Vth", color=GREY, fontsize=8)
a.set_xlabel("Gate voltage (V)")
a.set_ylabel("C / Cox")
a.set_ylim(0, 1.05)
a.set_title("C-V, p-type Na = 1e17 cm-3, tox = 2 nm", loc="left", fontsize=10, fontweight="bold")
a.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.4))
psi = np.array([p for p in np.linspace(-0.2, 1.0, 13) if abs(p) > 1e-6])
pp = np.linspace(-0.2, 1.0, 400)
b.semilogy(pp[np.abs(pp) > 1e-3], np.abs(cap.qs(pp[np.abs(pp) > 1e-3])), color=NAVY, lw=2, label="analytic charge sheet")
b.semilogy(psi, [abs(cap.qs_numeric(p)) for p in psi], "o", color=RED, ms=4.5, label="finite-difference Poisson")
b.set_xlabel("Surface potential (V)")
b.set_ylabel("|Qs| (C/cm2)")
b.set_title("Two solvers agree", loc="left", fontsize=10, fontweight="bold")
b.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "moscap")

# 2. transfer characteristics
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
vg = np.linspace(0, 1.0, 600)
for L, c in ((1000, GREY), (100, NAVY), (60, RED)):
    m = MOSFET(1.0, L)
    a.semilogy(vg, m.id(vg, 1.0) * 1e6, color=c, lw=2, label=f"L = {L} nm")
    b.plot(vg, m.id(vg, 1.0) * 1e6, color=c, lw=2)
a.set_ylim(1e-9, 3e3)
a.set_xlabel("Vgs (V)")
a.set_ylabel("Id per um width (uA/um)")
a.set_title("Bulk planar, Vds = 1 V", loc="left", fontsize=10, fontweight="bold")
a.legend(frameon=False, fontsize=8, loc="lower right")
b.set_xlabel("Vgs (V)")
b.set_ylabel("Id (uA/um)")
b.set_title("Same curves, linear", loc="left", fontsize=10, fontweight="bold")
save(fig, "transfer")

# 3. scaling study
Ls = np.array([200, 150, 120, 100, 80, 60, 50, 40, 30, 25, 20, 15])
vgs = np.linspace(0, 1.0, 2001)
rows = []
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.4))
for arch, c in (("bulk", RED), ("soi", GOLD), ("dg", TEAL)):
    dv, di, ss = [], [], []
    for L in Ls:
        m = MOSFET(1.0, float(L), arch=arch)
        dv.append(m.vth(0.05))
        di.append(extract.dibl(m.vth(0.05), m.vth(1.0), 0.05, 1.0))
        ss.append(extract.subthreshold_swing(vgs, m.id(vgs, 1.0), 1e-16))
        rows.append((arch, L, m.vth(0.05), di[-1], ss[-1], m.id(0, 1.0) * 1e9))
    lbl = {"bulk": "bulk planar", "soi": "fully depleted SOI (6 nm)", "dg": "double gate (6 nm)"}[arch]
    ax[0].plot(Ls, dv, color=c, lw=2, label=lbl)
    ax[1].semilogy(Ls, np.maximum(di, 0.05), color=c, lw=2)
    ax[2].plot(Ls, ss, color=c, lw=2)
ax[0].set_ylabel("Vth at Vds = 50 mV (V)")
ax[1].set_ylabel("DIBL (mV/V)")
ax[2].set_ylabel("Subthreshold swing (mV/dec)")
ax[2].set_ylim(55, 150)
ax[0].set_ylim(-0.1, 0.4)
for x in ax:
    x.set_xlabel("Gate length (nm)")
    x.invert_xaxis()
ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")
ax[0].set_title("Vth roll-off", loc="left", fontsize=10, fontweight="bold")
ax[1].set_title("DIBL", loc="left", fontsize=10, fontweight="bold")
ax[2].set_title("Swing", loc="left", fontsize=10, fontweight="bold")
save(fig, "scaling")

with open("results/scaling.csv", "w") as fh:
    fh.write("arch,l_nm,vth_v,dibl_mv_per_v,ss_mv_per_dec,ioff_na_per_um\n")
    for r in rows:
        fh.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.1f},{r[4]:.1f},{r[5]:.3f}\n")
print("figures in docs/, tables in results/")
