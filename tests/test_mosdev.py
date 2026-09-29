import numpy as np
import pytest
from mosdev import MOSCap, MOSFET, extract, VT, EPS_SI, phi_f
from mosdev.constants import Q


@pytest.fixture(scope="module")
def cap():
    return MOSCap(1e17, 2.0)


def test_fermi_potential():
    assert phi_f(1e17) == pytest.approx(0.4167, abs=1e-3)


def test_max_depletion_width(cap):
    w = np.sqrt(4 * EPS_SI * cap.phi_f / (Q * cap.na))
    assert cap.wd_max() == pytest.approx(w, rel=1e-12)
    assert 90e-7 < cap.wd_max() < 120e-7          # about 104 nm for 1e17


def test_threshold_matches_textbook(cap):
    assert cap.vth_exact() == pytest.approx(cap.vth_textbook(), abs=1e-6)


@pytest.mark.parametrize("psi", [-0.2, -0.05, 0.1, 0.3, 0.6, 0.8, 0.95])
def test_analytic_vs_numeric_poisson(cap, psi):
    a, n = float(cap.qs(psi)), cap.qs_numeric(psi)
    assert n == pytest.approx(a, rel=2e-3)


def test_gate_voltage_monotonic_and_invertible(cap):
    psi = np.linspace(-0.3, 1.1, 300)
    v = cap.vg(psi)
    assert np.all(np.diff(v) > 0)
    for p in (-0.1, 0.4, 0.9):
        assert cap.psi_s(float(cap.vg(p))) == pytest.approx(p, abs=1e-9)


def test_cv_limits(cap):
    lf, hf = cap.cv(np.array([cap.vfb - 3.0, cap.vth_exact() + 1.5]))
    assert lf[0] == pytest.approx(cap.cox, rel=0.03)      # accumulation
    assert lf[1] == pytest.approx(cap.cox, rel=0.04)      # inversion, low frequency
    assert hf[0] == pytest.approx(cap.cox, rel=0.03)
    assert hf[1] == pytest.approx(cap.cmin_hf(), rel=1e-6)  # inversion, high frequency stays at Cmin


def test_flat_band_capacitance(cap):
    cfb_s = EPS_SI / cap.ld
    ref = cap.cox * cfb_s / (cap.cox + cfb_s)
    lf, hf = cap.cv(np.array([cap.vfb]))
    assert lf[0] == pytest.approx(ref, rel=0.02)


def test_long_channel_saturation_square_law():
    m = MOSFET(1.0, 10000.0, theta=0.0, vsat=1e12, short_channel=False)
    vov = 0.5
    beta = m.mu0 * m.cox * m.w / m.l
    ref = beta / (2 * m.n_long) * vov ** 2
    assert float(m.id(m.vth_long + vov, 1.2)) == pytest.approx(ref, rel=0.03)


def test_linear_region_conductance():
    m = MOSFET(1.0, 10000.0, theta=0.0, vsat=1e12, short_channel=False)
    vov, vds = 0.6, 1e-3
    beta = m.mu0 * m.cox * m.w / m.l
    assert float(m.id(m.vth_long + vov, vds)) == pytest.approx(beta * vov * vds, rel=0.03)


def test_subthreshold_swing_equals_n_kT_ln10():
    m = MOSFET(1.0, 10000.0, short_channel=False)
    vg = np.linspace(0, 0.6, 3000)
    ss = extract.subthreshold_swing(vg, m.id(vg, 1.0), 1e-16)
    assert ss == pytest.approx(m.n_long * VT * np.log(10) * 1e3, rel=0.005)
    assert 59.5 < ss < 100


def test_reverse_drain_flips_sign():
    m = MOSFET(1.0, 1000.0, short_channel=False)
    assert float(m.id(0.8, -0.3)) == pytest.approx(-float(m.id(0.8, 0.3)))


def test_scale_length_factor():
    m = MOSFET(1.0, 100.0)
    assert m.theta_l() == pytest.approx(1 / (2 * np.cosh(m.l / (2 * m.lam)) - 2))
    assert MOSFET(1.0, 5000.0).theta_l() < 1e-12


def test_rolloff_and_dibl_grow_as_l_shrinks():
    prev_v, prev_d = 1e9, -1
    for L in (400, 200, 120, 80):
        m = MOSFET(1.0, L)
        v = m.vth(0.05)
        d = extract.dibl(m.vth(0.05), m.vth(1.0), 0.05, 1.0)
        assert v <= prev_v + 1e-12 and d >= prev_d
        prev_v, prev_d = v, d


def test_architecture_ordering_at_30nm():
    dibl = {}
    for a in ("bulk", "soi", "dg"):
        m = MOSFET(1.0, 30.0, arch=a)
        dibl[a] = extract.dibl(m.vth(0.05), m.vth(1.0), 0.05, 1.0)
    assert dibl["bulk"] > dibl["soi"] > dibl["dg"] > 0


def test_extraction_tracks_model_threshold():
    vg = np.linspace(0, 1.0, 2001)
    m = MOSFET(1.0, 1000.0)
    i = m.id(vg, 0.05)
    v_gm = extract.vth_gm_max(vg, i)
    v_cc = extract.vth_const_current(vg, i, m.w / m.l)
    assert abs(v_gm - m.vth(0.05)) < 0.06 and abs(v_cc - m.vth(0.05)) < 0.1
    m2 = MOSFET(1.0, 80.0)
    i2 = m2.id(vg, 0.05)
    assert extract.vth_gm_max(vg, i2) < v_gm             # extraction sees the roll-off


def test_current_monotonic_in_vgs_and_vds():
    m = MOSFET(1.0, 100.0)
    vg = np.linspace(0, 1.0, 200)
    assert np.all(np.diff(m.id(vg, 0.8)) > 0)
    vd = np.linspace(0, 1.0, 200)
    assert np.all(np.diff(m.id(0.8, vd)) >= 0)
