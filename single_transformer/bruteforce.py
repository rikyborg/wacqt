from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, Boltzmann
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import decimate

import simulator as sim

_wc = 2. * np.pi * 6.0296e9
_kappa = 2. * np.pi * 517e3
_Qb = 1e7
_chi = -2. * np.pi * 143e3
_Ql = _wc / _kappa
_AMP = 1.738e-5  # V, will be adjusted later to reach max_ph
max_ph = 227
detuning = 10.
double = True
nr_freqs = 4

res, para_g, para_e = sim.SimulationParameters.from_measurement(
    _wc, _chi, _Qb, _Ql)
assert res.success

w_g, Q_g = para_g.calculate_resonance()
w_e, Q_e = para_e.calculate_resonance()
chi = 0.5 * (w_e - w_g)
w_c = 0.5 * (w_g + w_e)
kappa_g = w_g / Q_g
kappa_e = w_e / Q_e
kappa = 0.5 * (kappa_g + kappa_e)


def demodulate(signal, nc, bw=None):
    if bw is None:
        bw = nc
    ns = len(signal)
    s_fft = np.fft.rfft(signal) / ns
    e_fft = np.zeros(bw, dtype=np.complex128)
    karray = np.arange(nc - bw // 2, nc + bw // 2 + 1)
    e_fft[karray - nc] = s_fft[karray]
    envelope = np.fft.ifft(e_fft) * bw
    return envelope


def demodulate_time(t, bw):
    t0 = t[0]
    dt = t[1] - t[0]
    t1 = t[-1] + dt
    t_e = np.linspace(t0, t1, bw, endpoint=False)
    return t_e


def erf(p):
    _dfc = p
    print(_dfc)
    # fc, df = para_g.tune(_fc + _dfc * _chi, _df, priority='f')
    fc, df = _fc + _dfc * _chi, _df
    para_g.set_df(df)
    para_e.set_df(df)

    para_g.set_Nbeats(4)
    para_e.set_Nbeats(4)

    t = para_g.get_time_arr()
    t_drive = para_g.get_drive_time_arr()
    AMP = _AMP
    for ii in range(3):
        carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
        window = np.zeros_like(t_drive)
        idx = np.s_[0 * para_g.ns:2 * para_g.ns]
        if double:
            w_m = 2 * np.pi * df / 2
        else:
            w_m = 2 * np.pi * df / 4
        window[idx] = np.sin(w_m * t_drive[idx])**(nr_freqs - 1)
        drive = window * carrier

        para_g.set_drive_V(drive)
        para_e.set_drive_V(drive)

        sol_g = para_g.simulate(print_time=False)  # , continue_run=True)
        P1_g = sol_g[:, 1]
        sol_e = para_e.simulate(print_time=False)  # , continue_run=True)
        P1_e = sol_e[:, 1]

        nc = int(round(fc / df))
        Nbeats = para_g.Nbeats
        t_envelope = demodulate_time(t, nc * Nbeats)
        P1_g_envelope = demodulate(P1_g, nc * Nbeats)
        P1_e_envelope = demodulate(P1_e, nc * Nbeats)

        nr_ph_g = 0.5 * 2. * np.abs(P1_g_envelope)**2 / para_g.L1 / (
            hbar * w_g)
        nr_ph_e = 0.5 * 2. * np.abs(P1_e_envelope)**2 / para_e.L1 / (
            hbar * w_e)
        idx_check = np.argmin(np.abs(t_envelope - t[idx.stop]))
        left_ph_g = nr_ph_g[idx_check]
        left_ph_e = nr_ph_e[idx_check]

        AMP = AMP * np.sqrt(max_ph / max(nr_ph_g.max(), nr_ph_e.max()))

    return max(left_ph_g, left_ph_e)


_fc = w_c / (2. * np.pi)
_df = 2. * np.abs(detuning * kappa) / (2. * np.pi)

# res = least_squares(erf, [0.001], bounds=(-1., +1.))
res = minimize_scalar(erf, method='Bounded', bounds=(-2, 2))
