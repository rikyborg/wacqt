from __future__ import absolute_import, division, print_function

import time

import numpy as np
from scipy.constants import hbar, Boltzmann

from utils import demodulate, demodulate_time, get_init_array

from simulators import sim_transformer as sim
# from simulators import sim_notch as sim
# from simulators import sim_reflection as sim
# from simulators import sim_transmission as sim

Nruns = 65536

_wc = 2. * np.pi * 6e9
_chi = 2. * np.pi * 2e6
_Qb = 1e7
_kappa = _chi / 10
_Ql = _wc / _kappa

AMP = 2.266e-6  # V
nr_freqs = 3
double = True
method = "sine"

save_filename = "fidelity_{:s}_chi_{:.0g}_kappa_{:.0g}_Nruns_{:d}.npz".format(
    sim.CONF,
    _chi / (2. * np.pi),
    _kappa / (2. * np.pi),
    Nruns,
)

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

Eph = hbar * w_c
Tph = 0.5 * Eph / Boltzmann

_fc = w_c / (2. * np.pi)
_df = 2. * np.abs(chi) / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
para_g.set_df(df)
para_e.set_df(df)


def inprod(f, g, t=None, dt=None):
    if t is not None:
        dt = t[1] - t[0]
        ns = len(t)
        T = ns * dt
    elif dt is not None:
        ns = len(f)
        T = ns * dt
    else:
        T = 1.
    return np.trapz(f * np.conj(g), x=t) / T


def norm(x, t=None, dt=None):
    return np.sqrt(np.real(inprod(x, x, t=t, dt=dt)))


def proj(v, u, t=None, dt=None):
    if not norm(u, t=t, dt=dt):
        return 0.
    else:
        return inprod(v, u, t=t, dt=dt) / inprod(u, u, t=t, dt=dt) * u


def dist(f, g, t=None, dt=None):
    return norm(f - g, t=t, dt=dt)


def get_envelopes(sol, para, fc):
    Vout = para.calculate_Vout(sol)

    df = para.df
    nc = int(round(fc / df))
    Nbeats = para.Nbeats
    t = para.get_time_arr()
    t_envelope = demodulate_time(t, nc * Nbeats)
    Vout_envelope = demodulate(Vout, nc * Nbeats)

    return (t_envelope, Vout_envelope)


# templates
print("Finding templates")
para_g.set_Nbeats(2)
para_e.set_Nbeats(2)

t = para_g.get_time_arr()
t_drive = para_g.get_drive_time_arr()
if method == "sine":
    carrier = np.cos(2. * np.pi * fc * t_drive)
    window = np.zeros_like(t_drive)
    if double:
        idx = np.s_[0 * para_g.ns:2 * para_g.ns]
    else:
        idx = np.s_[0 * para_g.ns:1 * para_g.ns]
    window[idx] = np.sin(2. * np.pi * 0.5 * df * t_drive[idx])**(nr_freqs - 1)
    _drive = window * carrier
else:
    raise NotImplementedError
drive = AMP * _drive
Vg = drive[:-1]

para_g.set_drive_V(drive)
para_e.set_drive_V(drive)
sol_g = para_g.simulate(print_time=True)
sol_e = para_e.simulate(print_time=True)
(t_envelope, Vout_g_envelope) = get_envelopes(sol_g, para_g, fc)
(t_envelope, Vout_e_envelope) = get_envelopes(sol_e, para_e, fc)
template_g = Vout_g_envelope.copy()
template_e = Vout_e_envelope.copy()
template_diff = template_e - template_g
# center_g = np.real(np.sum(np.conj(template_e - template_g) * template_g))
# center_e = np.real(np.sum(np.conj(template_e - template_g) * template_e))
# threshold = 0.5 * (center_g + center_e)
threshold = 0.5 * (
    norm(template_e, t_envelope)**2 - norm(template_g, t_envelope)**2)

print("Calculate init")
init_arr_g = get_init_array(para_g, Nruns)
init_arr_e = get_init_array(para_g, Nruns)

state_arr = np.zeros(Nruns, dtype=np.bool)
decision_arr = np.zeros(Nruns, dtype=np.float64)
dist_g_arr = np.zeros(Nruns, dtype=np.float64)
dist_e_arr = np.zeros(Nruns, dtype=np.float64)

t00 = time.time()
for ii in range(Nruns):
    print("{:d} of {:d}".format(ii + 1, Nruns))
    t0 = time.time()

    state = np.random.randint(2)
    if state:
        print("    excited")
        para_s = para_e
        # init_s = init_e
        init_array_s = init_arr_e
        state_arr[ii] = True
    else:
        print("    ground")
        para_s = para_g
        # init_s = init_g
        init_array_s = init_arr_g
        state_arr[ii] = False

    para_s.set_Nbeats(2)
    para_s.set_noise_T(T1=Tph)
    para_s.set_drive_V(drive)
    sol_s = para_s.simulate(init=init_array_s[ii, :])
    (t_envelope, Vout_s_envelope) = get_envelopes(sol_s, para_s, fc)

    signal = Vout_s_envelope
    # decision = np.real(np.sum(np.conj(template_e - template_g) * signal))
    decision = np.real(inprod(signal, template_diff, t_envelope))
    decision_arr[ii] = decision

    dist_g = dist(signal, template_g, t_envelope)
    dist_e = dist(signal, template_e, t_envelope)
    dist_g_arr[ii] = dist_g
    dist_e_arr[ii] = dist_e

    if (state and (decision > threshold)) or (not state and
                                              (decision < threshold)):
        print("    correct: {:.3g}".format(decision))
    else:
        print("*** ERROR: {:.3g} ****************".format(decision))

    t1 = time.time()
    print("    time: {:.2f}s".format(t1 - t0))
t11 = time.time()
print("Total time: {:s}".format(sim.format_sec(t11 - t00)))

np.savez(
    save_filename,
    state_arr=state_arr,
    decision_arr=decision_arr,
    dist_g_arr=dist_g_arr,
    dist_e_arr=dist_e_arr,
    para_g=para_g.pickable_copy(),
    para_e=para_e.pickable_copy(),
    template_g=template_g,
    template_e=template_e,
    threshold=threshold,
    t_envelope=t_envelope,
)
