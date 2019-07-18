from __future__ import absolute_import, division, print_function

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, Boltzmann

import simulator as sim


Nruns = 1024


_wc = 2. * np.pi * 6e9
_chi = 2. * np.pi * 2e6
_Qb = 10e6
_kappa = _chi / 10
_Ql = _wc / _kappa
AMP = 2.263e-6  # V
save_filename = "fidelity_chi_2e6_kappa_2e5_Nruns_1024.npz"

res, para_g, para_e = sim.SimulationParameters.from_measurement(_wc, _chi, _Qb, _Ql)

w_g, Q_g = para_g.calculate_resonance()
w_e, Q_e = para_e.calculate_resonance()
chi = 0.5 * (w_e - w_g)
w_c = 0.5 * (w_g + w_e)
kappa = 0.5 * (w_g / Q_g + w_e / Q_e)

Eph = hbar * w_c
Tph = 0.5 * Eph / Boltzmann
T0 = Tph

_fc = w_c / (2. * np.pi)
_df = 2. * np.abs(chi) / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
para_g.set_df(df)
para_e.set_df(df)


def get_envelopes(sol, para, Vg):
    P0 = sol[:, 0]
    P1 = sol[:, 1]
    V1 = sol[:, 2]

    Nimps = 31
    Npoints = 1000
    nc = int(round(fc / df))
    karray = np.arange((nc - Nimps // 2) * para.Nbeats, (nc + Nimps // 2) * para.Nbeats + 1)
    t_envelope = np.linspace(0., para.Nbeats / para.df, Npoints, endpoint=False)

    P0_spectrum = np.fft.rfft(P0) / len(P0)
    P1_spectrum = np.fft.rfft(P1) / len(P1)
    V1_spectrum = np.fft.rfft(V1) / len(V1)

    P0_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
    P1_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
    V1_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)

    P0_envelope_spectrum[karray - para.Nbeats * nc] = P0_spectrum[karray]
    P1_envelope_spectrum[karray - para.Nbeats * nc] = P1_spectrum[karray]
    V1_envelope_spectrum[karray - para.Nbeats * nc] = V1_spectrum[karray]

    P0_envelope = np.fft.ifft(P0_envelope_spectrum) * Npoints
    P1_envelope = np.fft.ifft(P1_envelope_spectrum) * Npoints
    V1_envelope = np.fft.ifft(V1_envelope_spectrum) * Npoints

    return (
        t_envelope,
        P0_envelope,
        P1_envelope,
        V1_envelope,
    )


def get_init_array(para, N):
    ns = 3 * N
    freqs = np.fft.rfftfreq(ns, para.dt)

    PSDv0_twosided = 2. * Boltzmann * para.noise_T0 * para.R0
    PSDv1_twosided = 2. * Boltzmann * para.noise_T1 * para.R1
    PSDv2_twosided = 2. * Boltzmann * para.noise_T2 * para.R2

    Vn0 = np.sqrt(PSDv0_twosided) * np.sqrt(para.fs) * np.random.randn(ns)
    Vn1 = np.sqrt(PSDv1_twosided) * np.sqrt(para.fs) * np.random.randn(ns)
    Vn2 = np.sqrt(PSDv2_twosided) * np.sqrt(para.fs) * np.random.randn(ns)

    Vn0_fft = np.fft.rfft(Vn0) / ns
    Vn1_fft = np.fft.rfft(Vn1) / ns
    Vn2_fft = np.fft.rfft(Vn2) / ns

    P0_fft = para.tfn0P0(freqs) * Vn0_fft + para.tfn1P0(freqs) * Vn1_fft + para.tfn2P0(freqs) * Vn2_fft
    P1_fft = para.tfn0P1(freqs) * Vn0_fft + para.tfn1P1(freqs) * Vn1_fft + para.tfn2P1(freqs) * Vn2_fft
    V1_fft = para.tfn01(freqs) * Vn0_fft + para.tfn11(freqs) * Vn1_fft + para.tfn21(freqs) * Vn2_fft

    P0 = np.fft.irfft(P0_fft) * para.ns
    P1 = np.fft.irfft(P1_fft) * para.ns
    V1 = np.fft.irfft(V1_fft) * para.ns

    init_array = np.empty((N, 3), np.float64)
    init_array[:, 0] = P0[N:-N]
    init_array[:, 1] = P1[N:-N]
    init_array[:, 2] = V1[N:-N]

    return init_array


# templates
print("Finding templates")
para_g.set_Nbeats(2)
para_e.set_Nbeats(2)

t = para_g.get_time_arr()
t_drive = para_g.get_drive_time_arr()
carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
window = np.zeros_like(t_drive)
idx = np.s_[0 * para_g.ns:2 * para_g.ns]
window[idx] = np.sin(2. * np.pi * 0.5 * df * t_drive[idx])**2
drive = window * carrier
Vg = drive[:-1]

para_g.set_drive_V(drive)
para_e.set_drive_V(drive)
sol_g = para_g.simulate(print_time=True)
sol_e = para_e.simulate(print_time=True)
(
    t_envelope,
    P0_g_envelope,
    P1_g_envelope,
    V1_g_envelope,
) = get_envelopes(sol_g, para_g, Vg)
(
    t_envelope,
    P0_e_envelope,
    P1_e_envelope,
    V1_e_envelope,
) = get_envelopes(sol_e, para_e, Vg)
template_g = P0_g_envelope.copy()
template_e = P0_e_envelope.copy()


if False:  # use faster method below
    # reach equilibrium and save init
    print("Finding equilibrium")
    Nrelax = int(round(2. * np.pi * df / kappa))
    para_g.set_Nbeats(3 * Nrelax)
    para_e.set_Nbeats(3 * Nrelax)
    para_g.set_noise_T(T1=Tph, T0=T0)
    para_e.set_noise_T(T1=Tph, T0=T0)
    para_g.set_drive_none()
    para_e.set_drive_none()
    para_g.simulate(print_time=True)
    para_e.simulate(print_time=True)
    init_g = para_g.next_init.copy()
    init_e = para_e.next_init.copy()
else:
    print("Calculate init")
    init_arr_g = get_init_array(para_g, Nruns)
    init_arr_e = get_init_array(para_g, Nruns)

state_arr = np.zeros(Nruns, dtype=np.bool)
decision_arr = np.zeros(Nruns, dtype=np.float64)

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

    if False:  # use faster method below
        # readout run
        print('    equilibrate')
        para_s.set_Nbeats(Nrelax)
        para_s.set_noise_T(T1=Tph, T0=T0)
        para_s.set_drive_none()
        para_s.simulate(init=init_s)
        print('    readout')
        para_s.set_Nbeats(2)
        para_s.set_noise_T(T1=Tph, T0=T0)
        para_s.set_drive_V(drive)
        sol_s = para_s.simulate(continue_run=True)
    else:
        para_s.set_Nbeats(2)
        para_s.set_noise_T(T1=Tph, T0=T0)
        para_s.set_drive_V(drive)
        sol_s = para_s.simulate(init=init_array_s[ii, :])
    (
        t_envelope,
        P0_s_envelope,
        P1_s_envelope,
        V1_s_envelope,
    ) = get_envelopes(sol_s, para_s, Vg)

    signal = P0_s_envelope
    decision = np.real(np.sum(np.conj(template_g - template_e) * signal))
    decision_arr[ii] = decision

    if (state and (decision < 0)) or (not state and (decision > 0)):
        print("    correct: {:.3g}".format(decision))
    else:
        print("*** ERROR: {:.3g} ****************".format(decision))

    t1 = time.time()
    print("    time: {:.2f}s".format(t1 - t0))
t11 = time.time()
print("Total time: {:s}".format(sim.format_sec(t11 - t00)))

# fig, ax = plt.subplots()
# ax.hist(decision_arr)
# fig.show()

np.savez(
    save_filename,
    state_arr=state_arr,
    decision_arr=decision_arr,
    para_g=para_g.pickable_copy(),
    para_e=para_e.pickable_copy(),
)
