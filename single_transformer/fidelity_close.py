from __future__ import absolute_import, division, print_function

import time

import numpy as np
from scipy.constants import hbar, Boltzmann

import simulator as sim

Nruns = 65536 // 4

# _wc = 2. * np.pi * 6e9
# _kappa = 2. * np.pi * 200e3
# _Qb = 1e7
# _chi = _kappa / 1.
# _Ql = _wc / _kappa
_wc = 2. * np.pi * 6.0296e9
_kappa = 2. * np.pi * 517e3
_Qb = 1e7
_chi = - 2. * np.pi * 143e3
_Ql = _wc / _kappa
AMP = 1.738e-5  # V
detuning = 10.
double = True
nr_freqs = 4

save_filename = "fidelity_close_{:s}_chi_{:.0g}_kappa_{:.0g}_Nruns_{:d}_4.npz".format(
    'double' if double else 'single',
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
kappa = 0.5 * (w_g / Q_g + w_e / Q_e)

Eph = hbar * w_c
Tph = 0.5 * Eph / Boltzmann

_fc = w_c / (2. * np.pi)
_df = 2. * np.abs(detuning * kappa) / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
para_g.set_df(df)
para_e.set_df(df)


def inprod(f, g, t=None):
    if t is not None:
        dt = t[1] - t[0]
        ns = len(t)
        T = ns * dt
    else:
        T = 1.
    return np.trapz(f * np.conj(g), x=t) / T


def norm(x, t=None):
    return np.sqrt(np.real(inprod(x, x, t=t)))


def proj(v, u, t=None):
    if not norm(u, t=t):
        return 0.
    else:
        return inprod(v, u, t=t) / inprod(u, u, t=t) * u


def dist(f, g, t=None):
    return norm(f - g, t=t)


def get_envelopes(sol, para, fc):
    I0 = sol[:, 0]
    P1 = sol[:, 1]
    V1 = sol[:, 2]
    V2 = para.calculate_V2(I0)

    df = para.df
    nc = int(round(fc / df))
    Nbeats = para.Nbeats
    t = para.get_time_arr()
    t_envelope = demodulate_time(t, nc * Nbeats)
    I0_envelope = demodulate(I0, nc * Nbeats)
    P1_envelope = demodulate(P1, nc * Nbeats)
    V1_envelope = demodulate(V1, nc * Nbeats)
    V2_envelope = demodulate(V2, nc * Nbeats)

    return (
        t_envelope,
        I0_envelope,
        P1_envelope,
        V1_envelope,
        V2_envelope,
    )


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

    I0_fft = para.tfn0I0(freqs) * Vn0_fft + para.tfn1I0(
        freqs) * Vn1_fft + para.tfn2I0(freqs) * Vn2_fft
    P1_fft = para.tfn0P1(freqs) * Vn0_fft + para.tfn1P1(
        freqs) * Vn1_fft + para.tfn2P1(freqs) * Vn2_fft
    V1_fft = para.tfn01(freqs) * Vn0_fft + para.tfn11(
        freqs) * Vn1_fft + para.tfn21(freqs) * Vn2_fft

    I0 = np.fft.irfft(I0_fft) * para.ns
    P1 = np.fft.irfft(P1_fft) * para.ns
    V1 = np.fft.irfft(V1_fft) * para.ns

    init_array = np.empty((N, 3), np.float64)
    init_array[:, 0] = I0[N:-N]
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
if double:
    w_m = 2 * np.pi * df / 2
else:
    w_m = 2 * np.pi * df / 4
window[idx] = np.sin(w_m * t_drive[idx])**(nr_freqs - 1)
drive = window * carrier
Vg = drive[:-1]

para_g.set_drive_V(drive)
para_e.set_drive_V(drive)
sol_g = para_g.simulate(print_time=True)
sol_e = para_e.simulate(print_time=True)
(
    t_envelope,
    I0_g_envelope,
    P1_g_envelope,
    V1_g_envelope,
    V2_g_envelope,
) = get_envelopes(sol_g, para_g, fc)
(
    t_envelope,
    I0_e_envelope,
    P1_e_envelope,
    V1_e_envelope,
    V2_e_envelope,
) = get_envelopes(sol_e, para_e, fc)
template_g = V2_g_envelope.copy()
template_e = V2_e_envelope.copy()
template_diff = template_e - template_g
# center_g = np.real(np.sum(np.conj(template_e - template_g) * template_g))
# center_e = np.real(np.sum(np.conj(template_e - template_g) * template_e))
# threshold = 0.5 * (center_g + center_e)
threshold = 0.5 * (norm(template_e, t_envelope)**2 - norm(template_g, t_envelope)**2)

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
    (
        t_envelope,
        I0_s_envelope,
        P1_s_envelope,
        V1_s_envelope,
        V2_s_envelope,
    ) = get_envelopes(sol_s, para_s, fc)

    signal = V2_s_envelope
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
