import numpy as np
from scipy.constants import Boltzmann


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


def get_init_array(para, N, return_noise=False):
    ns = 3 * N
    freqs = np.fft.rfftfreq(ns, para.dt)

    PSD_twosided = []
    PSD_twosided.append(2. * Boltzmann * para.noise_T0 * para.R0)
    PSD_twosided.append(2. * Boltzmann * para.noise_T1 * para.R1)
    if para.NNOISE == 3:
        PSD_twosided.append(2. * Boltzmann * para.noise_T2 * para.R2)

    Vn = np.empty((para.NNOISE, ns), np.float64)
    for ii in range(para.NNOISE):
        Vn[ii] = np.sqrt(PSD_twosided[ii]) * np.sqrt(
            para.fs) * np.random.randn(ns)

    Vn_fft = np.fft.rfft(Vn, axis=-1) / ns

    state_var_fft = np.zeros((para.NEQ, Vn_fft.shape[1]), np.complex128)
    for ii in range(para.NEQ):
        for jj in range(para.NNOISE):
            state_var_fft[ii, :] += para.state_variable_ntf(freqs, ii, jj) * Vn_fft[jj]

    state_var = np.fft.irfft(state_var_fft, axis=-1) * ns

    init_array = np.empty((N, para.NEQ), np.float64)
    for ii in range(para.NEQ):
        init_array[:, ii] = state_var[ii, N:-N]

    if return_noise:
        return init_array, Vn[:, N:-N]
    else:
        return init_array
