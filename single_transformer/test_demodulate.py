import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate

fc = 100.
fm = 10.
df = 1.
fs = 10_000.

T = 1 / df
dt = 1 / fs
ns = int(round(fs / df))

wc = 2 * np.pi * fc
wm = 2 * np.pi * fm
nc = int(round(fc / df))

t = np.linspace(0, T, ns, endpoint=False)
freqs = np.fft.rfftfreq(ns, dt)

x = np.sin(0.5 * wm * t)**2 * np.cos(wc * t)
x_fft = np.fft.rfft(x) / len(x)

xi = x * np.cos(wc * t)
xq = x * np.sin(wc * t)
xi_fft = np.fft.rfft(xi) / len(xi)
xq_fft = np.fft.rfft(xq) / len(xq)

t_d = t[::100]
freqs_d = np.fft.rfftfreq(ns // 100, 100 * dt)
xi_d = decimate(decimate(xi, 10), 10)
xq_d = decimate(decimate(xq, 10), 10)
xi_d_fft = np.fft.rfft(xi_d) / len(xi_d)
xq_d_fft = np.fft.rfft(xq_d) / len(xq_d)

t_e = t[::100]
freqs_e = np.fft.fftfreq(ns // 100, 100 * dt)
x_e_fft = np.zeros(ns // 100, dtype=np.complex128)
karray = np.arange(nc // 2, 3 * nc // 2)
x_e_fft[karray - nc] = x_fft[karray]
x_e = np.fft.ifft(x_e_fft)
x_e *= len(x_e)


def demodulate(signal, t, nc, bw=None):
    if bw is None:
        bw = nc
    ns = len(signal)
    s_fft = np.fft.rfft(signal) / ns
    e_fft = np.zeros(bw, dtype=np.complex128)
    karray = np.arange(nc - bw // 2, nc + bw // 2)
    e_fft[karray - nc] = s_fft[karray]
    envelope = np.fft.ifft(e_fft) * bw

    t0 = t[0]
    dt = t[1] - t[0]
    t1 = t[-1] + dt
    t_e = np.linspace(t0, t1, bw, endpoint=False)
    return envelope, t_e

fig1, ax1 = plt.subplots(2, 1, tight_layout=True)
ax11, ax12 = ax1

ax11.plot(t, x)
ax11.plot(t_d, 2 * xi_d)
ax11.plot(t_d, 2 * xq_d)
ax11.plot(t_d, 2 * x_e.real)
ax11.plot(t_d, 2 * x_e.imag)

ax12.semilogy(freqs, np.abs(x_fft))
ax12.semilogy(freqs_d, np.abs(xi_d_fft))
ax12.semilogy(freqs_d, np.abs(xq_d_fft))
ax12.semilogy(freqs_e, np.abs(x_e_fft))

fig1.show()
