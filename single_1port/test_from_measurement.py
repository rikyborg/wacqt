from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import simulator as sim

_wc = 2. * np.pi * 6e9
_chi = 2. * np.pi * 2e6
_Qb = 1e6
# _kappa = 2. * np.pi * 37.5e6 / 1e2
_kappa = _chi / 10
_Ql = _wc / _kappa

res, para_g, para_e = sim.SimulationParameters.from_measurement(_wc, _chi, _Qb, _Ql)

freqs = np.linspace(
    (_wc - 2 * _chi) / (2. * np.pi),
    (_wc + 2 * _chi) / (2. * np.pi),
    100000,
)
r_g = para_g.tf1(freqs)
r_e = para_e.tf1(freqs)


def lorentzian(f, f0, Q, A, P):
    return A * np.exp(1j * P) / (1. - f**2 / f0**2 + 1j * f / f0 / Q)


def erf(p, f, r):
    f0, Q, A, P = p
    e = lorentzian(f, f0, Q, A, P) - r
    return np.concatenate((e.real, e.imag))


res_g = least_squares(
    erf, [para_g.f01_d, para_g.Q1_d, 1e-3, np.pi],
    method='lm',
    args=(freqs, r_g),
)
res_e = least_squares(
    erf, [para_e.f01_d, para_e.Q1_d, 1e-3, np.pi],
    method='lm',
    args=(freqs, r_e),
)

fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax11, ax12 = ax1

ax11.axvline((_wc - _chi) / (2. * np.pi), ls='--', c='tab:gray')
ax11.axvline((_wc + _chi) / (2. * np.pi), ls='--', c='tab:gray')
ax11.semilogy(freqs, np.abs(r_g), c='tab:blue', lw=3)
ax11.semilogy(freqs, np.abs(r_e), c='tab:orange', lw=3)
ax11.semilogy(freqs, np.abs(lorentzian(freqs, *res_g.x)), c='tab:red', ls='--')
ax11.semilogy(freqs, np.abs(lorentzian(freqs, *res_e.x)), c='tab:green', ls='--')

ax12.axvline((_wc - _chi) / (2. * np.pi), ls='--', c='tab:gray')
ax12.axvline((_wc + _chi) / (2. * np.pi), ls='--', c='tab:gray')
ax12.plot(freqs, np.angle(r_g), c='tab:blue', lw=3)
ax12.plot(freqs, np.angle(r_e), c='tab:orange', lw=3)
ax12.plot(freqs, np.angle(lorentzian(freqs, *res_g.x)), c='tab:red', ls='--')
ax12.plot(freqs, np.angle(lorentzian(freqs, *res_e.x)), c='tab:green', ls='--')

ax11.set_title("Oscillator")

fig1.show()


fig2, ax2 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax21, ax22 = ax2

ax21.axvline((_wc - _chi) / (2. * np.pi), ls='--', c='tab:gray')
ax21.axvline((_wc + _chi) / (2. * np.pi), ls='--', c='tab:gray')
ax21.semilogy(freqs, np.abs(para_g.tfr(freqs)), c='tab:blue')
ax21.semilogy(freqs, np.abs(para_e.tfr(freqs)), c='tab:orange')

ax22.axvline((_wc - _chi) / (2. * np.pi), ls='--', c='tab:gray')
ax22.axvline((_wc + _chi) / (2. * np.pi), ls='--', c='tab:gray')
ax22.plot(freqs, np.angle(para_g.tfr(freqs)), c='tab:blue')
ax22.plot(freqs, np.angle(para_e.tfr(freqs)), c='tab:orange')

ax21.set_title("Reflection")

fig2.show()


w_g = 2. * np.pi * res_g.x[0]
w_e = 2. * np.pi * res_e.x[0]
w_c = 0.5 * (w_e + w_g)
chi = 0.5 * (w_e - w_g)
Ql_g = res_g.x[1]
Ql_e = res_e.x[1]
kappa_g = w_g / Ql_g
kappa_e = w_e / Ql_e

kappa = 0.5 * (kappa_e + kappa_g)  # average kappa
Ql = w_c / kappa  # OBS: different than average Ql!

print("w_c / 2pi: {:.3f} GHz, rel. err.: {:.1g}".format(1e-9 * w_c / (2. * np.pi), (w_c - _wc) / _wc))
print("chi / 2pi: {:.3f} MHz, rel. err.: {:.1g}".format(1e-6 * chi / (2. * np.pi), (np.abs(chi) - _chi) / _chi))
print("kappa_g / 2pi: {:.3f} kHz, rel. err.: {:.1g}".format(1e-3 * kappa_g / (2. * np.pi), (kappa_g - _kappa) / _kappa))
print("kappa_e / 2pi: {:.3f} kHz, rel. err.: {:.1g}".format(1e-3 * kappa_e / (2. * np.pi), (kappa_e - _kappa) / _kappa))
