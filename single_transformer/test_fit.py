import matplotlib.pyplot as plt
import numpy as np
from resonator_tools import circuit

import simulator as sim

_f0 = 6e9
_Qb = 600e3
_Ql = 60e3
_w0 = 2. * np.pi * _f0
_Qc = 1 / (1 / _Ql - 1 / _Qb)

res, para = sim.SimulationParameters.from_measurement_single(_w0, _Qb, _Ql)
assert res.success

w0, Ql = para.calculate_resonance()
f0 = w0 / (2. * np.pi)
Qb = para.R1 * np.sqrt(para.C1 / para.L1)
Qc = 1 / (1 / Ql - 1 / Qb)

print(w0 / _w0)
print(Ql / _Ql)
print(Qb / _Qb)
print(Qc / _Qc)

omegas = np.linspace(w0 * (1 - 5 / Ql), w0 * (1 + 5 / Ql), 1001)
freqs = omegas / (2. * np.pi)

G1 = para.tf1(freqs)
G2 = para.tf2(freqs)

port = circuit.notch_port(freqs, G2)
port.autofit()

G2_fit = port.z_data_sim
G2_norm = port.z_data_sim_norm

print()
print(port.fitresults['fr'] / f0)
print(port.fitresults['Ql'] / Ql)
print(port.fitresults['Qi_dia_corr'] / Qb)
print(port.fitresults['Qc_dia_corr'] / Qc)

fig = plt.figure(tight_layout=True)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)

for _ax in [ax2, ax3]:
    _ax.axvline(w0 / 2 / np.pi, ls='--', c='tab:gray')
    _ax.axvline(w0 * (1 - 0.5 / Ql) / 2 / np.pi, ls='--', c='tab:gray')
    _ax.axvline(w0 * (1 + 0.5 / Ql) / 2 / np.pi, ls='--', c='tab:gray')

ax1.plot(G2.real, G2.imag)
ax1.plot(G2_fit.real, G2_fit.imag, '--')
# ax1.plot(G2_norm.real, G2_norm.imag, '--')
# ax1.plot(G1.real, G1.imag)

ax2.semilogy(freqs, np.abs(G2))
ax2.semilogy(freqs, np.abs(G2_fit), '--')
ax2.semilogy(freqs, np.abs(G2_norm), '--')
ax2.semilogy(freqs, np.abs(G1))

ax3.plot(freqs, np.angle(G2))
ax3.plot(freqs, np.angle(G2_fit), '--')
ax3.plot(freqs, np.angle(G2_norm), '--')
ax3.plot(freqs, np.angle(G1))

for _label in ax2.get_xticklabels():
    _label.set_visible(False)
ax2.xaxis.offsetText.set_visible(False)

fig.show()
