import matplotlib.pyplot as plt
import numpy as np

from resonator_tools import circuit

import simulator as sim

with np.load('/home/riccardo/Downloads/vna_data.npz') as f:
    freqs = f['freqs']
    resp_raw = f['resp_raw']

# First use resonator_tools
port = circuit.notch_port(freqs, resp_raw)
port.autofit()
resp_norm = port.z_data
resp_sim = port.z_data_sim
resp_sim_norm = port.z_data_sim_norm
w0_sim = 2. * np.pi * port.fitresults['fr']
Ql_sim = port.fitresults['Ql']
Qb_sim = port.fitresults['Qi_dia_corr']

# Get starting attempt
res_0, para_0 = sim.SimulationParameters.from_measurement_single(
    w0_sim, Qb_sim, Ql_sim)
assert res_0.success
# Fit to data
res, para = sim.SimulationParameters.from_data(freqs, resp_sim_norm, para_0)
assert res
A, phi = res.x[-2:]
G2_fit_norm = A * np.exp(1j * phi) * para.tf2(freqs)

# Undo normalization
_delay, _amp_norm, _alpha, _fr, _Ql, _A2, _frcal = port.do_calibration(
    freqs, resp_raw, ignoreslope=True, guessdelay=True, fixed_delay=None)
G2_fit = G2_fit_norm * _amp_norm / np.exp(
    1j * (-_alpha + 2. * np.pi * _delay * freqs)) + _A2 * (freqs - _frcal)

# Plot
fig1 = plt.figure(tight_layout=True)
ax11 = fig1.add_subplot(1, 2, 1)
ax12 = fig1.add_subplot(2, 2, 2)
ax13 = fig1.add_subplot(2, 2, 4, sharex=ax12)

ax11.plot(resp_norm.real, resp_norm.imag)
ax11.plot(resp_sim_norm.real, resp_sim_norm.imag)
ax11.plot(G2_fit_norm.real, G2_fit_norm.imag)

ax12.semilogy(freqs, np.abs(resp_norm))
ax12.semilogy(freqs, np.abs(resp_sim_norm))
ax12.semilogy(freqs, np.abs(G2_fit_norm))

ax13.plot(freqs, np.angle(resp_norm))
ax13.plot(freqs, np.angle(resp_sim_norm))
ax13.plot(freqs, np.angle(G2_fit_norm))

for _label in ax12.get_xticklabels():
    _label.set_visible(False)
ax12.xaxis.offsetText.set_visible(False)

fig1.show()

fig2 = plt.figure(tight_layout=True)
ax21 = fig2.add_subplot(1, 2, 1)
ax22 = fig2.add_subplot(2, 2, 2)
ax23 = fig2.add_subplot(2, 2, 4, sharex=ax22)

ax21.plot(resp_raw.real, resp_raw.imag)
ax21.plot(resp_sim.real, resp_sim.imag)
ax21.plot(G2_fit.real, G2_fit.imag)

ax22.semilogy(freqs, np.abs(resp_raw))
ax22.semilogy(freqs, np.abs(resp_sim))
ax22.semilogy(freqs, np.abs(G2_fit))

ax23.plot(freqs, np.angle(resp_raw))
ax23.plot(freqs, np.angle(resp_sim))
ax23.plot(freqs, np.angle(G2_fit))

for _label in ax22.get_xticklabels():
    _label.set_visible(False)
ax22.xaxis.offsetText.set_visible(False)

fig2.show()
