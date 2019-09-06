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
resp_norm = port.z_data_sim_norm
w0_sim = 2. * np.pi * port.fitresults['fr']
Ql_sim = port.fitresults['Ql']
Qb_sim = port.fitresults['Qi_dia_corr']

# Get starting attempt
res_0, para_0 = sim.SimulationParameters.from_measurement_single(w0_sim, Qb_sim, Ql_sim)
assert res_0.success
# Fit to data
res, para = sim.SimulationParameters.from_data(freqs, resp_norm, para_0)
assert res
A, phi = res.x[-2:]
G2 = A * np.exp(1j * phi) * para.tf2(freqs)

# Plot
fig = plt.figure(tight_layout=True)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)

ax1.plot(resp_norm.real, resp_norm.imag)
ax1.plot(G2.real, G2.imag)

ax2.semilogy(freqs, np.abs(resp_norm))
ax2.semilogy(freqs, np.abs(G2))

ax3.plot(freqs, np.angle(resp_norm))
ax3.plot(freqs, np.angle(G2))

for _label in ax2.get_xticklabels():
    _label.set_visible(False)
ax2.xaxis.offsetText.set_visible(False)

fig.show()
