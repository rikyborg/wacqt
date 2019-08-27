import numpy as np
from resonator_tools import circuit

import simulator as sim

_f0 = 1e9
_Qb = 1e8
_Ql = 1e4
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

G2 = para.tf2(freqs)

port = circuit.notch_port(freqs, G2)
port.autofit()

print()
print(port.fitresults['fr'] / f0)
print(port.fitresults['Ql'] / Ql)
print(port.fitresults['Qi_dia_corr'] / Qb)
print(port.fitresults['Qc_dia_corr'] / Qc)
