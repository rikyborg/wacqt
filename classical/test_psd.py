from __future__ import division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, Planck
from scipy.signal import periodogram

from utils import get_init_array

from simulators import sim_transformer as sim
# from simulators import sim_notch as sim
# from simulators import sim_reflection as sim
# from simulators import sim_transmission as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

Navg = 100
FAST = False

kwargs = {'w0': 2. * np.pi * 1e9, 'Qb': 1e4, 'Ql': 1e2, 'R0': 50., 'fs': 20e9}
if sim.NNOISE == 3:
    kwargs['R2'] = 25.
res, para = sim.SimulationParameters.from_measurement_single(**kwargs)
assert res.success

w0, Q = para.calculate_resonance()
f0 = w0 / (2. * np.pi)

Tph = Planck * f0 / (2. * Boltzmann)
noise_kwargs = {'T0': Tph, 'T1': Tph}
if sim.NNOISE == 3:
    noise_kwargs['T2'] = Tph

df_ = f0 / Q / 10.  # Hz
_, df = para.tune(0., df_, regular=True)
para.set_df(df)
para.set_drive_none()

if FAST:
    para.set_noise_T(**noise_kwargs)
else:
    # Run once to get initial condition
    para.set_Nbeats(1)
    para.set_noise_T(**noise_kwargs)
    sol = para.simulate()

para.set_Nbeats(1)
freqs = np.fft.rfftfreq(para.ns, para.dt)
psd = np.zeros((sim.NEQ, len(freqs)),
               np.float128)  # accumulate with higher precision
psd_out = np.zeros(len(freqs), np.float128)  # accumulate with higher precision

t0 = time.time()
for ii in range(Navg):
    print(ii)
    if FAST:
        sol, noise = get_init_array(para, para.ns, return_noise=True)
        para.noise0_array[:para.ns] = noise[0, :]
        para.noise1_array[:para.ns] = noise[1, :]
        if para.NNOISE == 3:
            para.noise2_array[:para.ns] = noise[2, :]
    else:
        # regenerate noise
        para.set_noise_T(**noise_kwargs)
        sol = para.simulate(continue_run=True)
    Vout = para.calculate_Vout(sol)
    for jj in range(sim.NEQ):
        data = sol[-para.ns:, jj]
        mf, X = periodogram(
            data,
            para.fs,
            window='boxcar',
            nfft=None,
            detrend=False,
            return_onesided=True,
            scaling='density')
        psd[jj, :] += X
    mf, X = periodogram(
        Vout,
        para.fs,
        window='boxcar',
        nfft=None,
        detrend=False,
        return_onesided=True,
        scaling='density')
    psd_out[:] += X
t1 = time.time()
print("Simulation time: {}".format(sim.format_sec(t1 - t0)))

psd /= Navg
psd = np.float64(psd)  # cast back to double precision
psd_out /= Navg
psd_out = np.float64(psd_out)  # cast back to double precision

sigmas = []
sigmas.append(4. * Boltzmann * para.noise_T0 * para.R0)
sigmas.append(4. * Boltzmann * para.noise_T1 * para.R1)
if sim.NNOISE == 3:
    sigmas.append(4. * Boltzmann * para.noise_T2 * para.R2)
theo = np.zeros((sim.NEQ, len(freqs)))
theo_out = np.zeros(len(freqs))
for ii in range(sim.NEQ):
    for jj in range(sim.NNOISE):
        theo[ii, :] += sigmas[jj] * np.abs(
            para.state_variable_ntf(freqs, ii, jj))**2
for jj in range(sim.NNOISE):
    theo_out[:] += sigmas[jj] * np.abs(para.output_ntf(freqs, jj))**2

fig, ax = plt.subplots(tight_layout=True)
for jj in range(sim.NEQ):
    ax.loglog(freqs, psd[jj, :], label=para.state_variables_latex[jj])
    ax.loglog(freqs, theo[jj, :], '--', label=para.state_variables_latex[jj])
ax.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax.legend()
fig.show()

fig2, ax2 = plt.subplots(tight_layout=True)
ax2.loglog(freqs, psd_out, label=r"$V_\mathrm{OUT}$")
ax2.loglog(freqs, theo_out, '--', label=r"$V_\mathrm{OUT}$")
ax2.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax2.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax2.legend()
fig2.show()
