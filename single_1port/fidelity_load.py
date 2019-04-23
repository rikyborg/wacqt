import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

file_name = 'fidelity_chi_2e6_kappa_2e4_Nruns_65536.npz'

with np.load(file_name) as npz:
    state_arr = npz['state_arr']
    decision_arr = npz['decision_arr']
    para_g = np.asscalar(npz['para_g'])
    para_e = np.asscalar(npz['para_e'])

w_g, Q_g = para_g.calculate_resonance()
w_e, Q_e = para_e.calculate_resonance()
w_c = 0.5 * (w_g + w_e)
chi = 0.5 * (w_g - w_e)
kappa = 0.5 * (w_g / Q_g + w_e / Q_e)

N = len(state_arr)
correct = np.sum(state_arr * (decision_arr < 0)) + np.sum(np.logical_not(state_arr) * (decision_arr > 0))
fidelity = correct / N
print("Fidelity: {:.1%}".format(fidelity))

idx_e = state_arr
idx_g = np.logical_not(idx_e)

mu_g = np.mean(decision_arr[idx_g])
sigma_g = np.std(decision_arr[idx_g])
mu_e = np.mean(decision_arr[idx_e])
sigma_e = np.std(decision_arr[idx_e])

f_g = 1 - 0.5 * (1 + erf((0 - mu_g) / np.sqrt(2 * sigma_g**2)))
f_e = 0.5 * (1 + erf((0 - mu_e) / np.sqrt(2 * sigma_e**2)))
theo_f = 0.5 * (f_g + f_e)
print("Theoretical: {:.1%}".format(theo_f))
print("Infidelity: {:.2e}".format(1. - theo_f))


fig, ax = plt.subplots(tight_layout=True)

n, bins, patches = ax.hist(decision_arr[idx_g], bins=100, color='tab:blue', alpha=0.5, label='|g>')
xx = np.linspace(bins.min(), bins.max(), 200)
N_g = idx_g.sum()
A_g = N_g * (bins[1] - bins[0])
yy = A_g * np.exp(-(xx - mu_g)**2 / sigma_g**2 / 2) / np.sqrt(2. * np.pi * sigma_g**2)
ax.plot(xx, yy, '--', c='tab:blue')

n, bins, patches = ax.hist(decision_arr[idx_e], bins=100, color='tab:orange', alpha=0.5, label='|e>')
xx = np.linspace(bins.min(), bins.max(), 200)
N_e = idx_e.sum()
A_e = N_e * (bins[1] - bins[0])
yy = A_e * np.exp(-(xx - mu_e)**2 / sigma_e**2 / 2) / np.sqrt(2. * np.pi * sigma_e**2)
ax.plot(xx, yy, '--', c='tab:orange')

ax.legend(title="F = {:.1%}".format(fidelity))
ax.set_xlabel(r'Signal [$\mathrm{V}^2$]')
ax.set_ylabel('Counts')
ax.set_title(r"Reflection -- $\chi / \kappa = {:.0f}$".format(chi / kappa))

fig.show()
