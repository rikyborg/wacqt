import matplotlib.cm as mplcm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

with np.load("fidelity_g_e_p_8192.npz") as npzfile:
    amp = npzfile["amp"]
    phase = npzfile["phase"]
    wc = npzfile["wc"]
    wq = npzfile["wq"]
    chi = npzfile["chi"]
    kappa = npzfile["kappa"]
    N = npzfile["N"]
    wrot1 = npzfile["wrot1"]
    wrot2 = npzfile["wrot2"]
    ns = npzfile["ns"]
    Np = npzfile["Np"]
    Ntraj = npzfile["Ntraj"]
    scores_g = npzfile["scores_g"]
    scores_e = npzfile["scores_e"]
    scores_p = npzfile["scores_p"]
    ssz = npzfile["ssz"]
    tlist = npzfile["tlist"]


print()
print("*** Ground state ***")
correct_g = np.sum(scores_g < 0)
fidelity_g = correct_g / Ntraj
print("{:d} out of {:d}".format(correct_g, Ntraj))
print("Fidelity: {:.1%}".format(correct_g / Ntraj))
mu_g = np.mean(scores_g)
sigma_g = np.std(scores_g)
f_g = 0.5 * (1 + erf((0 - mu_g) / np.sqrt(2 * sigma_g**2)))
print("Theoretical: {:.1%}".format(f_g))
print("Infidelity: {:.2e}".format(1. - f_g))


print()
print("*** Excited state ***")
correct_e = np.sum(scores_e > 0)
fidelity_e = correct_e / Ntraj
print("{:d} out of {:d}".format(correct_e, Ntraj))
print("Fidelity: {:.1%}".format(correct_e / Ntraj))
mu_e = np.mean(scores_e)
sigma_e = np.std(scores_e)
f_e = 1 - 0.5 * (1 + erf((0 - mu_e) / np.sqrt(2 * sigma_e**2)))
print("Theoretical: {:.1%}".format(f_e))
print("Infidelity: {:.2e}".format(1. - f_e))

fig1, ax1 = plt.subplots(tight_layout=True)

n, bins, patches = ax1.hist(scores_g, bins=100, color='tab:blue', alpha=0.5, label='|g>')
xx = np.linspace(bins.min(), bins.max(), 200)
N_g = Ntraj
A_g = N_g * (bins[1] - bins[0])
yy = A_g * np.exp(-(xx - mu_g)**2 / sigma_g**2 / 2) / np.sqrt(2. * np.pi * sigma_g**2)
ax1.plot(xx, yy, '--', c='tab:blue')

n, bins, patches = ax1.hist(scores_e, bins=100, color='tab:orange', alpha=0.5, label='|e>')
xx = np.linspace(bins.min(), bins.max(), 200)
N_e = Ntraj
A_e = N_e * (bins[1] - bins[0])
yy = A_e * np.exp(-(xx - mu_e)**2 / sigma_e**2 / 2) / np.sqrt(2. * np.pi * sigma_e**2)
ax1.plot(xx, yy, '--', c='tab:orange')

ax1.legend(title=r"|$\psi_0$> =")
ax1.set_xlabel(r'Signal [$\mathrm{V}^2$]')
ax1.set_ylabel('Counts')
ax1.set_title(r"Basis states -- $\chi / \kappa = {:.0f}$".format(chi / kappa))

fig1.show()

print()
print("*** Plus state ***")
idx_pg = ssz[:, -1] < 0
idx_pe = ssz[:, -1] > 0
correct_p = np.sum(scores_p[idx_pe] > 0) + np.sum(scores_p[idx_pg] < 0)
fidelity_p = correct_p / Ntraj
print("{:d} out of {:d}".format(correct_p, Ntraj))
print("Fidelity: {:.1%}".format(correct_p / Ntraj))
mu_pg = np.mean(scores_p[idx_pg])
sigma_pg = np.std(scores_p[idx_pg])
f_pg = 0.5 * (1 + erf((0 - mu_pg) / np.sqrt(2 * sigma_pg**2)))
mu_pe = np.mean(scores_p[idx_pe])
sigma_pe = np.std(scores_p[idx_pe])
f_pe = 1 - 0.5 * (1 + erf((0 - mu_pe) / np.sqrt(2 * sigma_pe**2)))
f_p = 0.5 * (f_pg + f_pe)
print("Theoretical: {:.1%}".format(f_p))
print("Infidelity: {:.2e}".format(1. - f_p))


if False:
    fig2, ax2 = plt.subplots(tight_layout=True)
    n, bins, patches = ax2.hist(scores_p[idx_pg], bins=100, color='tab:blue', alpha=0.5, label='|g>')
    xx = np.linspace(bins.min(), bins.max(), 200)
    N_pg = idx_pg.sum()
    A_pg = N_pg * (bins[1] - bins[0])
    yy = A_pg * np.exp(-(xx - mu_pg)**2 / sigma_pg**2 / 2) / np.sqrt(2. * np.pi * sigma_pg**2)
    ax2.plot(xx, yy, '--', c='tab:blue')

    n, bins, patches = ax2.hist(scores_p[idx_pe], bins=100, color='tab:orange', alpha=0.5, label='|e>')
    xx = np.linspace(bins.min(), bins.max(), 200)
    N_pe = idx_pe.sum()
    A_pe = N_pe * (bins[1] - bins[0])
    yy = A_pe * np.exp(-(xx - mu_pe)**2 / sigma_pe**2 / 2) / np.sqrt(2. * np.pi * sigma_pe**2)
    ax2.plot(xx, yy, '--', c='tab:orange')

    # ax2.legend(title="F = {:.1%}".format(fidelity))
    ax2.set_xlabel(r'Signal [$\mathrm{V}^2$]')
    ax2.set_ylabel('Counts')
    ax2.set_title(r"Superposition -- $\chi / \kappa = {:.0f}$".format(chi / kappa))
    fig2.show()
else:
    fig2, ax2 = plt.subplots(tight_layout=True)
    Nbins = 128
    counts = np.zeros(Nbins, dtype=np.int64)
    thomo = np.zeros(Nbins)
    low, high = scores_p.min(), scores_p.max()
    step = (high - low) / Nbins
    for ii in range(Ntraj):
        kk = int(np.floor((scores_p[ii] - low) / step))
        kk = min(kk, Nbins - 1)
        counts[kk] += 1
        thomo[kk] += ssz[ii, -1]
    thomo /= counts

    ax2.axhline(0., ls='--', c='tab:gray')
    ax2.axvline(0., ls='--', c='tab:gray')
    ax2.plot(low + np.arange(Nbins) * step, thomo)

    ax2.set_xlabel(r'Signal [$\mathrm{V}^2$]')
    ax2.set_ylabel(r'$\left< \sigma_\mathrm{Z} \right>$')
    ax2.set_title(r"Superposition -- $\chi / \kappa = {:.0f}$".format(chi / kappa))
    fig2.show()

    fig2b, ax2b = plt.subplots(tight_layout=True)

    # n, bins, patches = ax2b.hist(scores_p, bins=Nbins)
    n, bins, patches = ax2b.hist(scores_p, bins=low + np.arange(Nbins + 1) * step)

    cmap = mplcm.get_cmap('cividis')
    for ii, patch in enumerate(patches):
        patch.set_color(cmap((thomo[ii] + 1) / 2))

    fig2b.show()

xx = np.tile(tlist, Ntraj)
yy = ssz.flatten()
fig3, ax3 = plt.subplots(tight_layout=True)
h, xedges, yedges, im3 = ax3.hist2d(xx, yy, bins=len(tlist), normed=True, norm=LogNorm())
ax3.set_ylim(-1.1, 1.1)
fig3.colorbar(im3)
fig3.show()
