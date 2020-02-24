import os

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

prl_single_column_width = 8.6 / 2.54  # inch
fig_width = 2 * prl_single_column_width
fig_height = fig_width * 3 / 4

# Change default font size for nicer plots
# rcParams['figure.dpi'] = 109  # ric-nanophys
rcParams['figure.dpi'] = 167  # ric-xps
rcParams['figure.figsize'] = [fig_width, fig_height]

# rcParams['figure.titlesize'] = 'medium'  # 'large'
# rcParams['axes.labelsize'] = 'small'  # 'medium'
# rcParams['axes.titlesize'] = 'medium'  # 'large'
# rcParams['legend.fontsize'] = 'small'  # 'medium'
# rcParams['xtick.labelsize'] = 'small'  # 'medium'
# rcParams['ytick.labelsize'] = 'small'  # 'medium'

# rcParams['lines.linewidth'] = 0.75  # 1.5
# rcParams['lines.markeredgewidth'] = 0.5  # 1.0
# rcParams['lines.markersize'] = 3.0  # 6.0
# rcParams['legend.columnspacing'] = 1.  # 2.0
# rcParams['legend.labelspacing'] = 0.25  # 0.5
# rcParams['legend.handletextpad'] = 0.4  # 0.8


def single_gaussian(x, m, s):
    return np.exp(-(x - m)**2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def double_gaussian(x, m, s):
    return 0.5 * (single_gaussian(x, m, s) + single_gaussian(x, -m, s))


pulse_list = [
    "short single",
    "single",
    "double",
    "cool",
    "long single",
    "long double",
    "almost",
]

chi_kappa_list = []
infidelity_list = []
leftover_list = []

for ii, pulse in enumerate(pulse_list):
    chi_kappa_list.append([])
    infidelity_list.append([])
    leftover_list.append([])

    load_folder = "results/{:s}".format(pulse)
    for load_filename in os.listdir(load_folder):
        if not load_filename.endswith(".npz"):
            continue
        if not load_filename.startswith("simulate_"):
            continue

        print("\n\n\n")
        print(load_filename)
        load_path = os.path.join(load_folder, load_filename)
        with np.load(load_path) as npzfile:
            amp = np.asscalar(npzfile["amp"])
            phase = np.asscalar(npzfile["phase"])
            wc = np.asscalar(npzfile["wc"])
            wq = np.asscalar(npzfile["wq"])
            chi = np.asscalar(npzfile["chi"])
            kappa = np.asscalar(npzfile["kappa"])
            N = np.asscalar(npzfile["N"])
            wrot1 = np.asscalar(npzfile["wrot1"])
            wrot2 = np.asscalar(npzfile["wrot2"])
            ns = np.asscalar(npzfile["ns"])
            Np = np.asscalar(npzfile["Np"])
            Ntraj = np.asscalar(npzfile["Ntraj"])
            scores_g = npzfile["scores_g"]
            scores_e = npzfile["scores_e"]
            tlist = npzfile["tlist"]
            template_g = npzfile["template_g"]
            template_e = npzfile["template_e"]
            template_diff = npzfile["template_diff"]
            avg_traj_g = npzfile["avg_traj_g"]
            avg_traj_e = npzfile["avg_traj_e"]

        chi_kappa_list[ii].append(chi / kappa)
        assert len(scores_g) == Ntraj
        assert len(scores_e) == Ntraj
        Ntraj_g = np.sum(np.isfinite(scores_g))
        Ntraj_e = np.sum(np.isfinite(scores_e))
        scores_g = scores_g[np.isfinite(scores_g)]
        scores_e = scores_e[np.isfinite(scores_e)]
        if Ntraj_g < Ntraj:
            print("*** WARNING: {:d} NaNs in scores_g!".format(Ntraj -
                                                               Ntraj_g))
        if Ntraj_e < Ntraj:
            print("*** WARNING: {:d} NaNs in scores_e!".format(Ntraj -
                                                               Ntraj_e))

        print()
        print("*** Ground state ***")
        correct_g = np.sum(scores_g < 0)
        fidelity_g = correct_g / Ntraj_g
        # print("{:d} out of {:d}".format(correct_g, Ntraj_g))
        # print("Fidelity: {:.1%}".format(correct_g / Ntraj_g))
        mu_g = np.mean(scores_g)
        sigma_g = np.std(scores_g)
        f_g = 0.5 * (1 + erf((0 - mu_g) / np.sqrt(2 * sigma_g**2)))
        # print("Theoretical: {:.1%}".format(f_g))
        print("Infidelity: {:.2e}".format(1. - f_g))
        ph_g = np.abs(template_g)**2 / kappa
        left_g = ph_g[-1]
        print("Leftover: {:.2e}".format(left_g))

        print()
        print("*** Excited state ***")
        correct_e = np.sum(scores_e > 0)
        fidelity_e = correct_e / Ntraj_e
        # print("{:d} out of {:d}".format(correct_e, Ntraj_e))
        # print("Fidelity: {:.1%}".format(correct_e / Ntraj_e))
        mu_e = np.mean(scores_e)
        sigma_e = np.std(scores_e)
        f_e = 1 - 0.5 * (1 + erf((0 - mu_e) / np.sqrt(2 * sigma_e**2)))
        # print("Theoretical: {:.1%}".format(f_e))
        print("Infidelity: {:.2e}".format(1. - f_e))
        ph_e = np.abs(template_e)**2 / kappa
        left_e = ph_e[-1]
        print("Leftover: {:.2e}".format(left_e))

        infidelity_list[ii].append(0.5 * (2.0 - f_g - f_e))
        leftover_list[ii].append(0.5 * (left_g + left_e))

chi_kappa_arr = np.array(chi_kappa_list)
infidelity_arr = np.array(infidelity_list)
leftover_arr = np.array(leftover_list)
for ii in range(len(pulse_list)):
    idx = np.argsort(chi_kappa_arr[ii])
    chi_kappa_arr[ii, :] = chi_kappa_arr[ii, idx]
    infidelity_arr[ii, :] = infidelity_arr[ii, idx]
    leftover_arr[ii, :] = leftover_arr[ii, idx]

fig1, ax1 = plt.subplots(tight_layout=True)
for ii, pulse in enumerate(pulse_list):
    ax1.loglog(chi_kappa_arr[ii], infidelity_arr[ii], label=pulse)
ax1.legend(loc="best")
ax1.set_xlabel(r"Ratio $\chi / \kappa$")
ax1.set_ylabel(r"Infidelity $1-F$")
fig1.show()

fig2, ax2 = plt.subplots(tight_layout=True)
for ii, pulse in enumerate(pulse_list):
    ax2.loglog(chi_kappa_arr[ii], leftover_arr[ii], label=pulse)
ax2.legend(loc="best")
ax2.set_xlabel(r"Ratio $\chi / \kappa$")
ax2.set_ylabel(r"Leftover photons $<n>$")
fig2.show()
