import numpy as np

chi_kappa = -10.
alpha = -200e6 * 2 * np.pi
wr = 6e9 * 2 * np.pi
kappa = 100e3 * 2 * np.pi
# Delta = -1.5e9 * 2 * np.pi
Ej_Ec = 50.

# chi_kappa = -5.
# alpha = -300e6 * 2 * np.pi
# wr = 6e9 * 2 * np.pi
# kappa = 300e3 * 2 * np.pi
# Delta = -1e9 * 2 * np.pi

chi = chi_kappa * kappa
Ec = -alpha
Ej = Ej_Ec * Ec
wq = np.sqrt(8 * Ej * Ec) - Ec
Delta = wq - wr
g = np.sqrt(Delta * chi * (Delta + alpha) / alpha)
Gamma = kappa * g**2 / Delta**2
# wq = wr + Delta
Ej = (1. / 8.) * (Ec + wq)**2 / Ec
Ej_Ec = Ej / Ec
n_crit = Delta**2 / (4 * g**2)
n_crit_tilde = (Delta + alpha)**2 / (8 * g**2)

print("chi / kappa = {:.3g}".format(chi_kappa))
print("wr = {:.3g}".format(wr / 2 / np.pi))
print("alpha = {:.3g}".format(alpha / 2 / np.pi))
print("kappa = {:.3g}".format(kappa / 2 / np.pi))
print("chi = {:.3g}".format(chi / 2 / np.pi))
print("Delta = {:.3g}".format(Delta / 2 / np.pi))
print("g = {:.3g}".format(g / 2 / np.pi))
print("wq = {:.3g}".format(wq / 2 / np.pi))
print("Gamma = {:.3g}".format(Gamma))
print("Gamma^-1 = {:.3g}".format(1. / Gamma))
print("n_crit = {:.3g}".format(n_crit))
print("n_crit_tilde = {:.3g}".format(n_crit_tilde))
print("Ec = {:.3g}".format(Ec / 2 / np.pi))
print("Ej = {:.3g}".format(Ej / 2 / np.pi))
print("Ej / Ec = {:.3g}".format(Ej / Ec))
