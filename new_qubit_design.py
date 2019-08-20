import numpy as np

chi_kappa = -5.
alpha = -2. * np.pi * 250e6
Ej_Ec = 50.
Gamma = 10e3
ncrit = None
wr = 2. * np.pi * 6e9

Ec = -alpha
Ej = Ej_Ec * Ec
wq = np.sqrt(8 * Ej * Ec) - Ec

if wr is None:
    kappa = 4 * Gamma * ncrit
    g = -8 * alpha * chi_kappa * Gamma * np.sqrt(
        ncrit**3) / (alpha - 16 * chi_kappa * Gamma * ncrit**2)
    Delta = 16 * alpha * chi_kappa * Gamma * ncrit**2 / (
        alpha - 16 * chi_kappa * Gamma * ncrit**2)

    wr = wq - Delta
    chi = chi_kappa * kappa
elif ncrit is None:
    Delta = wq - wr

    kappa = Gamma * np.sqrt(Delta * alpha / (Gamma * chi_kappa *
                                             (Delta + alpha)))
    g = np.sqrt(Delta * Gamma * chi_kappa * np.sqrt(
        Delta * alpha / (Gamma * chi_kappa * (Delta + alpha))) *
                (Delta + alpha) / alpha)
    ncrit = (1 / 4) * np.sqrt(Delta * alpha / (Gamma * chi_kappa *
                                               (Delta + alpha)))
    chi = chi_kappa * kappa
elif Gamma is None:
    Delta = wq - wr
    kappa = (1 / 4) * Delta * alpha / (chi_kappa * ncrit * (Delta + alpha))
    g = -1 / 2 * Delta * np.sqrt(1 / ncrit)
    Gamma = (1 / 16) * Delta * alpha / (chi_kappa * ncrit**2 * (Delta + alpha))
    chi = chi_kappa * kappa

ncrit_tilde = (Delta + alpha)**2 / (8 * g**2)

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
print("ncrit = {:.3g}".format(ncrit))
print("ncrit_tilde = {:.3g}".format(ncrit_tilde))
print("Ec = {:.3g}".format(Ec / 2 / np.pi))
print("Ej = {:.3g}".format(Ej / 2 / np.pi))
print("Ej / Ec = {:.3g}".format(Ej / Ec))
