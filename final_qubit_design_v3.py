from collections import OrderedDict
import sys

import numpy as np
import sympy as sp

sp.init_printing()

chi, alpha, g2, Delta, ncrit, Gamma, Ec, Ej, wq, wr, kappa, JCR, XKR = sp.symbols('chi, alpha, g2, Delta, ncrit, Gamma, Ec, Ej, wq, wr, kappa, JCR, XKR')

para = OrderedDict(
    chi=-2. * np.pi * 10e6,
    alpha=- 2. * np.pi * 300e6,
    g2=(2 * np.pi * 100e6)**2,
    Delta=None,
    ncrit=None,
    Gamma=None,
    # Ec=None,
    # Ej=None,
    # wq=None,
    wr=2. * np.pi * 5.70e9,
    kappa=2. * np.pi * 100e3,
    JCR=None,
    # XKR=-2.,
)

para['g2'] *= (para['wr'] / (2. * np.pi * 6e9))**2

system = [
    alpha * g2 / (Delta * (Delta + alpha)) - chi,
    Delta**2 / (4 * g2) - ncrit,
    kappa * g2 / Delta**2 - Gamma,
    # Ec + alpha,
    sp.sqrt(8 * (JCR * (-alpha)) * (-alpha)) - (-alpha) - (wr + Delta),
    # wr + Delta - wq,
    # Ej / (-alpha) - JCR,
    # chi / kappa - XKR,
]


n_para = len(para)
n_eqs = len(system)
n_var = n_para - n_eqs

given = sum([1 for k in para if para[k] is not None])
if given > n_var:
    print('Too few parameters given')
    sys.exit(1)
elif given < n_var:
    print('Too many parameters given')
    sys.exit(1)

idx_para = np.array([para[k] is not None for k in para], dtype=np.bool)
parameter_values = np.array(list(para.values()))[idx_para].astype(np.float64)
variable_names = [globals()[k] for k in para if para[k] is None]
parameter_names = [globals()[k] for k in para if para[k] is not None]
substitutions = list(zip(parameter_names, parameter_values))
num_system = [expr.subs(substitutions) for expr in system]

solutions = sp.solve(num_system, *variable_names)
print("Found {:d} solutions".format(len(solutions)))
for ii, sol in enumerate(reversed(solutions)):
    print('\n\n')
    print('*** Solution {:d}'.format(ii + 1))
    variable_values = np.array(sol, dtype=np.float64)

    sol_para = dict(para)
    sol_para.update({sp.pycode(variable_names[i]): variable_values[i] for i in range(n_eqs)})

    # sol_para['chi'] = sol_para['XKR'] * sol_para['kappa']
    sol_para['XKR'] = sol_para['chi'] / sol_para['kappa']
    sol_para['Ec'] = -sol_para['alpha']
    sol_para['Ej'] = sol_para['JCR'] * sol_para['Ec']
    sol_para['wq'] = sol_para['wr'] + sol_para['Delta']

    ncrit_tilde = (sol_para['Delta'] + sol_para['alpha'])**2 / (8 * sol_para['g2'])

    print("chi / kappa = {:.4g}".format(sol_para['XKR']))
    print("wr = {:.4g}".format(sol_para['wr'] / 2 / np.pi))
    print("alpha = {:.4g}".format(sol_para['alpha'] / 2 / np.pi))
    print("kappa = {:.4g}".format(sol_para['kappa'] / 2 / np.pi))
    print("chi = {:.4g}".format(sol_para['chi'] / 2 / np.pi))
    print("Delta = {:.4g}".format(sol_para['Delta'] / 2 / np.pi))
    print("g = {:.4g}".format(np.sqrt(sol_para['g2']) / 2 / np.pi))
    print("wq = {:.4g}".format(sol_para['wq'] / 2 / np.pi))
    print("Gamma = {:.4g}".format(sol_para['Gamma']))
    print("Gamma^-1 = {:.4g}".format(1. / sol_para['Gamma']))
    print("ncrit = {:.4g}".format(sol_para['ncrit']))
    print("ncrit_tilde = {:.4g}".format(ncrit_tilde))
    print("Ec = {:.4g}".format(sol_para['Ec'] / 2 / np.pi))
    print("Ej = {:.4g}".format(sol_para['Ej'] / 2 / np.pi))
    print("Ej / Ec = {:.4g}".format(sol_para['JCR']))
    print("Delta / g = {:.4g}".format(sol_para['Delta'] / np.sqrt(sol_para['g2'])))
