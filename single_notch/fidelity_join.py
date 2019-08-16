import numpy as np

file_list = [
    'fidelity_chi_2e+06_kappa_2e+05_Nruns_16384_1.npz',
    'fidelity_chi_2e+06_kappa_2e+05_Nruns_16384_2.npz',
    'fidelity_chi_2e+06_kappa_2e+05_Nruns_16384_3.npz',
    'fidelity_chi_2e+06_kappa_2e+05_Nruns_16384_4.npz',
]

N = int(file_list[0].split('_')[-2])
M = len(file_list)

state_arr_all = np.zeros(N * M, dtype=np.bool)
decision_arr_all = np.zeros(N * M, dtype=np.float64)
dist_g_arr_all = np.zeros(N * M, dtype=np.float64)
dist_e_arr_all = np.zeros(N * M, dtype=np.float64)

for ii, file_name in enumerate(file_list):
    with np.load(file_name) as npz:
        state_arr = npz['state_arr']
        decision_arr = npz['decision_arr']
        dist_g_arr = npz['dist_g_arr']
        dist_e_arr = npz['dist_e_arr']
        para_g = np.asscalar(npz['para_g'])
        para_e = np.asscalar(npz['para_e'])
        template_g = npz['template_g']
        template_e = npz['template_e']
        threshold = np.asscalar(npz['threshold'])
    state_arr_all[ii * N:(ii + 1) * N] = state_arr[:]
    decision_arr_all[ii * N:(ii + 1) * N] = decision_arr[:]
    dist_g_arr_all[ii * N:(ii + 1) * N] = dist_g_arr[:]
    dist_e_arr_all[ii * N:(ii + 1) * N] = dist_e_arr[:]

new_file_name = '_'.join(file_name.split('_')[:-2]) + "_{:d}.npz".format(N * M)
np.savez(
    new_file_name,
    state_arr=state_arr_all,
    decision_arr=decision_arr_all,
    dist_g_arr=dist_g_arr_all,
    dist_e_arr=dist_e_arr_all,
    para_g=para_g,
    para_e=para_e,
    template_g=template_g,
    template_e=template_e,
    threshold=threshold,
)
