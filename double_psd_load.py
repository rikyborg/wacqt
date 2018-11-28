from __future__ import division, print_function

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

# import simulator as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

# filename = "double_AMP_0.0e+00_V_df_1.6e+06_Hz_Navg_10000.npz"
filename = "double_AMP_5.0e-01_V_df_1.6e+06_Hz_Navg_10000.npz"

with np.load(filename) as npzfile:
    para = np.asscalar(npzfile['para'])

    freqs = npzfile['freqs']
    psd1 = npzfile['psd1']
    psd2 = npzfile['psd2']
    resp_low = npzfile['resp_low']
    resp_high = npzfile['resp_high']
    karray = npzfile['karray'],
    resp_zoom = npzfile['resp_zoom'],

    AMP = np.asscalar(npzfile['AMP'])
    fH_ = np.asscalar(npzfile['fH_'])
    fL_ = np.asscalar(npzfile['fL_'])
    QH = np.asscalar(npzfile['QH'])
    QL = np.asscalar(npzfile['QL'])
    center = np.asscalar(npzfile['center'])
    split = np.asscalar(npzfile['split'])
    width = np.asscalar(npzfile['width'])
    noise_T = np.asscalar(npzfile['noise_T'])
    fP_ = np.asscalar(npzfile['fP_'])
    df_ = np.asscalar(npzfile['df_'])
    fP = np.asscalar(npzfile['fP'])
    df = np.asscalar(npzfile['df'])
    fL = np.asscalar(npzfile['fL'])
    fH = np.asscalar(npzfile['fH'])

idx_zoom = np.logical_and(freqs > center - 1.5 * split, freqs < center + 1.5 * split)
freqs_zoom = freqs[idx_zoom]
psd1_zoom = psd1[idx_zoom]
psd2_zoom = psd2[idx_zoom]

fig1, ax1 = plt.subplots(tight_layout=True)
ax1.semilogy(freqs_zoom, psd1_zoom, label='cavity V1')
ax1.semilogy(freqs_zoom, psd2_zoom, label='qubit V2')
ax1.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax1.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax1.set_title(r"$A_\mathrm{P}$ = " + "{:.1e} V".format(AMP))
ax1.legend()
# ax1.axvline(fP_, ls='--')
# ax1.axvline(fL_, ls='--')
# ax1.axvline(fH_, ls='--')
ax1.axvline(fP, ls='--')
ax1.axvline(fL, ls='--')
ax1.axvline(fH, ls='--')
fig1.show()

# fig2, ax2 = plt.subplots(3, 2, tight_layout=True)
# (ax21, ax22), (ax23, ax24), (ax25, ax26) = ax2
# ax21.hist2d(resp_low.real, resp_high.real, bins=100)
# ax22.hist2d(resp_low.real, resp_high.imag, bins=100)
# ax23.hist2d(resp_low.imag, resp_high.real, bins=100)
# ax24.hist2d(resp_low.imag, resp_high.imag, bins=100)
# ax25.hist2d(resp_low.real, resp_low.imag, bins=100)
# ax26.hist2d(resp_high.real, resp_high.imag, bins=100)
# for ax_ in fig2.get_axes():
#     ax_.set_aspect('equal')
# ax21.set_title(r"$A_\mathrm{P}$ = " + "{:.1e} V".format(AMP))
# fig2.show()

I_low = (resp_low.real - resp_low.real.mean()) / resp_low.real.std()
Q_low = (resp_low.imag - resp_low.imag.mean()) / resp_low.imag.std()
I_high = (resp_high.real - resp_high.real.mean()) / resp_high.real.std()
Q_high = (resp_high.imag - resp_high.imag.mean()) / resp_high.imag.std()

fig3, ax3 = plt.subplots(3, 2, tight_layout=True)
(ax31, ax32), (ax33, ax34), (ax35, ax36) = ax3
ax31.hist2d(I_low, I_high, bins=100, range=[[-5, 5], [-5, 5]])
ax31.set_xlabel(r"$I_\mathrm{L}$")
ax31.set_ylabel(r"$I_\mathrm{H}$")
ax32.hist2d(I_low, Q_high, bins=100, range=[[-5, 5], [-5, 5]])
ax32.set_xlabel(r"$I_\mathrm{L}$")
ax32.set_ylabel(r"$Q_\mathrm{H}$")
ax33.hist2d(Q_low, I_high, bins=100, range=[[-5, 5], [-5, 5]])
ax33.set_xlabel(r"$Q_\mathrm{L}$")
ax33.set_ylabel(r"$I_\mathrm{H}$")
ax34.hist2d(Q_low, Q_high, bins=100, range=[[-5, 5], [-5, 5]])
ax34.set_xlabel(r"$Q_\mathrm{L}$")
ax34.set_ylabel(r"$Q_\mathrm{H}$")
ax35.hist2d(I_low, Q_low, bins=100, range=[[-5, 5], [-5, 5]])
ax35.set_xlabel(r"$I_\mathrm{L}$")
ax35.set_ylabel(r"$Q_\mathrm{L}$")
ax36.hist2d(I_high, Q_high, bins=100, range=[[-5, 5], [-5, 5]])
ax36.set_xlabel(r"$I_\mathrm{H}$")
ax36.set_ylabel(r"$Q_\mathrm{H}$")
for ax_ in fig3.get_axes():
    ax_.set_aspect('equal')
    ax_.set_xticks([])
    ax_.set_yticks([])
fig3.suptitle(r"$A_\mathrm{P}$ = " + "{:.1e} V".format(AMP))
fig3.show()