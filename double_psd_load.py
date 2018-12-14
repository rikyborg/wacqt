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
    karray = npzfile['karray']
    resp_zoom = npzfile['resp_zoom']

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

kP = int(round(fP / df))
iP = np.where(karray == kP)[0][0]
nr_pairs = min(kP - karray[0], karray[-1] - kP)
kL = int(round(fL / df))
kH = int(round(fH / df))
iL = np.where(karray == kL)[0][0]
iH = np.where(karray == kH)[0][0]
pair = kP - kL
assert pair == kH - kP
assert iL == iP - pair
assert iH == iP + pair


def on_key_press(event):
    global pair, kL, kH, iL, iH
    if event.key == 'left':
        pair -= 1
    elif event.key == 'right':
        pair += 1
    else:
        return

    if pair < 1:
        pair = 1
    if pair > nr_pairs:
        pair = nr_pairs
    kL = kP - pair
    kH = kP + pair

    lhl.set_data(freqs[[kL, kH]], psd1[[kL, kH]])

    H1, H2, H3, H4, H5, H6 = make_hists(pair)
    im1.set_data(H1)
    im1.autoscale()
    im2.set_data(H2)
    im2.autoscale()
    im3.set_data(H3)
    im3.autoscale()
    im4.set_data(H4)
    im4.autoscale()
    im5.set_data(H5)
    im5.autoscale()
    im6.set_data(H6)
    im6.autoscale()
    fig.canvas.draw()


def make_hists(pair):
    iL = iP - pair
    iH = iP + pair
    I_low = (resp_zoom[iL, :].real - resp_zoom[iL, :].real.mean()) / resp_zoom[iL, :].real.std()
    Q_low = (resp_zoom[iL, :].imag - resp_zoom[iL, :].imag.mean()) / resp_zoom[iL, :].imag.std()
    I_high = (resp_zoom[iH, :].real - resp_zoom[iH, :].real.mean()) / resp_zoom[iH, :].real.std()
    Q_high = (resp_zoom[iH, :].imag - resp_zoom[iH, :].imag.mean()) / resp_zoom[iH, :].imag.std()
    H1, xedges, yedges = np.histogram2d(I_low, I_high, bins=100, range=[[-5, 5], [-5, 5]], normed=True)
    H2, xedges, yedges = np.histogram2d(I_low, Q_high, bins=100, range=[[-5, 5], [-5, 5]], normed=True)
    H3, xedges, yedges = np.histogram2d(Q_low, I_high, bins=100, range=[[-5, 5], [-5, 5]], normed=True)
    H4, xedges, yedges = np.histogram2d(Q_low, Q_high, bins=100, range=[[-5, 5], [-5, 5]], normed=True)
    H5, xedges, yedges = np.histogram2d(I_low, Q_low, bins=100, range=[[-5, 5], [-5, 5]], normed=True)
    H6, xedges, yedges = np.histogram2d(I_high, Q_high, bins=100, range=[[-5, 5], [-5, 5]], normed=True)
    return H1, H2, H3, H4, H5, H6


idx_zoom = np.logical_and(freqs > center - 1.5 * split, freqs < center + 1.5 * split)
freqs_zoom = freqs[idx_zoom]
psd1_zoom = psd1[idx_zoom]
psd2_zoom = psd2[idx_zoom]

fig = plt.figure(tight_layout=True, figsize=(6.4 * 2, 4.8 * 1.5))
ax1 = fig.add_subplot(1, 2, 1)
ax31 = fig.add_subplot(3, 4, 3)
ax32 = fig.add_subplot(3, 4, 4)
ax33 = fig.add_subplot(3, 4, 7)
ax34 = fig.add_subplot(3, 4, 8)
ax35 = fig.add_subplot(3, 4, 11)
ax36 = fig.add_subplot(3, 4, 12)
ax3 = [ax31, ax32, ax33, ax34, ax35, ax36]
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
fig.canvas.mpl_connect("key_press_event", on_key_press)

# fig1, ax1 = plt.subplots(tight_layout=True)
ax1.axvline(fP, ls='--', c='tab:gray')
ax1.axvline(fL, ls='--', c='tab:gray')
ax1.axvline(fH, ls='--', c='tab:gray')
ax1.semilogy(freqs_zoom, psd1_zoom, '-', c='tab:orange', label='cavity V1')
# ax1.semilogy(freqs_zoom, psd2_zoom, label='qubit V2')
lhl, = ax1.semilogy(freqs[[kL, kH]], psd1[[kL, kH]], '.', c='tab:blue', ms=12)
ax1.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax1.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax1.set_title(r"$A_\mathrm{P}$ = " + "{:.1e} V".format(AMP))
# ax1.legend()
# ax1.axvline(fP_, ls='--')
# ax1.axvline(fL_, ls='--')
# ax1.axvline(fH_, ls='--')
# fig1.show()

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

I_low = (resp_zoom[iL, :].real - resp_zoom[iL, :].real.mean()) / resp_zoom[iL, :].real.std()
Q_low = (resp_zoom[iL, :].imag - resp_zoom[iL, :].imag.mean()) / resp_zoom[iL, :].imag.std()
I_high = (resp_zoom[iH, :].real - resp_zoom[iH, :].real.mean()) / resp_zoom[iH, :].real.std()
Q_high = (resp_zoom[iH, :].imag - resp_zoom[iH, :].imag.mean()) / resp_zoom[iH, :].imag.std()
H1, H2, H3, H4, H5, H6 = make_hists(pair)

# fig3, ax3 = plt.subplots(3, 2, tight_layout=True)
# (ax31, ax32), (ax33, ax34), (ax35, ax36) = ax3
im1 = ax31.imshow(H1)
ax31.set_xlabel(r"$I_\mathrm{L}$")
ax31.set_ylabel(r"$I_\mathrm{H}$")
im2 = ax32.imshow(H2)
ax32.set_xlabel(r"$I_\mathrm{L}$")
ax32.set_ylabel(r"$Q_\mathrm{H}$")
im3 = ax33.imshow(H3)
ax33.set_xlabel(r"$Q_\mathrm{L}$")
ax33.set_ylabel(r"$I_\mathrm{H}$")
im4 = ax34.imshow(H4)
ax34.set_xlabel(r"$Q_\mathrm{L}$")
ax34.set_ylabel(r"$Q_\mathrm{H}$")
im5 = ax35.imshow(H5)
ax35.set_xlabel(r"$I_\mathrm{L}$")
ax35.set_ylabel(r"$Q_\mathrm{L}$")
im6 = ax36.imshow(H6)
ax36.set_xlabel(r"$I_\mathrm{H}$")
ax36.set_ylabel(r"$Q_\mathrm{H}$")
# for ax_ in fig3.get_axes():
for ax_ in ax3:
    ax_.set_aspect('equal')
    ax_.set_xticks([])
    ax_.set_yticks([])
# fig3.suptitle(r"$A_\mathrm{P}$ = " + "{:.1e} V".format(AMP))
# fig3.show()
fig.show()
