from __future__ import division, print_function

import time

import matplotlib.pyplot as plt
import numpy as np

import simulator as sim

if __name__ == "__main__":
    df = 50e6  # Hz
    f_array = np.linspace(1e9, 5.5e9, 1000)
    para = sim.SimulationParameters(
        Cl=1e-15, Cr=1e-12,
        R1=3162., L1=1e-9, C1=1e-12,
        R2=3162., L2=2e-9, C2=2e-12,
        w_arr=[1e9, ], A_arr=[1., ], P_arr=[0., ],
    )

    # dw = 2. * np.pi * 250e3
    # wd_array = 2. * np.pi * np.linspace(2e9, 4e9, 100)
    # para = SimulationParameters(
    #     Cl=20e-15, Cr=10e-15,
    #     R1=1e6, L1=8e-9, C1=400e-15,
    #     R2=1e6, L2=8e-9, C2=400e-15,
    #     w_arr=[wd_array[0], ], A_arr=[1., ], P_arr=[0., ],
    #     dw=dw,
    # )

    actual_f_array = np.zeros_like(f_array)
    resp1_array = np.zeros_like(f_array, dtype=np.complex128)
    resp2_array = np.zeros_like(f_array, dtype=np.complex128)

    # Run first time to get initial condition
    fd_, df_ = para.tune(f_array[0], df, prio_f=True)
    para.set_df(df_)
    para.set_drive_frequencies([fd_, ])
    para.set_Nbeats(5)
    init = np.array([0., 0., 0., 0.])
    sol = para.simulate(init=init)

    para.set_Nbeats(2)
    t_start = time.time()
    for ii, fd in enumerate(f_array):
        print(ii)
        fd_, df_ = para.tune(fd, df, prio_f=True)
        para.set_df(df_)
        para.set_drive_frequencies([fd_, ])
        # para.set_noise_T(300.)
        sol = para.simulate(init=init)

        # V1 = sol[-para.ns:, 0]
        V1 = sol[-para.ns:, 1]
        V1_fft = np.fft.rfft(V1) / para.ns
        # V2 = sol[-para.ns:, 2]
        V2 = sol[-para.ns:, 3]
        V2_fft = np.fft.rfft(V2) / para.ns
        actual_f_array[ii] = para.w_arr[0] / (2. * np.pi)
        resp1_array[ii] = V1_fft[para.n_arr[0]]
        resp2_array[ii] = V2_fft[para.n_arr[0]]
    t_end = time.time()
    t_tot = t_end - t_start
    print("Total run took {:s}.".format(sim.format_sec(t_tot)))

    G1 = para.tf1(f_array)
    G2 = para.tf2(f_array)

    fig1, ax1 = plt.subplots(1, 1, sharex=True, tight_layout=True)
    ax1r = ax1.twinx()
    ax1.axvline(para.f01, ls='--', c='tab:gray')
    ax1.axvline(para.f02, ls='--', c='tab:olive')
    ax1.semilogy(actual_f_array, np.abs(resp1_array), '.', c='tab:blue')
    ax1.semilogy(f_array, np.abs(G1) / 2, '--', c='k')
    ax1r.plot(actual_f_array, np.angle(resp1_array), '.', c='tab:orange')
    ax1r.plot(f_array, np.angle(G1), '--', c='k')
    ax1.set_ylabel(r"Amplitude 1 [$\mathrm{V}$]")
    ax1r.set_ylabel(r"Phase 1 [$\mathrm{rad}$]")
    ax1.set_xlabel(r"Frequency [$\mathrm{Hz}$]")
    ax1.spines['left'].set_color('tab:blue')
    ax1.tick_params(axis='y', colors='tab:blue')
    ax1.yaxis.label.set_color('tab:blue')
    ax1r.spines['right'].set_color('tab:orange')
    ax1r.tick_params(axis='y', colors='tab:orange')
    ax1r.yaxis.label.set_color('tab:orange')
    fig1.show()

    fig2, ax2 = plt.subplots(1, 1, sharex=True, tight_layout=True)
    ax2r = ax2.twinx()
    ax2.axvline(para.f01, ls='--', c='tab:gray')
    ax2.axvline(para.f02, ls='--', c='tab:olive')
    ax2.semilogy(actual_f_array, np.abs(resp2_array), '.', c='tab:blue')
    ax2.semilogy(f_array, np.abs(G2) / 2, '--', c='k')
    ax2r.plot(actual_f_array, np.angle(resp2_array), '.', c='tab:orange')
    ax2r.plot(f_array, np.angle(G2), '--', c='k')
    ax2.set_ylabel(r"Amplitude 2 [$\mathrm{V}$]")
    ax2r.set_ylabel(r"Phase 2 [$\mathrm{rad}$]")
    ax2.set_xlabel(r"Frequency [$\mathrm{Hz}$]")
    ax2.spines['left'].set_color('tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2r.spines['right'].set_color('tab:orange')
    ax2r.tick_params(axis='y', colors='tab:orange')
    ax2r.yaxis.label.set_color('tab:orange')
    fig2.show()
