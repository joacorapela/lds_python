
"""
Offline smoothing of a simulated mouse trajectory
=================================================

The code below performs online Kalman filtering of a simulated mouse
trajectory.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import configparser
import numpy as np
import pandas as pd

import lds_python.inference
import utils

#%%
# Load simulated trajectory
# ~~~~~~~~~~~~~~~~~~~~~~~~~

simRes_filename = "../../results/lds_simulation.npz"
simRes = np.load(simRes_filename)

y = simRes["y"]
x = simRes["x"]
B = simRes["B"]
sigma_a = simRes["sigma_a"]
Qe = simRes["Qe"]
m0 = simRes["m0"]
V0 = simRes["V0"]
Z = simRes["Z"]
R = simRes["R"]
dt = simRes["dt"]

#%%
# Perform batch filtering
# ~~~~~~~~~~~~~~~~~~~~~~~
# View source code of `lds_python.inference.filterLDS_SS_withMissingValues_np
# <https://joacorapela.github.io/lds_python/_modules/lds_python/inference.html#filterLDS_SS_withMissingValues_np>`_

Q = sigma_a*Qe
filterRes = lds_python.inference.filterLDS_SS_withMissingValues_np(
    y=y, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)

#%%
# Perform batch smoothing
# ~~~~~~~~~~~~~~~~~~~~~~~
# View source code of `lds_python.inference.smoothLDS_SS
# <https://joacorapela.github.io/lds_python/_modules/lds_python/inference.html#smoothLDS_SS>`_

smoothRes = lds_python.inference.smoothLDS_SS(
    B=simRes["B"], xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
    xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)

#%%
# Set variables for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

N = y.shape[1]
time = np.arange(0, N*dt, dt)
smoothed_means = smoothRes["xnN"]
smoothed_covs = smoothRes["VnN"]
smoothed_std_x_y = np.sqrt(np.diagonal(a=smoothed_covs, axis1=0, axis2=1))
color_true = "blue"
color_measured = "black"
color_smoothed_pattern = "rgba(255,0,0,{:f})"
cb_alpha = 0.3

#%%
# Plot true, measured and smoothed positions (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[0, :]
true_y = x[3, :]
measured_x = y[0, :]
measured_y = y[1, :]
smoothed_mean_x = smoothed_means[0, 0, :]
smoothed_mean_y = smoothed_means[3, 0, :]

smoothed_ci_x_upper = smoothed_mean_x + 1.96*smoothed_std_x_y[:, 0]
smoothed_ci_x_lower = smoothed_mean_x - 1.96*smoothed_std_x_y[:, 0]
smoothed_ci_y_upper = smoothed_mean_y + 1.96*smoothed_std_x_y[:, 3]
smoothed_ci_y_lower = smoothed_mean_y - 1.96*smoothed_std_x_y[:, 3]

fig = utils.get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x, true_y=true_y,
    measured_x=measured_x, measured_y=measured_y,
    estimated_mean_x=smoothed_mean_x, estimated_mean_y=smoothed_mean_y,
    estimated_ci_x_upper=smoothed_ci_x_upper,
    estimated_ci_y_upper=smoothed_ci_y_upper,
    estimated_ci_x_lower=smoothed_ci_x_lower,
    estimated_ci_y_lower=smoothed_ci_y_lower,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_estimated_pattern=color_smoothed_pattern,
    xlabel="Time (sec)", ylabel="Position (pixels)")
# fig_filename_pattern = "../../figures/smoothed_pos.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

#%%
# Plot true and smoothed velocities (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[1, :]
true_y = x[4, :]
measured_x = None
measured_y = None
smoothed_mean_x = smoothed_means[1, 0, :]
smoothed_mean_y = smoothed_means[4, 0, :]

smoothed_ci_x_upper = smoothed_mean_x + 1.96*smoothed_std_x_y[:, 1]
smoothed_ci_x_lower = smoothed_mean_x - 1.96*smoothed_std_x_y[:, 1]
smoothed_ci_y_upper = smoothed_mean_y + 1.96*smoothed_std_x_y[:, 4]
smoothed_ci_y_lower = smoothed_mean_y - 1.96*smoothed_std_x_y[:, 4]

fig = utils.get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x, true_y=true_y,
    measured_x=measured_x, measured_y=measured_y,
    estimated_mean_x=smoothed_mean_x, estimated_mean_y=smoothed_mean_y,
    estimated_ci_x_upper=smoothed_ci_x_upper,
    estimated_ci_y_upper=smoothed_ci_y_upper,
    estimated_ci_x_lower=smoothed_ci_x_lower,
    estimated_ci_y_lower=smoothed_ci_y_lower,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_estimated_pattern=color_smoothed_pattern,
    xlabel="Time (sec)", ylabel="Velocity (pixels/sec)")
# fig_filename_pattern = "../../figures/smoothed_vel.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

#%%
# Plot true and smoothed accelerations (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[2, :]
true_y = x[5, :]
measured_x = None
measured_y = None
smoothed_mean_x = smoothed_means[2, 0, :]
smoothed_mean_y = smoothed_means[5, 0, :]

smoothed_ci_x_upper = smoothed_mean_x + 1.96*smoothed_std_x_y[:, 2]
smoothed_ci_x_lower = smoothed_mean_x - 1.96*smoothed_std_x_y[:, 2]
smoothed_ci_y_upper = smoothed_mean_y + 1.96*smoothed_std_x_y[:, 5]
smoothed_ci_y_lower = smoothed_mean_y - 1.96*smoothed_std_x_y[:, 5]

fig = utils.get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x, true_y=true_y,
    measured_x=measured_x, measured_y=measured_y,
    estimated_mean_x=smoothed_mean_x, estimated_mean_y=smoothed_mean_y,
    estimated_ci_x_upper=smoothed_ci_x_upper,
    estimated_ci_y_upper=smoothed_ci_y_upper,
    estimated_ci_x_lower=smoothed_ci_x_lower,
    estimated_ci_y_lower=smoothed_ci_y_lower,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_estimated_pattern=color_smoothed_pattern,
    xlabel="Time (sec)", ylabel="Acceleration (pixels/sec^2)")
# fig_filename_pattern = "../../figures/smoothed_acc.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

