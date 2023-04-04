
"""
Offline filtering of a simulated mouse trajectory
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
from ipynb.fs.full.utils import get_fig_kinematics_vs_time

#%%
# Load simulated trajectory
# ~~~~~~~~~~~~~~~~~~~~~~~~~

simRes_filename = "./results/lds_simulation.npz"
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
# Set variables for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

N = y.shape[1]
time = np.arange(0, N*dt, dt)
filtered_means = filterRes["xnn"]
filtered_covs = filterRes["Vnn"]
filter_std_x_y = np.sqrt(np.diagonal(a=filtered_covs, axis1=0, axis2=1))
color_true = "blue"
color_measured = "black"
color_filtered_pattern = "rgba(255,0,0,{:f})"
cb_alpha = 0.3

#%%
# Plot true, measured and filtered positions (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[0, :]
true_y = x[3, :]
measured_x = y[0, :]
measured_y = y[1, :]
filter_mean_x = filtered_means[0, 0, :]
filter_mean_y = filtered_means[3, 0, :]

filter_ci_x_upper = filter_mean_x + 1.96*filter_std_x_y[:, 0]
filter_ci_x_lower = filter_mean_x - 1.96*filter_std_x_y[:, 0]
filter_ci_y_upper = filter_mean_y + 1.96*filter_std_x_y[:, 3]
filter_ci_y_lower = filter_mean_y - 1.96*filter_std_x_y[:, 3]

fig = get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x, true_y=true_y,
    measured_x=measured_x, measured_y=measured_y,
    estimated_mean_x=filter_mean_x, estimated_mean_y=filter_mean_y,
    estimated_ci_x_upper=filter_ci_x_upper,
    estimated_ci_y_upper=filter_ci_y_upper,
    estimated_ci_x_lower=filter_ci_x_lower,
    estimated_ci_y_lower=filter_ci_y_lower,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_estimated_pattern=color_filtered_pattern,
    xlabel="Time (sec)", ylabel="Position (pixels)")
# fig_filename_pattern = "../../figures/filtered_pos.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

#%%
# Plot true and filtered velocities (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[1, :]
true_y = x[4, :]
measured_x = None
measured_y = None
filter_mean_x = filtered_means[1, 0, :]
filter_mean_y = filtered_means[4, 0, :]

filter_ci_x_upper = filter_mean_x + 1.96*filter_std_x_y[:, 1]
filter_ci_x_lower = filter_mean_x - 1.96*filter_std_x_y[:, 1]
filter_ci_y_upper = filter_mean_y + 1.96*filter_std_x_y[:, 4]
filter_ci_y_lower = filter_mean_y - 1.96*filter_std_x_y[:, 4]

fig = get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x, true_y=true_y,
    measured_x=measured_x, measured_y=measured_y,
    estimated_mean_x=filter_mean_x, estimated_mean_y=filter_mean_y,
    estimated_ci_x_upper=filter_ci_x_upper,
    estimated_ci_y_upper=filter_ci_y_upper,
    estimated_ci_x_lower=filter_ci_x_lower,
    estimated_ci_y_lower=filter_ci_y_lower,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_estimated_pattern=color_filtered_pattern,
    xlabel="Time (sec)", ylabel="Velocity (pixels/sec)")
# fig_filename_pattern = "../../figures/filtered_vel.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

#%%
# Plot true and filtered accelerations (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[2, :]
true_y = x[5, :]
measured_x = None
measured_y = None
filter_mean_x = filtered_means[2, 0, :]
filter_mean_y = filtered_means[5, 0, :]

filter_ci_x_upper = filter_mean_x + 1.96*filter_std_x_y[:, 2]
filter_ci_x_lower = filter_mean_x - 1.96*filter_std_x_y[:, 2]
filter_ci_y_upper = filter_mean_y + 1.96*filter_std_x_y[:, 5]
filter_ci_y_lower = filter_mean_y - 1.96*filter_std_x_y[:, 5]

fig = get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x, true_y=true_y,
    measured_x=measured_x, measured_y=measured_y,
    estimated_mean_x=filter_mean_x, estimated_mean_y=filter_mean_y,
    estimated_ci_x_upper=filter_ci_x_upper,
    estimated_ci_y_upper=filter_ci_y_upper,
    estimated_ci_x_lower=filter_ci_x_lower,
    estimated_ci_y_lower=filter_ci_y_lower,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_estimated_pattern=color_filtered_pattern,
    xlabel="Time (sec)", ylabel="Acceleration (pixels/sec^2)")
# fig_filename_pattern = "../../figures/filtered_acc.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

