"""
Infering a mouse positions, velocities and accelerations
========================================================

"""

import sys
import os
import random
import pickle
import configparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append("../../code/scripts")
import utils
sys.path.append("../../code/src")
import inference

# configuration variables
filtering_params_filename = "../../metadata/00000009_smoothing.ini"
start_position = 0
number_positions = 10000
filtering_params_section = "params"
color_measures = "black"
color_pattern_filtered = "rgba(255,0,0,{:f})"
cb_alpha = 0.3
data_filename ="../../data/positions_session003_start0.00_end15548.27.csv"

# get measurements
data = pd.read_csv(data_filename)
data = data.iloc[start_position:start_position+number_positions,:]
y = np.transpose(data[["x", "y"]].to_numpy())

# get confi params
filtering_params = configparser.ConfigParser()
filtering_params.read(filtering_params_filename)
pos_x0 = float(filtering_params[filtering_params_section]["pos_x0"])
pos_y0 = float(filtering_params[filtering_params_section]["pos_y0"])
vel_x0 = float(filtering_params[filtering_params_section]["vel_x0"])
vel_y0 = float(filtering_params[filtering_params_section]["vel_x0"])
acc_x0 = float(filtering_params[filtering_params_section]["acc_x0"])
acc_y0 = float(filtering_params[filtering_params_section]["acc_x0"])
sigma_ax = float(filtering_params[filtering_params_section]["sigma_ax"])
sigma_ay = float(filtering_params[filtering_params_section]["sigma_ay"])
sigma_x = float(filtering_params[filtering_params_section]["sigma_x"])
sigma_y = float(filtering_params[filtering_params_section]["sigma_y"])
sqrt_diag_V0_value = float(filtering_params[filtering_params_section]["sqrt_diag_V0_value"])

if np.isnan(pos_x0):
    pos_x0 = y[0, 0]
if np.isnan(pos_y0):
    pos_y0 = y[1, 0]

# build KF matrices for traking
date_times = pd.to_datetime(data["time"])
dt = (date_times.iloc[1]-date_times.iloc[0]).total_seconds()
# Taken from the book
# barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
# section 6.3.3
    # Eq. 6.3.3-2
B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
               [0, 1, dt, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, dt, .5*dt**2],
               [0, 0, 0, 0, 1, dt],
               [0, 0, 0, 0, 0, 1]],
              dtype=np.double)
Z = np.array([[1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0]],
              dtype=np.double)
    # Eq. 6.3.3-4
Qt = np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
               [dt**3/2, dt**2,   dt,      0, 0, 0],
               [dt**2/2, dt,      1,       0, 0, 0],
               [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
               [0, 0, 0, dt**3/2, dt**2,   dt],
               [0, 0, 0, dt**2/2, dt,      1]],
              dtype=np.double)
R = np.diag([sigma_x**2, sigma_y**2])
m0 = np.array([y[0, 0], 0, 0, y[1, 0], 0, 0], dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
Q = utils.buildQfromQt_np(Qt=Qt, sigma_ax=sigma_ax, sigma_ay=sigma_ay)

# filter
onlineKF = inference.OnlineKalmanFilter(B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
filtered_means = np.empty((6, 1, y.shape[1]), dtype=np.double)
filtered_covs = np.empty((6, 6, y.shape[1]), dtype=np.double)
for i in range(y.shape[1]):
    _, _ = onlineKF.predict()
    filtered_means[:, 0, i], filtered_covs[:, :, i] = \
        onlineKF.update(y=y[:, i])

# plot positions
time = (pd.to_datetime(data["time"]) - pd.to_datetime(data["time"][0])).dt.total_seconds().to_numpy()
filter_mean_x = filtered_means[0, 0, :]
filter_mean_y = filtered_means[3, 0, :]
filter_std_x_y = np.sqrt(np.diagonal(a=filtered_covs, axis1=0, axis2=1))

filter_ci_x_upper = filter_mean_x + 1.96*filter_std_x_y[:, 0]
filter_ci_x_lower = filter_mean_x - 1.96*filter_std_x_y[:, 0]
filter_ci_y_upper = filter_mean_y + 1.96*filter_std_x_y[:, 3]
filter_ci_y_lower = filter_mean_y - 1.96*filter_std_x_y[:, 3]

fig = go.Figure()
trace_mes_x = go.Scatter(
                x=time, y=y[0, :],
                mode="markers",
                marker={"color": color_measures},
                name="measured x",
                showlegend=True,
            )
trace_mes_y = go.Scatter(
                x=time, y=y[1, :],
                mode="markers",
                marker={"color": color_measures},
                name="measured y",
                showlegend=True,
            )
trace_filter_x = go.Scatter(
                x=time, y=filter_mean_x,
                mode="markers",
                marker={"color": color_pattern_filtered.format(1.0)},
                name="filtered x",
                showlegend=True,
                legendgroup="filtered_x",
            )
trace_filter_x_cb = go.Scatter(
                x=np.concatenate([time, time[::-1]]),
                y=np.concatenate([filter_ci_x_upper,
                                  filter_ci_x_lower[::-1]]),
                fill="toself",
                fillcolor=color_pattern_filtered.format(cb_alpha),
                line=dict(color=color_pattern_filtered.format(0.0)),
                showlegend=False,
                legendgroup="filtered_x",
            )
trace_filter_y = go.Scatter(
                x=time, y=filter_mean_y,
                mode="markers",
                marker={"color":
                        color_pattern_filtered.format(1.0)},
                name="filtered y",
                showlegend=True,
                legendgroup="filtered_y",
            )
trace_filter_y_cb = go.Scatter(
                x=np.concatenate([time, time[::-1]]),
                y=np.concatenate([filter_ci_y_upper,
                                  filter_ci_y_lower[::-1]]),
                fill="toself",
                fillcolor=color_pattern_filtered.format(cb_alpha),
                line=dict(color=color_pattern_filtered.format(0.0)),
                showlegend=False,
                legendgroup="filtered_y",
            )
fig.add_trace(trace_mes_x)
fig.add_trace(trace_mes_y)
fig.add_trace(trace_filter_x)
fig.add_trace(trace_filter_x_cb)
fig.add_trace(trace_filter_y)
fig.add_trace(trace_filter_y_cb)

fig.update_layout(xaxis_title="time (seconds)", yaxis_title="position (pixels)",
                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                  yaxis_range=[filter_mean_x.min(), filter_mean_x.max()])
fig
