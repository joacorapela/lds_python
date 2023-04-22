"""
Comparison between EM and gradient ascent for parameter learning
================================================================

The code below compares the expectation maximization (EM) and gradient ascent
algorithm for learning the parameters of a linear dynamical sytem for tracking.


"""

import numpy as np
import torch
import plotly.graph_objs as go

import lds_python.simulation
import lds_python.learning

#%%
# Simulation
# ----------
# We will simulate a linear dynamical system for tracking.
# The parameters of this model are highlighted in red.
#
# .. math::
#    \begin{equation}
#        \begin{alignedat}{4}
#            & x_n&&=A x_{n-1}+w_n&&\quad with\quad&&w_n\sim\mathcal{N}(0,\textcolor{red}{\sigma^2}Q_e),&&\quad x_n\in\mathbb{R}^6&&\label{eq:lds}\\
#            & y_n&&=C x_n+v_n&&\quad with\quad&&v_n\sim\mathcal{N}(0,R),&&\quad y_n\in\mathbb{R}^2,&&\quad R=\left[\begin{array}{c,c}
#                                                                                                                       \textcolor{red}{\sigma_x^2}&0\\
#                                                                                                                       0&\textcolor{red}{\sigma_y^2}
#                                                                                                                    \end{array}\right]\\
#            & x_0&&\in\mathcal{N}(\textcolor{red}{m_0},\textcolor{red}{V_0})&& &&
#        \end{alignedat}
#    \end{equation}
#

#%%
# Define simulation parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pos_x0 = 0.0
pos_y0 = 0.0
vel_x0 = 0.0
vel_y0 = 0.0
acc_x0 = 0.0
acc_y0 = 0.0
dt = 1e-1
num_pos = 500
sqrt_noise_intensity_true = 0.1
sigma_x = 5e-0
sigma_y = 5e-0
sqrt_diag_V0_value = 1e-3

# Taken from the book
# barShalomEtAl01^estimationWithApplicationToTrackingAndNavigation.pdf
# section 6.2.3
# Eq. 6.2.3^7
B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
              [0, 1,  dt,       0, 0, 0],
              [0, 0,  1,        0, 0, 0],
              [0, 0,  0,        1, dt, .5*dt**2],
              [0, 0,  0,        0, 1,  dt],
              [0, 0,  0,        0, 0,  1]], dtype=np.double)
Z = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]], dtype=np.double)
# Eq. 6.2.3^8
Qe = np.array([[dt**5/20, dt**4/8, dt**3/6, 0, 0, 0],
               [dt**4/8,  dt**3/3, dt**2/2, 0, 0, 0],
               [dt**3/6,  dt**2/2, dt,      0, 0, 0],
               [0, 0, 0,                    dt**5/20, dt**4/8, dt**3/6],
               [0, 0, 0,                    dt**4/8,  dt**3/3, dt**2/2],
               [0, 0, 0,                    dt**3/6,  dt**2/2, dt]],
              dtype=np.double)
Q_true = Qe*sqrt_noise_intensity_true**2
sqrt_diag_R = np.array([sigma_x, sigma_y])
R = np.diag(sqrt_diag_R**2)
m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
              dtype=np.double)
sqrt_diag_V0 = np.ones(6)*sqrt_diag_V0_value
V0 = np.diag(sqrt_diag_V0**2)

#%%
# Perform simulation
# ^^^^^^^^^^^^^^^^^^
# Code for `lds_python.simulation.simulateLDS
# <https://joacorapela.github.io/lds_python/_modules/lds_python/simulation.html#simulateLDS>`_
x0, x, y = lds_python.simulation.simulateLDS(N=num_pos, B=B, Q=Q_true, Z=Z, R=R,
                                             m0=m0, V0=V0)

#%%
# Plot simulated state and measurement positions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig = go.Figure()
trace_x = go.Scatter(x=x[0, :], y=x[3, :], mode="lines+markers",
                     showlegend=True, name="state")
trace_y = go.Scatter(x=y[0, :], y=y[1, :], mode="lines+markers",
                     showlegend=True, name="measurement", opacity=0.3)
trace_start = go.Scatter(x=[x0[0]], y=[x0[3]], mode="markers",
                         text="initial state", marker={"size": 7},
                         showlegend=False)
fig.add_trace(trace_x)
fig.add_trace(trace_y)
fig.add_trace(trace_start)
fig.update_layout(xaxis_title="horizontal direction",
                  yaxis_title="vertical direction")
fig

#%%
# Estimation of the :math:`\sigma^2` parameter only
# -------------------------------------------------

sqrt_noise_intensity0 = 0.25
ga_vars_to_estimate = {"sqrt_noise_intensity": True, "sqrt_diag_R": False,
                       "m0": False, "sqrt_diag_V0": False}
em_vars_to_estimate = {"sqrt_noise_intensity": True, "R": False,
                       "m0": False, "V0": False}
em_max_iter = 200

#%%
# Grid search
# ^^^^^^^^^^^^

sqrt_noise_intensity_min = 0.001
sqrt_noise_intensity_max = 2.0
sqrt_noise_intensity_step = 0.005
sqrt_noise_intensities = np.arange(sqrt_noise_intensity_min, sqrt_noise_intensity_max, sqrt_noise_intensity_step)
gs_log_likes = np.empty(len(sqrt_noise_intensities))
for i, sqrt_noise_intensity in enumerate(sqrt_noise_intensities):
    Q_gs = Qe * sqrt_noise_intensity**2
    filterRes = lds_python.inference.filterLDS_SS_withMissingValues_np(
        y=y, B=B, Q=Q_gs, m0=m0, V0=V0, Z=Z, R=R)
    gs_log_likes[i] = filterRes["logLike"]
    print(f"log likelihood for sqrt_noise_intensity={sqrt_noise_intensity:.04f}: {gs_log_likes[i]}")
argmax = np.argmax(gs_log_likes)
gs_max_ll = gs_log_likes[argmax]
gs_max_sqrt_noise_intensity = sqrt_noise_intensities[argmax]
print(f"max log^likelihood: {gs_max_ll}, "
      f"max sqrt noise intensity: {gs_max_sqrt_noise_intensity}")

#%%
# Gradient acent
# ^^^^^^^^^^^^^^
# Code for `lds_python.learning.torch_optimize_SS_tracking_diagV0
# <https://joacorapela.github.io/lds_python/_modules/lds_python/learning.html#torch_optimize_SS_tracking_diagV0>`_

tolerance_grad = 1e-9
tolerance_change = 1e-7
sqrt_diag_R_torch = torch.DoubleTensor([sigma_x, sigma_y])
m0_torch = torch.from_numpy(m0)
sqrt_diag_V0_torch = torch.DoubleTensor([sqrt_diag_V0_value
                                         for i in range(len(m0))])
y_torch = torch.from_numpy(y.astype(np.double))
B_torch = torch.from_numpy(B.astype(np.double))
Qe_torch = torch.from_numpy(Qe.astype(np.double))
Z_torch = torch.from_numpy(Z.astype(np.double))
optim_res_ga = lds_python.learning.torch_optimize_SS_tracking_diagV0(
    y=y_torch, B=B_torch, sqrt_noise_intensity0=sqrt_noise_intensity0,
    Qe=Qe_torch, Z=Z_torch, sqrt_diag_R_0=sqrt_diag_R_torch, m0_0=m0_torch,
    sqrt_diag_V0_0=sqrt_diag_V0_torch, max_iter=em_max_iter,
    vars_to_estimate=ga_vars_to_estimate, tolerance_grad=tolerance_grad,
    tolerance_change=tolerance_change)

#%%
# EM
# ^^
# Code for `lds_python.learning.em_SS_tracking
# <https://joacorapela.github.io/lds_python/_modules/lds_python/learning.html#em_SS_tracking>`_
Qe_reg_param = 1e-5
Qe_regularized = Qe + Qe_reg_param*np.eye(Qe.shape[0])
optim_res_em = lds_python.learning.em_SS_tracking(
    y=y, B=B, sqrt_noise_intensity0=sqrt_noise_intensity0,
    Qe=Qe_regularized, Z=Z, R_0=R, m0_0=m0, V0_0=V0,
    vars_to_estimate=em_vars_to_estimate, max_iter=em_max_iter)

#%%
# Plots
# ^^^^^

#%%
# Convergence
# """""""""""

fig = go.Figure()
trace = go.Scatter(x=optim_res_ga["elapsed_time"], y=optim_res_ga["log_like"],
                  name="Gradient ascent", mode="lines+markers")
fig.add_trace(trace)
trace = go.Scatter(x=optim_res_em["elapsed_time"], y=optim_res_em["log_like"],
                   name="EM", mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis_title="Elapsed Time (sec)",
                  yaxis_title="Log Likelihood")
fig

#%%
# Noise intensity estimates
# """""""""""""""""""""""""

fig = go.Figure()
trace = go.Bar(x=["Grid Search", "Gradient Ascent", "EM"],
               y=[gs_max_sqrt_noise_intensity**2,
                  optim_res_ga["estimates"]["sqrt_noise_intensity"]**2,
                  optim_res_em["sqrt_noise_intensity"]**2])
fig.add_trace(trace)
fig.add_hline(y=sqrt_noise_intensity_true**2)
fig.update_layout(xaxis_title="Estimation Method",
                  yaxis_title="Noise Intensity")
fig

#%%
# Estimation of all parameters
# ----------------------------

pos_x0_0 = 0.1
pos_y0_0 = -0.1
vel_x0_0 = 0.01
vel_y0_0 = -0.01
acc_x0_0 = 0.001
acc_y0_0 = -0.001
sqrt_diag_V0_value0 = 1e-2
sqrt_noise_intensity0 = 0.25
sigma_x_0 = 5e-2
sigma_y_0 = 5e-2
ga_vars_to_estimate = {"sqrt_noise_intensity": True, "sqrt_diag_R": True, "m0": True,
                       "sqrt_diag_V0": True}
em_vars_to_estimate = {"sqrt_noise_intensity": True, "R": True, "m0": True,
                       "V0": True}
em_max_iter = 400

m0_0 = np.array([pos_x0_0, vel_x0_0, acc_x0_0, pos_y0_0, vel_y0_0, acc_y0_0])
sqrt_diag_V0_0 = np.array([sqrt_diag_V0_value0 for i in range(len(m0_0))])
sqrt_diag_R_0 = np.array([sigma_x_0, sigma_y_0])

#%%
# Gradient acent
# ^^^^^^^^^^^^^^
# Code for `lds_python.learning.torch_optimize_SS_tracking_diagV0
# <https://joacorapela.github.io/lds_python/_modules/lds_python/learning.html#torch_optimize_SS_tracking_diagV0>`_

tolerance_grad = 1e-9
tolerance_change = 1e-7
m0_0_torch = torch.from_numpy(m0_0)
sqrt_diag_V0_0_torch = torch.from_numpy(sqrt_diag_V0_0)
sqrt_diag_R_0_torch = torch.from_numpy(sqrt_diag_R_0)

y_torch = torch.from_numpy(y.astype(np.double))
B_torch = torch.from_numpy(B.astype(np.double))
Qe_torch = torch.from_numpy(Qe.astype(np.double))
Z_torch = torch.from_numpy(Z.astype(np.double))
optim_res_ga = lds_python.learning.torch_optimize_SS_tracking_diagV0(
    y=y_torch, B=B_torch, sqrt_noise_intensity0=sqrt_noise_intensity0,
    Qe=Qe_torch, Z=Z_torch, sqrt_diag_R_0=sqrt_diag_R_0_torch, m0_0=m0_0_torch,
    sqrt_diag_V0_0=sqrt_diag_V0_0_torch, max_iter=em_max_iter,
    vars_to_estimate=ga_vars_to_estimate, tolerance_grad=tolerance_grad,
    tolerance_change=tolerance_change)

#%%
# EM
# ^^
# Code for `lds_python.learning.em_SS_tracking
# <https://joacorapela.github.io/lds_python/_modules/lds_python/learning.html#em_SS_tracking>`_
V0_0 = np.diag(sqrt_diag_V0_0**2)
R0 = np.diag(sqrt_diag_R_0)
Qe_reg_param = 1e-5
Qe_regularized = Qe + Qe_reg_param*np.eye(Qe.shape[0])
optim_res_em = lds_python.learning.em_SS_tracking(
    y=y, B=B, sqrt_noise_intensity0=sqrt_noise_intensity0,
    Qe=Qe_regularized, Z=Z, R_0=R0, m0_0=m0_0, V0_0=V0_0,
    vars_to_estimate=em_vars_to_estimate, max_iter=em_max_iter)

#%%
# Plots
# ^^^^^

#%%
# Convergence
# """""""""""

fig = go.Figure()
trace = go.Scatter(x=optim_res_ga["elapsed_time"], y=optim_res_ga["log_like"],
                  name="Gradient ascent", mode="lines+markers")
fig.add_trace(trace)
trace = go.Scatter(x=optim_res_em["elapsed_time"], y=optim_res_em["log_like"],
                   name="EM", mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis_title="Elapsed Time (sec)",
                  yaxis_title="Log Likelihood")
fig

#%%
# :math:`\sigma^2` estimates
# """"""""""""""""""""""""""

fig = go.Figure()
trace = go.Bar(x=["Gradient Ascent", "EM"],
               y=[optim_res_ga["estimates"]["sqrt_noise_intensity"]**2,
                  optim_res_em["sqrt_noise_intensity"]**2])
fig.add_trace(trace)
fig.add_hline(y=sqrt_noise_intensity_true**2)
fig.update_layout(xaxis_title="Estimation Method",
                  yaxis_title=r"\sigma^2")
fig

#%%
# :math:`\sigma_x^2` estimates
# """"""""""""""""""""""""""""

fig = go.Figure()
trace = go.Bar(x=["Gradient Ascent", "EM"],
               y=[optim_res_ga["estimates"]["sqrt_diag_R"][0]**2,
                  optim_res_em["R"][0, 0]])
fig.add_trace(trace)
fig.add_hline(y=sqrt_diag_R[0]**2)
fig.update_layout(xaxis_title="Estimation Method",
                  yaxis_title=r"$\sigma_x$")
fig

#%%
# :math:`m_0[0]` estimates
# """"""""""""""""""""""""

fig = go.Figure()
trace = go.Bar(x=["Gradient Ascent", "EM"],
               y=[optim_res_ga["estimates"]["m0"][0],
                  optim_res_em["m0"][0]])
fig.add_trace(trace)
fig.add_hline(y=m0[0])
fig.update_layout(xaxis_title="Estimation Method",
                  yaxis_title=r"$m_0[0]$")
fig

#%%
# :math:`V_0[0,0]` estimates
# """"""""""""""""""""""""""

fig = go.Figure()
trace = go.Bar(x=["Gradient Ascent", "EM"],
               y=[optim_res_ga["estimates"]["sqrt_diag_V0"][0]**2,
                  optim_res_em["V0"][0, 0]])
fig.add_trace(trace)
fig.add_hline(y=sqrt_diag_V0[0]**2)
fig.update_layout(xaxis_title="Estimation Method",
                  yaxis_title=r"$V_0[0, 0]$")
fig

