"""
Problem in EM for the estimation of the noise-intensity parameter of a linear dynamical system
==============================================================================================

The code below reproduces a problem in the estimation of the noise-intensity
parameter, :math:`\sigma^2` in the linear dynamical system in Eq. xx.

"""

import numpy as np
import plotly.graph_objs as go

import lds_python.simulation
import lds_python.learning

#%%
# Simulation
# ----------
# We will simulate a linear dynamical system for tracking.

#%%
# Define simulation parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pos_x0 = 0.0
pos_y0 = 0.0
vel_x0 = 0.0
vel_y0 = 0.0
ace_x0 = 0.0
ace_y0 = 0.0
dt = 1e-1
num_pos = 500
sqrt_noise_intensity_true = 0.1
sigma_x = 5e-0
sigma_y = 5e-0
sqrt_diag_V0_value = 1e-3

# Taken from the book
# barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
# section 6.2.3
# Eq. 6.2.3-7
B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
              [0, 1,  dt,       0, 0, 0],
              [0, 0,  1,        0, 0, 0],
              [0, 0,  0,        1, dt, .5*dt**2],
              [0, 0,  0,        0, 1,  dt],
              [0, 0,  0,        0, 0,  1]], dtype=np.double)
Z = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]], dtype=np.double)
# Eq. 6.2.3-8
Qe = np.array([[1+dt**5/20, dt**4/8, dt**3/6, 0, 0, 0],
               [dt**4/8, 1+dt**3/3,  dt**2/2, 0, 0, 0],
               [dt**3/6, dt**2/2,  1+dt,      0, 0, 0],
               [0, 0, 0,                    1+dt**5/20, dt**4/8, dt**3/6],
               [0, 0, 0,                    dt**4/8, 1+dt**3/3,  dt**2/2],
               [0, 0, 0,                    dt**3/6, dt**2/2,  1+dt]],
              dtype=np.double)
Q_true = Qe*sqrt_noise_intensity_true**2
R = np.diag([sigma_x**2, sigma_y**2])
m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
              dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

if m0.ndim == 1:
    m0 = np.expand_dims(m0, 1)

#%%
# Perform simulation
# ^^^^^^^^^^^^^^^^^^

x0, x, y = lds_python.simulation.simulateLDS(N=num_pos, B=B, Q=Q_true, Z=Z, R=R,
                                             m0=m0.squeeze(), V0=V0)

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
# Problem: Expected value of complete log likelihood function should be maximal at the optimal parameter of the log likelihood
# ----------------------------------------------------------------------------------------------------------------------------


#%%
# Compute expected value of complete log likelihood function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

sqrt_noise_intensity_min  = 0.01
sqrt_noise_intensity_max  = 10.0
sqrt_noise_intensity_step = 0.01

N = y.shape[1]
M = Qe.shape[0]

kf = lds_python.inference.filterLDS_SS_withMissingValues_np(
    y=y, B=B, Q=Q_true, m0=m0, V0=V0, Z=Z, R=R)
ks = lds_python.inference.smoothLDS_SS(B=B, xnn=kf["xnn"], Vnn=kf["Vnn"],
                                       xnn1=kf["xnn1"], Vnn1=kf["Vnn1"],
                                       m0=m0, V0=V0)
S11, S10, S00 = lds_python.learning.posteriorCorrelationMatrices(
    Z=Z, B=B, KN=kf["KN"], Vnn=kf["Vnn"], xnN=ks["xnN"], VnN=ks["VnN"],
    x0N=ks["x0N"], V0N=ks["V0N"], Jn=ks["Jn"], J0=ks["J0"])
W = S11 - S10 @ B.T - B @ S10.T + B @ S00 @ B.T
Qe_inv = np.linalg.inv(Qe)
U = W @ Qe_inv
K = np.trace(U)

sqrt_noise_intensities = np.arange(sqrt_noise_intensity_min, sqrt_noise_intensity_max, sqrt_noise_intensity_step)
e_pos_ll = np.empty(len(sqrt_noise_intensities))
for i, sqrt_noise_intensity in enumerate(sqrt_noise_intensities):
    noise_intensity = sqrt_noise_intensity**2
    e_pos_ll[i] = -N*M/2*np.log(noise_intensity)-K/(2*noise_intensity)
index_max = np.argmax(e_pos_ll)
sqrt_noise_intensity_max = sqrt_noise_intensities[index_max]

#%%
# Plot expected value of complete log likelihood function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fig = go.Figure()
trace = go.Scatter(x=sqrt_noise_intensities, y=e_pos_ll, mode="lines+markers")
fig.add_trace(trace)
fig.add_vline(sqrt_noise_intensity_max, line_color="red")
fig.add_vline(sqrt_noise_intensity_true, line_color="blue")
fig.update_layout(xaxis_title="Square Root of Noise Intensity",
                  yaxis_title="Expected Value of Complete LL")
fig

#%%
# Problem: Expected value of complete log likelihood should have the same derivative as the log likelihood
# --------------------------------------------------------------------------------------------------------

#%%
# Compute expected value of complete log likelihood function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

sqrt_noise_intensity_min  = 0.01
sqrt_noise_intensity_max  = 10.0
sqrt_noise_intensity_step = 0.01
sqrt_noise_intensity_old  = 0.75

N = y.shape[1]
M = Qe.shape[0]

Q_old = Qe*sqrt_noise_intensity_old**2

kf = lds_python.inference.filterLDS_SS_withMissingValues_np(
    y=y, B=B, Q=Q_old, m0=m0, V0=V0, Z=Z, R=R)
ks = lds_python.inference.smoothLDS_SS(B=B, xnn=kf["xnn"], Vnn=kf["Vnn"],
                                       xnn1=kf["xnn1"], Vnn1=kf["Vnn1"],
                                       m0=m0, V0=V0)
S11, S10, S00 = lds_python.learning.posteriorCorrelationMatrices(
    Z=Z, B=B, KN=kf["KN"], Vnn=kf["Vnn"], xnN=ks["xnN"], VnN=ks["VnN"],
    x0N=ks["x0N"], V0N=ks["V0N"], Jn=ks["Jn"], J0=ks["J0"])
W = S11 - S10 @ B.T - B @ S10.T + B @ S00 @ B.T
Qe_inv = np.linalg.inv(Qe)
U = W @ Qe_inv
K = np.trace(U)

sqrt_noise_intensities = np.arange(sqrt_noise_intensity_min, sqrt_noise_intensity_max, sqrt_noise_intensity_step)
e_pos_ll = np.empty(len(sqrt_noise_intensities))
for i, sqrt_noise_intensity in enumerate(sqrt_noise_intensities):
    noise_intensity = sqrt_noise_intensity**2
    e_pos_ll[i] = -N*M/2*np.log(noise_intensity)-K/(2*noise_intensity)
index_max = np.argmax(e_pos_ll)
sqrt_noise_intensity_max = sqrt_noise_intensities[index_max]

#%%
# Plot expected value of complete log likelihood function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fig = go.Figure()
trace = go.Scatter(x=sqrt_noise_intensities, y=e_pos_ll, mode="lines+markers")
fig.add_trace(trace)
fig.add_vline(sqrt_noise_intensity_max, line_color="red")
fig.add_vline(sqrt_noise_intensity_true, line_color="blue")
fig.update_layout(xaxis_title="Square Root of Noise Intensity",
                  yaxis_title="Expected Value of Complete LL")
fig

#%%
# Compute the log likelihood of the noise intensity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
log_likes = np.empty(len(sqrt_noise_intensities))
for i, sqrt_noise_intensity in enumerate(sqrt_noise_intensities):
    Q = Qe * sqrt_noise_intensity**2
    filter_res = lds_python.inference.filterLDS_SS_withMissingValues_np(
        y=y, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
    log_likes[i] = filter_res["logLike"]
index_max = np.argmax(log_likes)
sqrt_noise_intensity_max = sqrt_noise_intensities[index_max]

#%%
# Plot the log likelihood of the noise intensity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig = go.Figure()
trace = go.Scatter(x=sqrt_noise_intensities, y=log_likes, mode="lines+markers")
fig.add_trace(trace)
fig.add_vline(sqrt_noise_intensity_max, line_color="red")
fig.add_vline(sqrt_noise_intensity_true, line_color="blue")
fig.update_layout(xaxis_title="Square Root of Noise Intensity",
                  yaxis_title="Log Likelihood")

#%%
# Check: learn all the coefficients of the state noise covariance matrix
# ----------------------------------------------------------------------

#%%
# Define learning parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

tol = 1e-6
max_iter = 100
sqrt_noise_intensity0 = 0.7
Q0 = sqrt_noise_intensity0**2 * Qe

est_res = lds_python.learning.em_LDS_SS(
    y=y, B0=B, Q0=Q0, Z0=Z, R0=R, m0=m0, V0=V0, max_iter=max_iter,
    tol=tol, varsToEstimate=dict(m0=False, V0=False, B=False, Q=True, Z=False,
                                 R=False))

#%%
# Plot log likelihood as  function of iteration number
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
log_like = est_res["log_like"]
iteration = np.arange(1, len(log_like)+1)
fig = go.Figure()
trace = go.Scatter(x=iteration, y=log_like, mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis_title="Iteration number",
                  yaxis_title="Log Likelihood")
fig

#%%
# Check: conditioning of :math:`Qe`
# ---------------------------------

print(f"Condition number of Qe: {np.linalg.cond(Qe)}")

