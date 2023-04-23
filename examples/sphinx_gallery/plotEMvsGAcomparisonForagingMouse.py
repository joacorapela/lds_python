
"""
Comparison between EM and gradient ascent for tracking a foraging mouse
=======================================================================

The code below compares the expectation maximization (EM) and gradient ascent
algorithm for tracking a foraging mouse.


"""

import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np
import pandas as pd
import torch

import lds_python.learning


#%%
# Define estimation parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

skip_estimation_sigma_a = False
skip_estimation_R = False
skip_estimation_m0 = False
skip_estimation_V0 = False
data_filename = "../../data/positions_session003_start0.00_end15548.27.csv"

start_position = 0
number_positions = 5000
pos_x0 = 0.0
pos_y0 = 0.0
vel_x0 = 0.0
vel_y0 = 0.0
ace_x0 = 0.0
ace_y0 = 0.0
sigma_a0 = 1.0
sigma_x0 = 0.1
sigma_y0 = 0.1
sqrt_diag_V0_value = 0.1
lbfgs_max_iter = 2
lbfgs_tolerance_grad = -1
lbfgs_tolerance_change = 1e-3
lbfgs_lr = 1.0
lbfgs_n_epochs = int(estMeta["optim_params"]["lbfgs_n_epochs"])
lbfgs_tol = float(estMeta["optim_params"]["lbfgs_tol"])

data = pd.read_csv(filepath_or_buffer=data_filename)
data = data.iloc[start_position:start_position+number_positions,:]
y = np.transpose(data[["x", "y"]].to_numpy())
date_times = pd.to_datetime(data["time"])
dt = (date_times.iloc[1]-date_times.iloc[0]).total_seconds()

if math.isnan(pos_x0):
    pos_x0 = y[0, 0]
if math.isnan(pos_y0):
    pos_y0 = y[1, 0]

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
Qe = np.array([[dt**5/20, dt**4/8, dt**3/6, 0, 0, 0],
               [dt**4/8, dt**3/3,  dt**2/2, 0, 0, 0],
               [dt**3/6, dt**2/2,  dt,      0, 0, 0],
               [0, 0, 0,                    dt**5/20, dt**4/8, dt**3/6],
               [0, 0, 0,                    dt**4/8, dt**3/3,  dt**2/2],
               [0, 0, 0,                    dt**3/6, dt**2/2,  dt]],
              dtype=np.double)
m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
              dtype=np.double)

vars_to_estimate = {}
if skip_estimation_sigma_a:
    vars_to_estimate["sigma_a"] = False
else:
    vars_to_estimate["sigma_a"] = True

if skip_estimation_R:
    vars_to_estimate["sqrt_diag_R"] = False
else:
    vars_to_estimate["sqrt_diag_R"] = True

if skip_estimation_m0:
    vars_to_estimate["m0"] = False
else:
    vars_to_estimate["m0"] = True

if skip_estimation_V0:
    vars_to_estimate["sqrt_diag_V0"] = False
else:
    vars_to_estimate["sqrt_diag_V0"] = True

sqrt_diag_R_torch = torch.DoubleTensor([sigma_x0, sigma_y0])
m0_torch = torch.from_numpy(m0)
sqrt_diag_V0_torch = torch.DoubleTensor([sqrt_diag_V0_value
                                         for i in range(len(m0))])
y_torch = torch.from_numpy(y.astype(np.double))
B_torch = torch.from_numpy(B.astype(np.double))
Qe_torch = torch.from_numpy(Qe.astype(np.double))
Z_torch = torch.from_numpy(Z.astype(np.double))

optim_res = lds_python.learning.torch_lbfgs_optimize_SS_tracking_diagV0(
    y=y_torch, B=B_torch, sigma_a0=sigma_a0,
    Qe=Qe_torch, Z=Z_torch, sqrt_diag_R_0=sqrt_diag_R_torch, m0_0=m0_torch,
    sqrt_diag_V0_0=sqrt_diag_V0_torch, max_iter=max_iter, lr=lr,
    vars_to_estimate=vars_to_estimate, tolerance_grad=tolerance_grad,
    tolerance_change=tolerance_change, n_epochs=n_epochs, tol=tol)

print(optim_res["termination_info"])

# save results
est_prefix_used = True
while est_prefix_used:
    estRes_number = random.randint(0, 10**8)
    estRes_metadata_filename = \
        estRes_metadata_filename_pattern.format(estRes_number)
    if not os.path.exists(estRes_metadata_filename):
        est_prefix_used = False
estRes_data_filename = estRes_data_filename_pattern.format(estRes_number)

estimRes_metadata = configparser.ConfigParser()
estimRes_metadata["data_params"] = {"data_filename": data_filename}
estimRes_metadata["estimation_params"] = {"estInitNumber": estMeta_number}
with open(estRes_metadata_filename, "w") as f:
    estimRes_metadata.write(f)

with open(estRes_data_filename, "wb") as f:
    pickle.dump(optim_res, f)
print("Saved results to {:s}".format(estRes_data_filename))

breakpoint()

if __name__ == "__main__":
    main(sys.argv)
