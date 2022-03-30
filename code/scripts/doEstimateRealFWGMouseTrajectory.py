import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np
import scipy.interpolate
import pandas as pd

sys.path.append("../src")
import learning


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estMeta_number", help="estimation metadata number",
                        type=int)
    parser.add_argument("--start_position", type=int, default=0,
                        help="start position to smooth")
    parser.add_argument("--number_positions", type=int, default=10000,
                        help="number of positions to smooth")
    parser.add_argument("--estInit_metadata_filename_pattern", type=str,
                        default="../../metadata/{:08d}_estimation.ini",
                        help="estimation initialization metadata filename pattern")
    parser.add_argument("--data_filename", type=str,
                        default="../../data/postions_session003_start0.00_end15548.27.csv",
                        help="inputs positions filename")
    parser.add_argument("--estRes_metadata_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.ini",
                        help="estimation results metadata filename pattern")
    parser.add_argument("--estRes_data_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.{:s}",
                        help="estimation results data filename pattern")
    args = parser.parse_args()

    estMeta_number = args.estMeta_number
    start_position = args.start_position
    number_positions = args.number_positions
    estInit_metadata_filename_pattern = args.estInit_metadata_filename_pattern
    data_filename = args.data_filename
    estRes_metadata_filename_pattern = args.estRes_metadata_filename_pattern
    estRes_data_filename_pattern = args.estRes_data_filename_pattern

    estInit_metadata_filename = \
        estInit_metadata_filename_pattern.format(estMeta_number)

    estMeta = configparser.ConfigParser()
    estMeta.read(estInit_metadata_filename)
    pos_x0 = float(estMeta["initial_params"]["pos_x0"])
    pos_y0 = float(estMeta["initial_params"]["pos_y0"])
    vel_x0 = float(estMeta["initial_params"]["vel_x0"])
    vel_y0 = float(estMeta["initial_params"]["vel_y0"])
    ace_x0 = float(estMeta["initial_params"]["ace_x0"])
    ace_y0 = float(estMeta["initial_params"]["ace_y0"])
    sigma_ax0 = float(estMeta["initial_params"]["sigma_ax"])
    sigma_ay0 = float(estMeta["initial_params"]["sigma_ay"])
    sigma_x0 = float(estMeta["initial_params"]["sigma_x"])
    sigma_y0 = float(estMeta["initial_params"]["sigma_y"])
    sqrt_diag_V0_value = float(estMeta["initial_params"]["sqrt_diag_v0_value"])
    em_max_iter = int(estMeta["optim_params"]["em_max_iter"])

    data = pd.read_csv(filepath_or_buffer=data_filename)
    data = data.iloc[start_position:start_position+number_positions,:]
    y = np.transpose(data[["x", "y"]].to_numpy())
    date_times = pd.to_datetime(data["time"])
    dt = (date_times.iloc[1]-date_times.iloc[0]).total_seconds()

    times = np.arange(0, y.shape[1]*dt, dt)
    not_nan_indices_y0 = set(np.where(np.logical_not(np.isnan(y[0, :])))[0])
    not_nan_indices_y1 = set(np.where(np.logical_not(np.isnan(y[1, :])))[0])
    not_nan_indices = np.array(sorted(not_nan_indices_y0.union(not_nan_indices_y1)))
    y_no_nan = y[:, not_nan_indices]
    t_no_nan = times[not_nan_indices]
    y_interpolated = np.empty_like(y)
    tck, u = scipy.interpolate.splprep([y_no_nan[0, :], y_no_nan[1, :]], s=0, u=t_no_nan)
    y_interpolated[0, :], y_interpolated[1, :] = scipy.interpolate.splev(times, tck)
    y = y_interpolated

    if math.isnan(pos_x0):
        pos_x0 = y[0, 0]
    if math.isnan(pos_y0):
        pos_y0 = y[1, 0]

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

    sqrt_diag_R_0 = np.array([sigma_x0, sigma_y0])
    m0_0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
                    dtype=np.double)
    sqrt_diag_V0_0 = np.array([sqrt_diag_V0_value for i in range(len(m0_0))])

    # run EM
    # vars_to_estimate = {"sigma_a": True, "R": False, "m0": False, "V0": False}
    # R, m0, V0, sigma_a = learning.em_SS_tracking_DWPA(y=simRes["y"], B=simRes["B"], sigma_a0=sigma_a0, Qt=simRes["Qt"], Z=simRes["Z"], R_0=R_0, m0_0=m0_0, V0_0=V0_0, vars_to_estimate=vars_to_estimate, max_iter=em_max_iter)
    # R, m0, V0, sigma_a = learning.optim_SS_tracking_DWPA_fullV0(y=simRes["y"], B=simRes["B"], sigma_a0=sigma_a0, Qt=simRes["Qt"], Z=simRes["Z"], diag_R_0=diag_R_0, m0_0=m0_0, V0_0=V0_0, max_iter=em_max_iter)
    optim_res = learning.optim_SS_tracking_DWPA_diagV0(
        y=y, B=B, sigma_ax0=sigma_ax0, sigma_ay0=sigma_ay0, Qt=Qt,
        Z=Z, sqrt_diag_R_0=sqrt_diag_R_0, m0_0=m0_0,
        sqrt_diag_V0_0=sqrt_diag_V0_0, max_iter=em_max_iter)

    # save results
    est_prefix_used = True
    while est_prefix_used:
        estRes_number = random.randint(0, 10**8)
        estRes_metadata_filename = estRes_metadata_filename_pattern.format(estRes_number)
        if not os.path.exists(estRes_metadata_filename):
            est_prefix_used = False
    estRes_data_filename_pickle = estRes_data_filename_pattern.format(estRes_number, "pickle")
    estRes_data_filename_ini = estRes_data_filename_pattern.format(estRes_number, "ini")
#     estRes_number = 0
#     estRes_metadata_filename = estRes_metadata_filename_pattern.format(estRes_number)
#     estRes_data_filename = estRes_data_filename_pattern.format(estRes_number)

    estimRes_metadata = configparser.ConfigParser()
    estimRes_metadata["data_params"] = {"data_filename": data_filename}
    estimRes_metadata["estimation_params"] = {"estInitNumber": estMeta_number}
    with open(estRes_metadata_filename, "w") as f: estimRes_metadata.write(f)

    estimRes_data = configparser.ConfigParser()
    m0_str = "[" + ",".join([str(item) for item in optim_res["x"]["m0"].tolist()]) + "]"
    sqrt_diag_R_str = "[" + ",".join([str(item) for item in optim_res["x"]["sqrt_diag_R"].tolist()]) + "]"
    sqrt_diag_V0_str = "[" + ",".join([str(item) for item in optim_res["x"]["sqrt_diag_V0"].tolist()]) + "]"
    # estimRes_data["results"] = {"sigma_a": optim_res["x"]["sigma_a"].item(), "m0": optim_res["x"]["m0"], "sqrt_diag_R": optim_res["x"]["sqrt_diag_R"], "sqrt_diag_V0": optim_res["x"]["sqrt_diag_V0"]}
    estimRes_data["results"] = {"sigma_ax": optim_res["x"]["sigma_ax"].item(), "sigma_ay": optim_res["x"]["sigma_ay"].item(), "m0": m0_str, "sqrt_diag_R": sqrt_diag_R_str, "sqrt_diag_V0": sqrt_diag_V0_str}
    with open(estRes_data_filename_ini, "w") as f: estimRes_data.write(f)

    with open(estRes_data_filename_pickle, "wb") as f: pickle.dump(optim_res, f)
    # np.savez(estRes_data_filename, sqrt_diag_R=sqrt_diag_R, m0=m0, sqrt_diag_V0=sqrt_diag_V0, sigma_a=sigma_a)
    print("Saved results to {:s}".format(estRes_data_filename_pickle))
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
