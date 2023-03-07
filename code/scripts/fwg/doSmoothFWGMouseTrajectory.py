import sys
import os
import random
import pickle
import math
import argparse
import configparser
import numpy as np
import pandas as pd

import utils
sys.path.append("../src")
import inference

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filtering_params_filename", type=str,
                        default="", help="filtering parameters filename")
    parser.add_argument("--start_position", type=int, default=0,
                        help="start position to smooth")
    parser.add_argument("--number_positions", type=int, default=10000,
                        help="number of positions to smooth")
    parser.add_argument("--filtering_params_section", type=str,
                        default="params",
                        help="section of ini file containing the filtering params")
    parser.add_argument("--data_filename", type=str,
                        default="../../data/positions_session003_start0.00_end15548.27.csv",
                        help="inputs positions filename")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothed.{:s}")
    args = parser.parse_args()

    start_position = args.start_position
    number_positions = args.number_positions
    filtering_params_filename = args.filtering_params_filename
    filtering_params_section = args.filtering_params_section
    data_filename = args.data_filename
    results_filename_pattern = args.results_filename_pattern

    data = pd.read_csv(filepath_or_buffer=data_filename)
    data = data.iloc[start_position:start_position+number_positions,:]
    y = np.transpose(data[["x", "y"]].to_numpy())

    smoothing_params = configparser.ConfigParser()
    smoothing_params.read(filtering_params_filename)
    pos_x0 = float(smoothing_params[filtering_params_section]["pos_x0"])
    pos_y0 = float(smoothing_params[filtering_params_section]["pos_y0"])
    vel_x0 = float(smoothing_params[filtering_params_section]["vel_x0"])
    vel_y0 = float(smoothing_params[filtering_params_section]["vel_x0"])
    acc_x0 = float(smoothing_params[filtering_params_section]["acc_x0"])
    acc_y0 = float(smoothing_params[filtering_params_section]["acc_x0"])
    sigma_ax = float(smoothing_params[filtering_params_section]["sigma_ax"])
    sigma_ay = float(smoothing_params[filtering_params_section]["sigma_ay"])
    sigma_x = float(smoothing_params[filtering_params_section]["sigma_x"])
    sigma_y = float(smoothing_params[filtering_params_section]["sigma_y"])
    sqrt_diag_V0_value = float(smoothing_params[filtering_params_section]["sqrt_diag_V0_value"])

    if math.isnan(pos_x0):
        pos_x0 = y[0, 0]
    if math.isnan(pos_y0):
        pos_y0 = y[1, 0]

    m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                  dtype=np.double)
    V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
    R = np.diag([sigma_x**2, sigma_y**2])

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
    m0.shape = (len(m0), 1)
    V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
    Q = utils.buildQfromQt_np(Qt=Qt, sigma_ax=sigma_ax, sigma_ay=sigma_ay)

    filter_res = inference.filterLDS_SS_withMissingValues_np(y=y, B=B, Q=Q,
                                                             m0=m0, V0=V0, Z=Z,
                                                             R=R)
    smooth_res = inference.smoothLDS_SS(B=B, xnn=filter_res["xnn"],
                                       Vnn=filter_res["Vnn"],
                                       xnn1=filter_res["xnn1"],
                                       Vnn1=filter_res["Vnn1"],
                                       m0=m0, V0=V0)
    results = {"time": data["time"],
               "measurements": y,
               "filter_res": filter_res,
               "smooth_res": smooth_res,
              }

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved smoothing results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "data_filename": data_filename,
        "start_position": start_position,
        "number_positions": number_positions,
        "filtering_params_filename": filtering_params_filename,
        "filtering_params_section": filtering_params_section,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
