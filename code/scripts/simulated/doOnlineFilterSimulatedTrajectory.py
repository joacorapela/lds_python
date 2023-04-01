import sys
import os.path
import random
import argparse
import configparser
import numpy as np
import pandas as pd

import utils
sys.path.append("../src")
import inference

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", type=int,
                        help="simulation result number")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--filtering_params_filename", type=str,
                        default="", help="filtering parameters filename")
    parser.add_argument("--filtering_params_section", type=str,
                        default="initial_params",
                        help="section of ini file containing the filtering params")
    parser.add_argument("--results_filenames_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}", 
                        help="results filename pattern")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    simRes_filename_pattern = args.simRes_filename_pattern
    filtering_params_filename = args.filtering_params_filename
    filtering_params_section = args.filtering_params_section
    results_filenames_pattern = args.results_filenames_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number, "npz")
    simRes = np.load(simRes_filename)

    if len(filtering_params_filename) > 0:
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
        sqrt_diag_V0_value = float(filtering_params[filtering_params_section]
                              ["sqrt_diag_V0_value"])

        m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                      dtype=np.double)
        V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
        R = np.diag([sigma_x**2, sigma_y**2])
    else:
        # sigma_ax = simRes["sigma_ax"]
        # sigma_ay = simRes["sigma_ay"]
        sigma_ax = simRes["sigma_a"]
        sigma_ay = simRes["sigma_a"]
        m0 = simRes["m0"]
        V0 = simRes["V0"]
        R = simRes["R"]

    # save filtering metadata
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filenames_pattern.format(
            res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False

    filtered_metadata = configparser.ConfigParser()
    filtered_metadata["params"] = {"simRes_number": simRes_number,
                                   "filtering_params_filename":
                                   filtering_params_filename,
                                   "filtering_params_section":
                                   filtering_params_section}
    with open(filtered_metadata_filename, "w") as f:
        filtered_metadata.write(f)

    # do online filtering
    Q = utils.buildQfromQt_np(Qt=simRes["Qt"], sigma_ax=sigma_ax, sigma_ay=sigma_ay)
    onlineKF = inference.OnlineKalmanFilter(B=simRes["B"], Q=Q, m0=m0, V0=V0,
                                            Z=simRes["Z"], R=R)
    ys = simRes["y"]
    filtered_means = np.empty((6, 1, ys.shape[1]), dtype=np.double)
    filtered_covs = np.empty((6, 6, ys.shape[1]), dtype=np.double)
    for i in range(ys.shape[1]):
        _, _ = onlineKF.predict()
        filtered_means[:, 0, i], filtered_covs[:, :, i] = \
            onlineKF.update(y=ys[:, i])

    # save results
    results_filename = results_filenames_pattern.format(
        res_number, "csv")

    results = {"filtered_means": filtered_means,
               "filtered_covs": filtered_covs}
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)

    print("Saved results to {:s}".format(results_filename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
