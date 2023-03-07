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
                        default="../../results/{:08d}_smoothed.{:s}", 
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
        sqrt_diag_V0_value = float(smoothing_params[filtering_params_section]
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

    Q = utils.buildQfromQt_np(Qt=simRes["Qt"], sigma_ax=sigma_ax, sigma_ay=sigma_ay)
    filterRes = inference.filterLDS_SS_withMissingValues_np(
        y=simRes["y"], B=simRes["B"], Q=Q, m0=m0, V0=V0, Z=simRes["Z"], R=R)
    smoothRes = inference.smoothLDS_SS(B=simRes["B"],
                                       xnn=filterRes["xnn"],
                                       Vnn=filterRes["Vnn"],
                                       xnn1=filterRes["xnn1"],
                                       Vnn1=filterRes["Vnn1"],
                                       m0=m0, V0=V0)
    data = {"fpos1": filterRes["xnn"][0, 0, :],
            "fpos2": filterRes["xnn"][3, 0, :],
            "fvel1": filterRes["xnn"][1, 0, :],
            "fvel2": filterRes["xnn"][4, 0, :],
            "facc1": filterRes["xnn"][2, 0, :],
            "facc2": filterRes["xnn"][5, 0, :],
            "spos1": smoothRes["xnN"][0, 0, :],
            "spos2": smoothRes["xnN"][3, 0, :],
            "svel1": smoothRes["xnN"][1, 0, :],
            "svel2": smoothRes["xnN"][4, 0, :],
            "sacc1": smoothRes["xnN"][2, 0, :],
            "sacc2": smoothRes["xnN"][5, 0, :]}
    df = pd.DataFrame(data=data)

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        smoothed_metadata_filename = results_filenames_pattern.format(
            res_number, "ini")
        if not os.path.exists(smoothed_metadata_filename):
            res_prefix_used = False
    smoothed_data_filename = results_filenames_pattern.format(
        res_number, "csv")

    df.to_csv(smoothed_data_filename)
    print("Saved results to {:s}".format(smoothed_data_filename))
    smoothed_metadata = configparser.ConfigParser()
    smoothed_metadata["params"] = {"simRes_number": simRes_number,
                                   "filtering_params_filename":
                                   filtering_params_filename,
                                   "filtering_params_section":
                                   filtering_params_section}
    with open(smoothed_metadata_filename, "w") as f:
        smoothed_metadata.write(f)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
