import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np

sys.path.append("../../src")
import learning


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", help="simulation result number",
                        type=int)
    parser.add_argument("estMeta_number", help="estimation metadata number",
                        type=int)
    parser.add_argument("--simRes_filename_pattern",
                        help="simulation result filename pattern",
                        default= "../../../results/{:08d}_simulation.npz")
    parser.add_argument("--estInit_metadata_filename_pattern", type=str,
                        default="../../../metadata/{:08d}_estimation.ini",
                        help="estimation initialization metadata filename pattern")
    parser.add_argument("--estRes_metadata_filename_pattern", type=str,
                        default="../../../results/{:08d}_estimation.ini",
                        help="estimation results metadata filename pattern")
    parser.add_argument("--estRes_data_filename_pattern", type=str,
                        default="../../../results/{:08d}_estimation.pickle",
                        help="estimation results data filename pattern")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    estMeta_number = args.estMeta_number
    simRes_filename_pattern = args.simRes_filename_pattern
    estInit_metadata_filename_pattern = args.estInit_metadata_filename_pattern
    estRes_metadata_filename_pattern = args.estRes_metadata_filename_pattern
    estRes_data_filename_pattern = args.estRes_data_filename_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number)
    simRes = np.load(simRes_filename)
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
    noise_intensity0 = float(estMeta["initial_params"]["noise_intensity"])
    sigma_x0 = float(estMeta["initial_params"]["sigma_x"])
    sigma_y0 = float(estMeta["initial_params"]["sigma_y"])
    sqrt_diag_V0_value = float(estMeta["initial_params"]["sqrt_diag_v0_value"])
    em_max_iter = int(estMeta["optim_params"]["em_max_iter"])

    if math.isnan(pos_x0):
        pos_x0 = simRes["y"][0, 0]
    if math.isnan(pos_y0):
        pos_y0 = simRes["y"][1, 0]

    # R_0 = np.diag([sigma_x0**2, sigma_y0**2])
    sqrt_diag_R_0 = np.array([sigma_x0, sigma_y0])
    R_0 = np.diag(sqrt_diag_R_0**2)
    m0_0 = np.array([[pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0]],
                    dtype=np.double).T
    # m0_0.shape = (len(m0_0), 1)
    # V0_0 = np.diag([V0_diag_value for i in range(len(m0_0))])
    sqrt_diag_V0_0 = np.array([sqrt_diag_V0_value for i in range(len(m0_0))])
    V0_0 = np.diag(sqrt_diag_V0_0**2)

    # run EM
    # vars_to_estimate = {"noise_intensity": True, "R": False, "m0": False, "V0": False}
    vars_to_estimate = {"noise_intensity": True, "R": False, "m0": False, "V0": False}
    R, m0, V0, noise_intensity = learning.em_SS_tracking_CWPA(y=simRes["y"], B=simRes["B"], noise_intensity0=noise_intensity0, Qe=simRes["Qe"], Z=simRes["Z"], R_0=R_0, m0_0=m0_0, V0_0=V0_0, vars_to_estimate=vars_to_estimate, max_iter=em_max_iter)
    # R, m0, V0, noise_intensity = learning.optim_SS_tracking__fullV0(y=simRes["y"], B=simRes["B"], noise_intensity0=noise_intensity0, Qt=simRes["Qt"], Z=simRes["Z"], diag_R_0=diag_R_0, m0_0=m0_0, V0_0=V0_0, max_iter=em_max_iter)
    # optim_res = learning.optim_SS_tracking__diagV0(y=simRes["y"], B=simRes["B"], noise_intensityx0=noise_intensityx0, noise_intensityy0=noise_intensityy0, Qt=simRes["Qt"], Z=simRes["Z"], sqrt_diag_R_0=sqrt_diag_R_0, m0_0=m0_0, sqrt_diag_V0_0=sqrt_diag_V0_0, max_iter=em_max_iter)

    # save results
    est_prefix_used = True
    while est_prefix_used:
        estRes_number = random.randint(0, 10**8)
        estRes_metadata_filename = \
            estRes_metadata_filename_pattern.format(estRes_number)
        if not os.path.exists(estRes_metadata_filename):
            est_prefix_used = False
    estRes_data_filename = estRes_data_filename_pattern.format(estRes_number)
#     estRes_number = 0
#     estRes_metadata_filename = estRes_metadata_filename_pattern.format(estRes_number)
#     estRes_data_filename = estRes_data_filename_pattern.format(estRes_number)

    estimRes_metadata = configparser.ConfigParser()
    estimRes_metadata["simulation_params"] = {"simResNumber": simRes_number}
    estimRes_metadata["estimation_params"] = {"estInitNumber": estMeta_number}
    with open(estRes_metadata_filename, "w") as f:
        estimRes_metadata.write(f)

    with open(estRes_data_filename, "wb") as f:
        pickle.dump(optim_res, f)
    # np.savez(estRes_data_filename, sqrt_diag_R=sqrt_diag_R, m0=m0, sqrt_diag_V0=sqrt_diag_V0, noise_intensity=noise_intensity)
    print("Saved results to {:s}".format(estRes_data_filename))
    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
