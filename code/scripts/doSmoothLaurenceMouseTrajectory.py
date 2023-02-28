import sys
import argparse
import configparser
import pickle
import numpy as np
import pandas as pd

import utils
sys.path.append("../src")
import inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtering_params_filename", type=str,
                        help="filtering parameters filename",
                        default="../../metadata/00000009_smoothing.ini")
    parser.add_argument("--bodypart", type=str, help="body part to track",
                        default="upper_back")
    parser.add_argument("--min_likelihood", type=float,
                        help=("positions with likelihood less than "
                              "min_likelihood are set to np.nan"),
                        default=0.99)
    parser.add_argument("--first_sample", type=int,
                        help="start position to smooth", default=0)
    parser.add_argument("--number_samples", type=int,
                        help="number of samples to smooth", default=10000)
    parser.add_argument("--fs", type=float, help="sampling frequency", default=40)
    parser.add_argument("--filtering_params_section", type=str,
                        help=("section of ini file containing the filtering "
                              "params"), default="params")
    parser.add_argument("--data_filename", type=str,
                        help="inputs positions filename",
                        default="../../data/cam1DLC_resnet50_NPX_7May20shuffle1_500000.h5")
    parser.add_argument("--smoothed_results_filename_pattern", type=str,
                        help="smoothed data filename pattern",
                        default="../../results/smoothed_results_laurence_firstSample{:d}_numberOfSamples{:d}.pickle")
    args = parser.parse_args()

    filtering_params_filename = args.filtering_params_filename
    bodypart = args.bodypart
    min_likelihood = args.min_likelihood
    first_sample = args.first_sample
    number_samples = args.number_samples
    fs = args.fs
    filtering_params_section = args.filtering_params_section
    data_filename = args.data_filename
    smoothed_results_filename_pattern = args.smoothed_results_filename_pattern

    smoothed_results_filename = args.smoothed_results_filename_pattern.format(
        first_sample, number_samples)

    df = pd.read_hdf(data_filename)
    level0_values =  df.columns.levels[0]
    if len(level0_values) >1:
        raise ValueError(f"More than one level0 values in {data_filename}")
    data = df[level0_values[0]][bodypart]
    low_like_indices = data["likelihood"] < min_likelihood
    data.loc[low_like_indices, "x"] = np.nan
    data.loc[low_like_indices, "y"] = np.nan

    y = np.row_stack((data["x"], data["y"]))
    y = y[:, first_sample:(first_sample+number_samples)]

    dt = 1.0 / fs

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
    if np.isnan(pos_x0):
        pos_x0 = y[0, 0]
    if np.isnan(pos_y0):
        pos_y0 = y[0, 1]

    m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                  dtype=np.double)
    V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
    R = np.diag([sigma_x**2, sigma_y**2])

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
    time = np.arange(first_sample*dt, (first_sample+number_samples)*dt, dt)
    results = {"time": time, "fs": fs, "pos": y, "filter_res": filter_res, "smooth_res": smooth_res}
    with open(smoothed_results_filename, "wb") as f:
        pickle.dump(results, f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
