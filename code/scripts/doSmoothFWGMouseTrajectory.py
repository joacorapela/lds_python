import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append("../src")
import lds_functions

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_position", type=int, default=0,
                        help="start position to smooth")
    parser.add_argument("--number_positions", type=int, default=10000,
                        help="number of positions to smooth")
    parser.add_argument("----sigma_a", type=float, default=0.1,
                        help="acceleration standard deviation")
    parser.add_argument("----sigma_x", type=float, default=0.001,
                        help="x-position standard deviation")
    parser.add_argument("--sigma_y", type=float, default=0.001,
                        help="y-position standard deviation")
    parser.add_argument("--V0_diag_value0", type=float, default=0.001,
                        help="value for diagonal of the initial state covariance matrix")
    parser.add_argument("--data_filename", type=str,
                        default="../../data/postions_session003_start0.00_end15548.27.csv",
                        help="inputs positions filename")
    parser.add_argument("--smoothed_data_filename_pattern", type=str,
                        default="../../results/postions_smoothed_session003_start0.00_end15548.27_startPosition{:d}_numPosition{:d}.csv",
                        help="smoothed data filename pattern")
    args = parser.parse_args()

    start_position = args.start_position
    number_positions = args.number_positions
    sigma_a = args.sigma_a
    sigma_x = args.sigma_x
    sigma_y = args.sigma_y
    V0_diag_value0 = args.V0_diag_value0
    data_filename = args.data_filename
    smoothed_data_filename_pattern = args.smoothed_data_filename_pattern

    smoothed_data_filename = args.smoothed_data_filename_pattern.format(
        start_position, number_positions)
    data = pd.read_csv(filepath_or_buffer=data_filename)
    data = data.iloc[start_position:start_position+number_positions,:]
    y = np.transpose(data[["x", "y"]].to_numpy())
    date_times = pd.to_datetime(data["time"])
    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.3.3
    dt = (date_times[1]-date_times[0]).total_seconds()
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
    Q = np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)*sigma_a**2
    R = np.diag([sigma_x**2, sigma_y**2])
    m0 = np.array([y[0, 0], 0, 0, y[1, 0], 0, 0], dtype=np.double)
    V0 = np.diag(np.ones(len(m0))*V0_diag_value0)

    filterRes = lds_functions.filterLDS_SS_withMissingValues(y=y, B=B, Q=Q,
                                                             m0=m0, V0=V0, Z=Z,
                                                             R=R)
    smoothRes = lds_functions.smoothLDS_SS(B=B,
                                           xnn=filterRes["xnn"],
                                           Vnn=filterRes["Vnn"],
                                           xnn1=filterRes["xnn1"],
                                           Vnn1=filterRes["Vnn1"],
                                           m0=m0, V0=V0)
    data={"time": data["time"], "pos1": y[0,:], "pos2": y[1,:],
          "fpos1": filterRes["xnn"][0,0,:], "fpos2": filterRes["xnn"][3,0,:],
          "fvel1": filterRes["xnn"][1,0,:], "fvel2": filterRes["xnn"][4,0,:],
          "facc1": filterRes["xnn"][2,0,:], "facc2": filterRes["xnn"][5,0,:],
          "spos1": smoothRes["xnN"][0,0,:], "spos2": smoothRes["xnN"][3,0,:],
          "svel1": smoothRes["xnN"][1,0,:], "svel2": smoothRes["xnN"][4,0,:],
          "sacc1": smoothRes["xnN"][2,0,:], "sacc2": smoothRes["xnN"][5,0,:]}
    df = pd.DataFrame(data=data)
    df.to_csv(smoothed_data_filename)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
