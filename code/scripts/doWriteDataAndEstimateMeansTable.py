import sys
import argparse
import pickle
import pandas as pd

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estMeta_number", help="estimation metadata number",
                        type=int)
    parser.add_argument("--smoothRes_data_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothed.pickle",
                        help="estimation results data filename pattern")
    parser.add_argument("--smoothRes_means_data_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothed_means.csv",
                        help="estimation results mean data filename_pattern")
    args = parser.parse_args()

    estMeta_number = args.estMeta_number
    smoothRes_data_filename_pattern = args.smoothRes_data_filename_pattern
    smoothRes_means_data_filename_pattern = args.smoothRes_means_data_filename_pattern

    smoothRes_data_filename = \
        args.smoothRes_data_filename_pattern.format(estMeta_number)
    smoothRes_means_data_filename = \
        smoothRes_means_data_filename_pattern.format(estMeta_number)

    with open(smoothRes_data_filename, "rb") as f:
        smoothRes_data = pickle.load(f)

    data={"time":  smoothRes_data["time"],
          "pos1":  smoothRes_data["measurements"][0,:],
          "pos2":  smoothRes_data["measurements"][1,:],
          "fpos1": smoothRes_data["filter_res"]["xnn"][0,0,:],
          "fpos2": smoothRes_data["filter_res"]["xnn"][3,0,:],
          "fvel1": smoothRes_data["filter_res"]["xnn"][1,0,:],
          "fvel2": smoothRes_data["filter_res"]["xnn"][4,0,:],
          "facc1": smoothRes_data["filter_res"]["xnn"][2,0,:],
          "facc2": smoothRes_data["filter_res"]["xnn"][5,0,:],
          "spos1": smoothRes_data["smooth_res"]["xnN"][0,0,:],
          "spos2": smoothRes_data["smooth_res"]["xnN"][3,0,:],
          "svel1": smoothRes_data["smooth_res"]["xnN"][1,0,:],
          "svel2": smoothRes_data["smooth_res"]["xnN"][4,0,:],
          "sacc1": smoothRes_data["smooth_res"]["xnN"][2,0,:],
          "sacc2": smoothRes_data["smooth_res"]["xnN"][5,0,:]}
    df = pd.DataFrame(data=data)
    df.to_csv(smoothRes_means_data_filename)

if __name__ == "__main__":
    main(sys.argv)
