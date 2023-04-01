import sys
import pickle
import numpy as np
import argparse
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_bodypart", type=str,
                        help="bodypart used as source vector point",
                        default="tailbase")
    parser.add_argument("--dst_bodypart", type=str,
                        help="bodypart used as destination vector point",
                        default="snout")
    parser.add_argument("--first_sample", type=int,
                        help="start position to smooth", default=0)
    parser.add_argument("--number_samples", type=int,
                        help="number of samples to smooth", default=10000)
    parser.add_argument("--smoothed_results_filename_pattern", type=str,
                        help="smoothed data filename pattern",
                        default=("../../../results/commutator/"
                                 "smoothed_results_bodypart_{:s}_firstSample{:d}_"
                                 "numberOfSamples{:d}.pickle"))
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("../../../figures/commutator/"
                                 "smoothed_angles_srcBodypart_{:s}_dstBodypart_{:s}_firstSample{:d}_"
                                 "numberOfSamples{:d}.{:s}"))

    args = parser.parse_args()

    src_bodypart = args.src_bodypart
    dst_bodypart = args.dst_bodypart
    first_sample = args.first_sample
    number_samples = args.number_samples
    smoothed_results_filename_pattern = args.smoothed_results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    src_smoothed_results_filename = smoothed_results_filename_pattern.format(
        src_bodypart, first_sample, number_samples)
    dst_smoothed_results_filename = smoothed_results_filename_pattern.format(
        dst_bodypart, first_sample, number_samples)
    with open(src_smoothed_results_filename, "rb") as f:
        load_res = pickle.load(f)
    time = load_res["time"]
    smooth_res = load_res["smooth_res"]
    src_x = smooth_res["xnN"][0,0,:]
    src_y = smooth_res["xnN"][3,0,:]
    with open(dst_smoothed_results_filename, "rb") as f:
        load_res = pickle.load(f)
    smooth_res = load_res["smooth_res"]
    dst_x = smooth_res["xnN"][0,0,:]
    dst_y = smooth_res["xnN"][3,0,:]

    resultant_x = dst_x - src_x
    resultant_y = dst_y - src_y
    angle = np.arctan(resultant_y/resultant_x)

    fig = go.Figure()
    trace = go.Scatter(x=time, y=angle)
    fig.add_trace(trace)
    fig.update_layout(
        title=(f"source bodypart: {src_bodypart}, "
               f"destination bodypart: {dst_bodypart}"),
        xaxis_title="time (sec)", yaxis_title="angle (radians)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig.write_image(fig_filename_pattern.format(src_bodypart, dst_bodypart,
                                                first_sample, number_samples,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(src_bodypart, dst_bodypart,
                                               first_sample, number_samples,
                                               "html"))

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
