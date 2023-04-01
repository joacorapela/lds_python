import sys
import numpy as np
import pickle
import pandas as pd
import datetime
import argparse
import configparser
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("K", type=int, help="number of states in HMM model")
    parser.add_argument("--hmm_result_filename_pattern", type=str,
                        default="../../results/87174971_smoothed_means_withStates_K{:d}.csv",
                        help="prefix of hmm results filename")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/87174971_smoothed_means_withStates_K{:d}.{:s}",
                        help="figure filename pattern")

    args = parser.parse_args()

    K = args.K
    hmm_result_filename_pattern = args.hmm_result_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    hmm_result_filename = hmm_result_filename_pattern.format(K)
    hmm_res = pd.read_csv(hmm_result_filename)

    state_posterior_columns = [hmm_res["state{:d}".format(k)]
                               for k in range(1, K+1)]
    state_posteriors = np.column_stack(state_posterior_columns)
    most_probable_state = np.argmax(state_posteriors, axis=1)

    hmm_res["most_probable_state"] = most_probable_state
    fig = go.Figure()
    trace = go.Scatter(x=hmm_res["spos1"], y=hmm_res["spos2"],
                       mode="lines+markers",
                       marker=dict(color=most_probable_state))
    fig.add_trace(trace)

    fig.write_image(fig_filename_pattern.format(K, "png"))
    fig.write_html(fig_filename_pattern.format(K, "html"))
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
