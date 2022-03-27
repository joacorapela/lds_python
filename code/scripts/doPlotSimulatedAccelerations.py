import sys
import pdb
import numpy as np
import pandas as pd
import argparse
import configparser
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", type=int,
                        help="simulation result number")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_simulation_accelerations.{:s}")
    parser.add_argument("--xlabel", help="xlabel", default="Sample")
    parser.add_argument("--ylabel", help="ylabel", default="Acceleration")

    args = parser.parse_args()

    simRes_number = args.simRes_number
    simRes_filename_pattern = args.simRes_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern
    xlabel = args.xlabel
    ylabel = args.ylabel

    simRes_filename = simRes_filename_pattern.format(simRes_number, "npz")
    simRes = np.load(simRes_filename)

    N = simRes["y"].shape[1]
    time = np.arange(0, N*simRes["dt"], simRes["dt"])
    fig = go.Figure()
    trace_acc1 = go.Scatter(x=time, y=simRes["x"][2, :],
                            mode="lines+markers",
                            name="acceleration 1",
                            showlegend=True,
                            )
    trace_acc2 = go.Scatter(x=time, y=simRes["x"][5, :],
                            mode="lines+markers",
                            name="acceleration 2",
                            showlegend=True,
                            )
    fig.add_trace(trace_acc1)
    fig.add_trace(trace_acc2)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.write_image(fig_filename_pattern.format(simRes_number, "png"))
    fig.write_html(fig_filename_pattern.format(simRes_number, "html"))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
