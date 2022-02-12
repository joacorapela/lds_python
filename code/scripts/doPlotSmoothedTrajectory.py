import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import argparse
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoothed_data_filename",
                        help="smoothed data filename",
                        default="../../results/postions_smoothed_session003_start0.00_end15548.27_startPosition0_numPosition10000.csv")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/postions_smoothed_session003_start0.00_end15548.27_startPosition1_numPosition10000.{:s}")
    parser.add_argument("--xlabel", help="xlabel", default="x (pixels)")
    parser.add_argument("--ylabel", help="ylabel", default="y (pixels)")

    args = parser.parse_args()

    smoothed_data_filename = args.smoothed_data_filename
    fig_filename_pattern = args.fig_filename_pattern
    xlabel = args.xlabel
    ylabel = args.ylabel

    smoothed_data = pd.read_csv(smoothed_data_filename)
    fig = go.Figure()
    trace_pos = go.Scatter(x=smoothed_data["pos1"], y=smoothed_data["pos2"],
                           mode="markers",
                           marker={"color": "black"},
                           customdata=smoothed_data["time"],
                           hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                           name="measured",
                           showlegend=True,
                           )
    trace_filtered = go.Scatter(x=smoothed_data["fpos1"],
                                y=smoothed_data["fpos2"],
                                mode="markers",
                                marker={"color": "red"},
                                customdata=smoothed_data["time"],
                                hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                name="filtered",
                                showlegend=True,
                                )
    trace_smoothed = go.Scatter(x=smoothed_data["spos1"],
                                y=smoothed_data["spos2"],
                                mode="markers",
                                marker={"color": "green"},
                                customdata=smoothed_data["time"],
                                hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                name="smoothed",
                                showlegend=True,
                                )
    fig.add_trace(trace_pos)
    fig.add_trace(trace_filtered)
    fig.add_trace(trace_smoothed)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
