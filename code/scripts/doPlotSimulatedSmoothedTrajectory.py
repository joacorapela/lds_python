import sys
import pdb
import numpy as np
import pandas as pd
import argparse
import configparser
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothed_data_number", type=int,
                        help="number corresponding to smoothed data filename")
    parser.add_argument("--variable", type=str, default="pos",
                        help="variable to plot: pos, vel, acc")
    parser.add_argument("--color_measured", type=str, default="black",
                        help="color for measured markers")
    parser.add_argument("--color_true", type=str, default="blue",
                        help="color for true markers")
    parser.add_argument("--color_filtered", type=str, default="red",
                        help="color for filtered markers")
    parser.add_argument("--color_smoothed", type=str, default="green",
                        help="color for smoothed markers")
    parser.add_argument("--symbol_x", type=str, default="circle",
                        help="color for x markers")
    parser.add_argument("--symbol_y", type=str, default="circle-open",
                        help="color for y markers")
    parser.add_argument("--smoothed_data_filenames_pattern", type=str,
                        default="../../results/{:08d}_smoothed.{:s}",
                        help="smoothed_data filename pattern")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_{:s}_smoothed.{:s}")

    args = parser.parse_args()

    smoothed_data_number = args.smoothed_data_number
    variable = args.variable
    color_measured = args.color_measured
    color_true = args.color_true
    color_filtered = args.color_filtered
    color_smoothed = args.color_smoothed
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    smoothed_data_filenames_pattern = \
        args.smoothed_data_filenames_pattern
    simRes_filename_pattern = args.simRes_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    smoothed_data_filename = \
        smoothed_data_filenames_pattern.format(smoothed_data_number, "csv")
    smoothed_data = pd.read_csv(smoothed_data_filename)

    smoothed_metadata_filename = \
        smoothed_data_filenames_pattern.format(smoothed_data_number, "ini")
    smoothed_metadata = configparser.ConfigParser()
    smoothed_metadata.read(smoothed_metadata_filename)
    simRes_number = int(smoothed_metadata["params"]["simRes_number"])
    simRes_filename = simRes_filename_pattern.format(simRes_number, "npz")
    simRes = np.load(simRes_filename)

    N = simRes["y"].shape[1]
    time = np.arange(0, N*simRes["dt"], simRes["dt"])
    fig = go.Figure()
    if variable == "pos":
        trace_mes = go.Scatter(x=simRes["y"][0, :], y=simRes["y"][1, :],
                               mode="markers",
                               marker={"color": color_measured},
                               customdata=time,
                               hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                               name="measured",
                               showlegend=True,
                               )
        trace_true = go.Scatter(x=simRes["x"][0, :], y=simRes["x"][3, :],
                                mode="markers",
                                marker={"color": color_true},
                                customdata=time,
                                hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                name="true",
                                showlegend=True,
                                )
        trace_filtered = go.Scatter(x=smoothed_data["fpos1"],
                                    y=smoothed_data["fpos2"],
                                    mode="markers",
                                    marker={"color": color_filtered},
                                    customdata=time,
                                    hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                    name="filtered",
                                    showlegend=True,
                                    )
        trace_smoothed = go.Scatter(x=smoothed_data["spos1"],
                                    y=smoothed_data["spos2"],
                                    mode="markers",
                                    marker={"color": color_smoothed},
                                    customdata=time,
                                    hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                    name="smoothed",
                                    showlegend=True,
                                    )
        fig.add_trace(trace_mes)
        fig.add_trace(trace_true)
        fig.add_trace(trace_filtered)
        fig.add_trace(trace_smoothed)
        fig.update_layout(xaxis_title="x (pixels)", yaxis_title="y (pixels)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "vel":
        trace_true_x = go.Scatter(x=time, y=simRes["x"][1, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_x},
                                  name="true x",
                                  showlegend=True,
                                  )
        trace_true_y = go.Scatter(x=time, y=simRes["x"][4, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_y},
                                  name="true y",
                                  showlegend=True,
                                  )
        trace_filtered_x = go.Scatter(x=time,
                                      y=smoothed_data["fvel1"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=time,
                                      y=smoothed_data["fvel2"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        trace_smoothed_x = go.Scatter(x=time,
                                      y=smoothed_data["svel1"],
                                      mode="markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_x},
                                      name="smoothed x",
                                      showlegend=True,
                                      )
        trace_smoothed_y = go.Scatter(x=time,
                                      y=smoothed_data["svel2"],
                                      mode="markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_y},
                                      name="smoothed x",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.add_trace(trace_smoothed_x)
        fig.add_trace(trace_smoothed_y)
        fig.update_layout(xaxis_title="time", yaxis_title="velocity",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "acc":
        trace_true_x = go.Scatter(x=time, y=simRes["x"][2, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_x},
                                  name="true x",
                                  showlegend=True,
                                  )
        trace_true_y = go.Scatter(x=time, y=simRes["x"][5, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_y},
                                  name="true y",
                                  showlegend=True,
                                  )
        trace_filtered_x = go.Scatter(x=time,
                                      y=smoothed_data["facc1"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=time,
                                      y=smoothed_data["facc2"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        trace_smoothed_x = go.Scatter(x=time,
                                      y=smoothed_data["sacc1"],
                                      mode="markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_x},
                                      name="smoothed x",
                                      showlegend=True,
                                      )
        trace_smoothed_y = go.Scatter(x=time,
                                      y=smoothed_data["sacc2"],
                                      mode="markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_y},
                                      name="smoothed x",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.add_trace(trace_smoothed_x)
        fig.add_trace(trace_smoothed_y)
        fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))


    fig.write_image(fig_filename_pattern.format(smoothed_data_number, variable, "png"))
    fig.write_html(fig_filename_pattern.format(smoothed_data_number, variable, "html"))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
