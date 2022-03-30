import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import argparse
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothed_data_filename",
                        help="smoothed data filename")
    parser.add_argument("fig_filename_pattern",
                        help="figure filename pattern")
    parser.add_argument("--variable", type=str, default="pos",
                        help="variable to plot: pos, vel, acc")
    parser.add_argument("--start_position", type=int, default=0,
    # parser.add_argument("--start_position", type=int, default=10000,
                        help="start position to smooth")
    parser.add_argument("--number_positions", type=int, default=10000,
    # parser.add_argument("--number_positions", type=int, default=4500,
                        help="number of positions to smooth")
    parser.add_argument("--color_measured", type=str, default="black",
                        help="color for measured markers")
    parser.add_argument("--color_filtered", type=str, default="red",
                        help="color for filtered markers")
    parser.add_argument("--color_smoothed", type=str, default="green",
                        help="color for smoothed markers")
    parser.add_argument("--symbol_x", type=str, default="circle",
                        help="color for x markers")
    parser.add_argument("--symbol_y", type=str, default="circle-open",
                        help="color for y markers")
    parser.add_argument("--data_filename", type=str,
                        default="../../data/positions_session003_start0.00_end15548.27.csv",
                        help="inputs positions filename")

    args = parser.parse_args()

    smoothed_data_filename = args.smoothed_data_filename
    fig_filename_pattern = args.fig_filename_pattern
    variable = args.variable
    start_position = args.start_position
    number_positions = args.number_positions
    color_measured = args.color_measured
    color_filtered = args.color_filtered
    color_smoothed = args.color_smoothed
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    data_filename = args.data_filename

    data = pd.read_csv(filepath_or_buffer=data_filename)
    data = data.iloc[start_position:start_position+number_positions,:]
    y = np.transpose(data[["x", "y"]].to_numpy())

    smoothed_data = pd.read_csv(smoothed_data_filename)
    dt = (pd.to_datetime(smoothed_data["time"].iloc[1]) - pd.to_datetime(smoothed_data["time"].iloc[0])).total_seconds()
    y_vel_fd = np.zeros_like(y)
    y_vel_fd[:, 1:] = (y[:, 1:] - y[:, :-1])/dt
    y_vel_fd[:, 0] = y_vel_fd[:, 1]
    y_acc_fd = np.zeros_like(y_vel_fd)
    y_acc_fd[:, 1:] = (y_vel_fd[:, 1:] - y_vel_fd[:, :-1])/dt
    y_acc_fd[:, 0] = y_acc_fd[:, 1]

    fig = go.Figure()
    if variable == "pos":
        trace_mes = go.Scatter(x=y[0, :], y=y[1, :],
                               mode="markers",
                               marker={"color": color_measured},
                               customdata=smoothed_data["time"],
                               hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                               name="measured",
                               showlegend=True,
                               )
        trace_filtered = go.Scatter(x=smoothed_data["fpos1"],
                                    y=smoothed_data["fpos2"],
                                    mode="markers",
                                    marker={"color": color_filtered},
                                    customdata=smoothed_data["time"],
                                    hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                    name="filtered",
                                    showlegend=True,
                                    )
        trace_smoothed = go.Scatter(x=smoothed_data["spos1"],
                                    y=smoothed_data["spos2"],
                                    mode="markers",
                                    marker={"color": color_smoothed},
                                    customdata=smoothed_data["time"],
                                    hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                    name="smoothed",
                                    showlegend=True,
                                    )
        fig.add_trace(trace_mes)
        fig.add_trace(trace_filtered)
        fig.add_trace(trace_smoothed)
        fig.update_layout(xaxis_title="x (pixels)", yaxis_title="y (pixels)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "vel":
        trace_mes_x = go.Scatter(x=smoothed_data["time"],
                                 y=y_vel_fd[0, :],
                                 mode="lines+markers",
                                 marker={"color": color_measured},
                                 name="measured x",
                                 showlegend=True,
                                 )
        trace_mes_y = go.Scatter(x=smoothed_data["time"],
                                 y=y_vel_fd[1, :],
                                 mode="lines+markers",
                                 marker={"color": color_measured},
                                 name="measured y",
                                 showlegend=True,
                                 )
        trace_filtered_x = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["fvel1"],
                                      mode="lines+markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["fvel2"],
                                      mode="lines+markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        trace_smoothed_x = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["svel1"],
                                      mode="lines+markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_x},
                                      name="smoothed x",
                                      showlegend=True,
                                      )
        trace_smoothed_y = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["svel2"],
                                      mode="lines+markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_y},
                                      name="smoothed y",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_mes_x)
        fig.add_trace(trace_mes_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.add_trace(trace_smoothed_x)
        fig.add_trace(trace_smoothed_y)
        fig.update_layout(xaxis_title="time", yaxis_title="velocity",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "acc":
        trace_mes_x = go.Scatter(x=smoothed_data["time"],
                                 y=y_acc_fd[0, :],
                                 mode="lines+markers",
                                 marker={"color": color_measured},
                                 name="measured x",
                                 showlegend=True,
                                 )
        trace_mes_y = go.Scatter(x=smoothed_data["time"],
                                 y=y_acc_fd[1, :],
                                 mode="lines+markers",
                                 marker={"color": color_measured},
                                 name="measured y",
                                 showlegend=True,
                                 )
        trace_filtered_x = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["facc1"],
                                      mode="lines+markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["facc2"],
                                      mode="lines+markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        trace_smoothed_x = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["sacc1"],
                                      mode="lines+markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_x},
                                      name="smoothed x",
                                      showlegend=True,
                                      )
        trace_smoothed_y = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["sacc2"],
                                      mode="lines+markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_y},
                                      name="smoothed y",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_mes_x)
        fig.add_trace(trace_mes_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.add_trace(trace_smoothed_x)
        fig.add_trace(trace_smoothed_y)
        fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))

    fig.write_image(fig_filename_pattern.format(variable, "png"))
    fig.write_html(fig_filename_pattern.format(variable, "html"))
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
