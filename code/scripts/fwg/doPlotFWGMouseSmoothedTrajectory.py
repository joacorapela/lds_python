import sys
import pdb
import numpy as np
import pickle
import pandas as pd
import datetime
import argparse
import configparser
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothed_result_number", type=int,
                        help="number corresponding to smoothed result filename")
    parser.add_argument("--variable", type=str, default="pos_vs_time",
                        help="variable to plot: pos, vel, acc")
    parser.add_argument("--color_measures", type=str,
                        default="black",
                        help="color for measures markers")
    parser.add_argument("--color_finite_differences", type=str,
                        default="black",
                        help="color for finite differences markers")
    parser.add_argument("--color_pattern_filtered", type=str,
                        default="rgba(255,0,0,{:f})",
                        help="color for filtered markers")
    parser.add_argument("--color_pattern_smoothed", type=str,
                        default="rgba(0,255,0,{:f})",
                        help="color for smoothed markers")
    parser.add_argument("--cb_alpha", type=float,
                        default=0.3,
                        help="opacity coefficient for confidence bands")
    parser.add_argument("--symbol_x", type=str, default="circle",
                        help="color for x markers")
    parser.add_argument("--symbol_y", type=str, default="circle-open",
                        help="color for y markers")
    parser.add_argument("--smoothed_result_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothed.{:s}", 
                        help="smoothed result filename pattern")
    parser.add_argument("--fig_filename_pattern_pattern", type=str,
                        default="../../figures/{:08d}_smoothed_{:s}.{{:s}}", 
                        help="figure filename pattern")
    parser.add_argument("--data_filename", type=str,
                        default="../../data/positions_session003_start0.00_end15548.27.csv",
                        help="inputs positions filename")

    args = parser.parse_args()

    smoothed_result_number = args.smoothed_result_number
    variable = args.variable
    color_measures = args.color_measures
    color_finite_differences = args.color_finite_differences
    color_pattern_filtered = args.color_pattern_filtered
    color_pattern_smoothed = args.color_pattern_smoothed
    cb_alpha = args.cb_alpha
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    smoothed_result_filename_pattern = args.smoothed_result_filename_pattern
    fig_filename_pattern_pattern = args.fig_filename_pattern_pattern
    data_filename = args.data_filename

    smoothed_result_filename = \
        smoothed_result_filename_pattern.format(smoothed_result_number,
                                                "pickle")
    metadata_filename = \
        smoothed_result_filename_pattern.format(smoothed_result_number, "ini")
    fig_filename_pattern = \
        fig_filename_pattern_pattern.format(smoothed_result_number, variable)

    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    start_position = int(metadata["params"]["start_position"])
    number_positions = int(metadata["params"]["number_positions"])

    data = pd.read_csv(filepath_or_buffer=data_filename)
    data = data.iloc[start_position:start_position+number_positions,:]
    y = np.transpose(data[["x", "y"]].to_numpy())

    with open(smoothed_result_filename, "rb") as f:
        smooth_res = pickle.load(f)
    time = (pd.to_datetime(smooth_res["time"]) - pd.to_datetime(smooth_res["time"][0])).dt.total_seconds().to_numpy()
    fig = go.Figure()
    if variable == "pos_vs_time":
        filter_mean_x = smooth_res["filter_res"]["xnn"][0, 0, :]
        filter_mean_y = smooth_res["filter_res"]["xnn"][3, 0, :]
        filter_std_x_y = np.sqrt(np.diagonal(a=smooth_res["filter_res"]["Vnn"],
                                             axis1=0, axis2=1))
        filter_ci_x_upper = filter_mean_x + 1.96*filter_std_x_y[:, 0]
        filter_ci_x_lower = filter_mean_x - 1.96*filter_std_x_y[:, 0]
        filter_ci_y_upper = filter_mean_y + 1.96*filter_std_x_y[:, 3]
        filter_ci_y_lower = filter_mean_y - 1.96*filter_std_x_y[:, 3]

        smooth_mean_x = smooth_res["smooth_res"]["xnN"][0, 0, :]
        smooth_mean_y = smooth_res["smooth_res"]["xnN"][3, 0, :]
        smooth_std_x_y = np.sqrt(np.diagonal(a=smooth_res["smooth_res"]["VnN"],
                                             axis1=0, axis2=1))
        smooth_ci_x_upper = smooth_mean_x + 1.96*smooth_std_x_y[:, 0]
        smooth_ci_x_lower = smooth_mean_x - 1.96*smooth_std_x_y[:, 0]
        smooth_ci_y_upper = smooth_mean_y + 1.96*smooth_std_x_y[:, 3]
        smooth_ci_y_lower = smooth_mean_y - 1.96*smooth_std_x_y[:, 3]

        trace_mes_x = go.Scatter(
            x=time, y=y[0, :],
            mode="markers",
            marker={"color": color_measures},
            name="measured x",
            showlegend=True,
        )
        trace_mes_y = go.Scatter(
            x=time, y=y[1, :],
            mode="markers",
            marker={"color": color_measures},
            name="measured y",
            showlegend=True,
        )
        trace_filter_x = go.Scatter(
            x=time, y=filter_mean_x,
            mode="markers",
            marker={"color": color_pattern_filtered.format(1.0)},
            name="filtered x",
            showlegend=True,
            legendgroup="filtered_x",
        )
        trace_filter_x_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([filter_ci_x_upper, filter_ci_x_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_filtered.format(cb_alpha),
            line=dict(color=color_pattern_filtered.format(0.0)),
            showlegend=False,
            legendgroup="filtered_x",
        )
        trace_filter_y = go.Scatter(
            x=time, y=filter_mean_y,
            mode="markers",
            marker={"color": color_pattern_filtered.format(1.0)},
            name="filtered y",
            showlegend=True,
            legendgroup="filtered_y",
        )
        trace_filter_y_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([filter_ci_y_upper, filter_ci_y_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_filtered.format(cb_alpha),
            line=dict(color=color_pattern_filtered.format(0.0)),
            showlegend=False,
            legendgroup="filtered_y",
        )
        trace_smooth_x = go.Scatter(
            x=time, y=smooth_mean_x,
            mode="markers",
            marker={"color": color_pattern_smoothed.format(1.0)},
            name="smoothed x",
            showlegend=True,
            legendgroup="smoothed_x",
        )
        trace_smooth_x_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([smooth_ci_x_upper, smooth_ci_x_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_smoothed.format(cb_alpha),
            line=dict(color=color_pattern_smoothed.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_x",
        )
        trace_smooth_y = go.Scatter(
            x=time, y=smooth_mean_y,
            mode="markers",
            marker={"color": color_pattern_smoothed.format(1.0)},
            name="smoothed y",
            showlegend=True,
            legendgroup="smoothed_y",
        )
        trace_smooth_y_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([smooth_ci_y_upper, smooth_ci_y_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_smoothed.format(cb_alpha),
            line=dict(color=color_pattern_smoothed.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_y",
        )

        fig.add_trace(trace_mes_x)
        fig.add_trace(trace_mes_y)
        fig.add_trace(trace_filter_x)
        fig.add_trace(trace_filter_x_cb)
        fig.add_trace(trace_filter_y)
        fig.add_trace(trace_filter_y_cb)
        fig.add_trace(trace_smooth_x)
        fig.add_trace(trace_smooth_x_cb)
        fig.add_trace(trace_smooth_y)
        fig.add_trace(trace_smooth_y_cb)

        fig.update_layout(xaxis_title="time (seconds)",
                          yaxis_title="position (pixels)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          yaxis_range=[filter_mean_x.min(), filter_mean_x.max()],
                         )
    elif variable == "vel":
        filter_mean_x = smooth_res["filter_res"]["xnn"][1, 0, :]
        filter_mean_y = smooth_res["filter_res"]["xnn"][4, 0, :]
        filter_std_x_y = np.sqrt(np.diagonal(a=smooth_res["filter_res"]["Vnn"],
                                             axis1=0, axis2=1))
        filter_ci_x_upper = filter_mean_x + 1.96*filter_std_x_y[:, 1]
        filter_ci_x_lower = filter_mean_x - 1.96*filter_std_x_y[:, 1]
        filter_ci_y_upper = filter_mean_y + 1.96*filter_std_x_y[:, 4]
        filter_ci_y_lower = filter_mean_y - 1.96*filter_std_x_y[:, 4]

        smooth_mean_x = smooth_res["smooth_res"]["xnN"][1, 0, :]
        smooth_mean_y = smooth_res["smooth_res"]["xnN"][4, 0, :]
        smooth_std_x_y = np.sqrt(np.diagonal(a=smooth_res["smooth_res"]["VnN"],
                                             axis1=0, axis2=1))
        smooth_ci_x_upper = smooth_mean_x + 1.96*smooth_std_x_y[:, 1]
        smooth_ci_x_lower = smooth_mean_x - 1.96*smooth_std_x_y[:, 1]
        smooth_ci_y_upper = smooth_mean_y + 1.96*smooth_std_x_y[:, 4]
        smooth_ci_y_lower = smooth_mean_y - 1.96*smooth_std_x_y[:, 4]

        dt = (pd.to_datetime(smooth_res["time"].iloc[1]) - pd.to_datetime(smooth_res["time"].iloc[0])).total_seconds()
        y_vel_fd = np.zeros_like(y)
        y_vel_fd[:, 1:] = (y[:, 1:] - y[:, :-1])/dt
        y_vel_fd[:, 0] = y_vel_fd[:, 1]

        trace_fd_x = go.Scatter(
            x=time,
            y=y_vel_fd[0, :],
            mode="lines+markers",
            marker={"color": color_finite_differences},
            name="finite diff. x",
            showlegend=True,
        )
        trace_fd_y = go.Scatter(
            x=time,
            y=y_vel_fd[1, :],
            mode="lines+markers",
            marker={"color": color_finite_differences},
            name="finite diff. y",
            showlegend=True,
        )
        trace_filter_x = go.Scatter(
            x=time,
            y=filter_mean_x,
            mode="lines+markers",
            marker={"color": color_pattern_filtered.format(1.0), "symbol": symbol_x},
            name="filtered x",
            showlegend=True,
            legendgroup="filtered_x",
        )
        trace_filter_x_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([filter_ci_x_upper, filter_ci_x_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_filtered.format(cb_alpha),
            line=dict(color=color_pattern_filtered.format(0.0)),
            showlegend=False,
            legendgroup="filtered_x",
        )
        trace_filter_y = go.Scatter(
            x=time,
            y=filter_mean_y,
            mode="lines+markers",
            marker={"color": color_pattern_filtered.format(1.0), "symbol": symbol_y},
            name="filtered y",
            showlegend=True,
            legendgroup="filtered_y",
        )
        trace_filter_y_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([filter_ci_y_upper, filter_ci_y_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_filtered.format(cb_alpha),
            line=dict(color=color_pattern_filtered.format(0.0)),
            showlegend=False,
            legendgroup="filtered_y",
        )
        trace_smooth_x = go.Scatter(
            x=time,
            y=smooth_mean_x,
            mode="lines+markers",
            marker={"color": color_pattern_smoothed.format(1.0), "symbol": symbol_x},
            name="smoothed x",
            showlegend=True,
            legendgroup="smoothed_x",
        )
        trace_smooth_x_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([smooth_ci_x_upper, smooth_ci_x_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_smoothed.format(cb_alpha),
            line=dict(color=color_pattern_smoothed.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_x",
        )
        trace_smooth_y = go.Scatter(
            x=time,
            y=smooth_mean_y,
            mode="lines+markers",
            marker={"color": color_pattern_smoothed.format(1.0), "symbol": symbol_y},
            name="smoothed y",
            showlegend=True,
            legendgroup="smoothed_y",
        )
        trace_smooth_y_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([smooth_ci_y_upper, smooth_ci_y_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_smoothed.format(cb_alpha),
            line=dict(color=color_pattern_smoothed.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_y",
        )
        fig.add_trace(trace_fd_x)
        fig.add_trace(trace_fd_y)
        fig.add_trace(trace_filter_x)
        fig.add_trace(trace_filter_x_cb)
        fig.add_trace(trace_filter_y)
        fig.add_trace(trace_filter_y_cb)
        fig.add_trace(trace_smooth_x)
        fig.add_trace(trace_smooth_x_cb)
        fig.add_trace(trace_smooth_y)
        fig.add_trace(trace_smooth_y_cb)
        fig.update_layout(xaxis_title="time (seconds)",
                          yaxis_title="velocity (pixels/second)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "acc":
        filter_mean_x = smooth_res["filter_res"]["xnn"][2, 0, :]
        filter_mean_y = smooth_res["filter_res"]["xnn"][5, 0, :]
        filter_std_x_y = np.sqrt(np.diagonal(a=smooth_res["filter_res"]["Vnn"],
                                             axis1=0, axis2=1))
        filter_ci_x_upper = filter_mean_x + 1.96*filter_std_x_y[:, 2]
        filter_ci_x_lower = filter_mean_x - 1.96*filter_std_x_y[:, 2]
        filter_ci_y_upper = filter_mean_y + 1.96*filter_std_x_y[:, 5]
        filter_ci_y_lower = filter_mean_y - 1.96*filter_std_x_y[:, 5]

        smooth_mean_x = smooth_res["smooth_res"]["xnN"][2, 0, :]
        smooth_mean_y = smooth_res["smooth_res"]["xnN"][5, 0, :]
        smooth_std_x_y = np.sqrt(np.diagonal(a=smooth_res["smooth_res"]["VnN"],
                                             axis1=0, axis2=1))
        smooth_ci_x_upper = smooth_mean_x + 1.96*smooth_std_x_y[:, 2]
        smooth_ci_x_lower = smooth_mean_x - 1.96*smooth_std_x_y[:, 2]
        smooth_ci_y_upper = smooth_mean_y + 1.96*smooth_std_x_y[:, 5]
        smooth_ci_y_lower = smooth_mean_y - 1.96*smooth_std_x_y[:, 5]

        dt = (pd.to_datetime(smooth_res["time"].iloc[1]) - pd.to_datetime(smooth_res["time"].iloc[0])).total_seconds()
        y_vel_fd = np.zeros_like(y)
        y_vel_fd[:, 1:] = (y[:, 1:] - y[:, :-1])/dt
        y_vel_fd[:, 0] = y_vel_fd[:, 1]
        y_acc_fd = np.zeros_like(y_vel_fd)
        y_acc_fd[:, 1:] = (y_vel_fd[:, 1:] - y_vel_fd[:, :-1])/dt
        y_acc_fd[:, 0] = y_acc_fd[:, 1]

        trace_fd_x = go.Scatter(
            x=time,
            y=y_acc_fd[0, :],
            mode="lines+markers",
            marker={"color": color_finite_differences},
            name="finite diff. x",
            showlegend=True,
        )
        trace_fd_y = go.Scatter(
            x=time,
            y=y_acc_fd[1, :],
            mode="lines+markers",
            marker={"color": color_finite_differences},
            name="finite diff. y",
            showlegend=True,
        )
        trace_filter_x = go.Scatter(
            x=time,
            y=filter_mean_x,
            mode="lines+markers",
            marker={"color": color_pattern_filtered.format(1.0), "symbol": symbol_x},
            name="filtered x",
            showlegend=True,
            legendgroup="filtered_x",
        )
        trace_filter_x_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([filter_ci_x_upper, filter_ci_x_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_filtered.format(cb_alpha),
            line=dict(color=color_pattern_filtered.format(0.0)),
            showlegend=False,
            legendgroup="filtered_x",
        )
        trace_filter_y = go.Scatter(
            x=time,
            y=filter_mean_y,
            mode="lines+markers",
            marker={"color": color_pattern_filtered.format(1.0), "symbol": symbol_y},
            name="filtered y",
            showlegend=True,
            legendgroup="filtered_y",
        )
        trace_filter_y_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([filter_ci_y_upper, filter_ci_y_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_filtered.format(cb_alpha),
            line=dict(color=color_pattern_filtered.format(0.0)),
            showlegend=False,
            legendgroup="filtered_y",
        )
        trace_smooth_x = go.Scatter(
            x=time,
            y=smooth_mean_x,
            mode="lines+markers",
            marker={"color": color_pattern_smoothed.format(1.0), "symbol": symbol_x},
            name="smoothed x",
            showlegend=True,
            legendgroup="smoothed_x",
        )
        trace_smooth_x_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([smooth_ci_x_upper, smooth_ci_x_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_smoothed.format(cb_alpha),
            line=dict(color=color_pattern_smoothed.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_x",
        )
        trace_smooth_y = go.Scatter(
            x=time,
            y=smooth_mean_y,
            mode="lines+markers",
            marker={"color": color_pattern_smoothed.format(1.0), "symbol": symbol_y},
            name="smoothed y",
            showlegend=True,
            legendgroup="smoothed_y",
        )
        trace_smooth_y_cb = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([smooth_ci_y_upper, smooth_ci_y_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_smoothed.format(cb_alpha),
            line=dict(color=color_pattern_smoothed.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_y",
        )
        fig.add_trace(trace_fd_x)
        fig.add_trace(trace_fd_y)
        fig.add_trace(trace_filter_x)
        fig.add_trace(trace_filter_x_cb)
        fig.add_trace(trace_filter_y)
        fig.add_trace(trace_filter_y_cb)
        fig.add_trace(trace_smooth_x)
        fig.add_trace(trace_smooth_x_cb)
        fig.add_trace(trace_smooth_y)
        fig.add_trace(trace_smooth_y_cb)

        fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
