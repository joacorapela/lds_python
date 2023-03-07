import sys
import pickle
import numpy as np
import pandas as pd
import datetime
import argparse
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothed_results_filename",
                        help="smoothed data filename")
    parser.add_argument("fig_filename_pattern",
                        help="figure filename pattern")
    parser.add_argument("--min_likelihood", type=float,
                        help=("positions with likelihood less than "
                              "min_likelihood are set to np.nan"),
                        default=0.99)
    parser.add_argument("--variable", type=str,
                        help="variable to plot: pos, vel, acc", default="pos")
    parser.add_argument("--color_measured", type=str,
                        help="color for measured markers", default="black")
    parser.add_argument("--color_fd", type=str,
                        help="color for finite differences markers",
                        default="blue")
    parser.add_argument("--color_filtered_pattern", type=str,
                        help="color for filtered markers",
                        default="rgba(255,0,0,{:f})")
    parser.add_argument("--color_smoothed_pattern", type=str,
                        help="color for smoothed markers",
                        default="rgba(0,255,0,{:f})")
    parser.add_argument("--symbol_x", type=str,
                        help="color for x markers", default="circle")
    parser.add_argument("--symbol_y", type=str,
                        help="color for y markers", default="circle-open")
    parser.add_argument("--cb_transparency", type=float,
                        help="transparency factor for confidence bands", default=0.3)

    args = parser.parse_args()

    smoothed_results_filename = args.smoothed_results_filename
    fig_filename_pattern = args.fig_filename_pattern
    variable = args.variable
    color_measured = args.color_measured
    color_fd = args.color_fd
    color_filtered_pattern = args.color_filtered_pattern
    color_smoothed_pattern = args.color_smoothed_pattern
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    cb_transparency = args.cb_transparency

    with open(smoothed_results_filename, "rb") as f:
        load_res = pickle.load(f)
    fs = load_res["fs"]
    y = load_res["pos"]
    filter_res = load_res["filter_res"]
    smooth_res = load_res["smooth_res"]
    time = load_res["time"]

    dt = 1.0 / fs

    fig = go.Figure()
    if variable == "pos":
        mpos1 = y[0,:]
        mpos2 = y[1,:]

        fpos1_means = filter_res["xnn"][0,0,:]
        fpos2_means = filter_res["xnn"][3,0,:]
        f_covs   = filter_res["Vnn"]
        f_vars   = np.diagonal(f_covs, axis1=0, axis2=1)
        fpos1_stds = np.sqrt(f_vars[:, 0])
        fpos1_upper = fpos1_means + 1.96 * fpos1_stds
        fpos1_lower = fpos1_means - 1.96 * fpos1_stds
        fpos2_stds = np.sqrt(f_vars[:, 3])
        fpos2_upper = fpos2_means + 1.96 * fpos2_stds
        fpos2_lower = fpos2_means - 1.96 * fpos2_stds

        spos1_means = smooth_res["xnN"][0,0,:]
        spos2_means = smooth_res["xnN"][3,0,:]
        s_covs   = smooth_res["VnN"]
        s_vars   = np.diagonal(s_covs, axis1=0, axis2=1)
        spos1_stds = np.sqrt(s_vars[:, 0])
        spos1_upper = spos1_means + 1.96 * spos1_stds
        spos1_lower = spos1_means - 1.96 * spos1_stds
        spos2_stds = np.sqrt(s_vars[:, 3])
        spos2_upper = spos2_means + 1.96 * spos2_stds
        spos2_lower = spos2_means - 1.96 * spos2_stds

        trace_measured_pos1 = go.Scatter(
            x=time, y=mpos1,
            mode="lines+markers",
            marker={"color": color_measured},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>pos</b>:%{y:.3f}",
            name="x measured",
            showlegend=True,
        )
        trace_filtered_pos1 = go.Scatter(
            x=time, y=fpos1_means,
            mode="lines+markers",
            marker={"color": color_filtered_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>pos</b>:%{y:.3f}",
            name="x filtered",
            showlegend=True,
            legendgroup="filtered_pos1",
        )
        trace_filtered_pos1_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((fpos1_upper, fpos1_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_filtered_pattern.format(cb_transparency),
            line=dict(color=color_filtered_pattern.format(0.0)),
            showlegend=False,
            legendgroup="filtered_pos1",
        )
        trace_smoothed_pos1 = go.Scatter(
            x=time, y=spos1_means,
            mode="lines+markers",
            marker={"color": color_smoothed_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>pos</b>:%{y:.3f}",
            name="x smoothed",
            showlegend=True,
            legendgroup="smoothed_pos1",
        )
        trace_smoothed_pos1_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((spos1_upper, spos1_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_smoothed_pattern.format(cb_transparency),
            line=dict(color=color_smoothed_pattern.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_pos1",
        )
        trace_measured_pos2 = go.Scatter(
            x=time, y=mpos2,
            mode="lines+markers",
            marker={"color": color_measured},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>pos</b>:%{y:.3f}",
            name="y measured",
            showlegend=True,
        )
        trace_filtered_pos2 = go.Scatter(
            x=time, y=fpos2_means,
            mode="lines+markers",
            marker={"color": color_filtered_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>pos</b>:%{y:.3f}",
            name="y filtered",
            showlegend=True,
            legendgroup="filtered_pos2",
        )
        trace_filtered_pos2_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((fpos2_upper, fpos2_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_filtered_pattern.format(cb_transparency),
            line=dict(color=color_filtered_pattern.format(0.0)),
            showlegend=False,
            legendgroup="filtered_pos2",
        )
        trace_smoothed_pos2 = go.Scatter(
            x=time, y=spos2_means,
            mode="lines+markers",
            marker={"color": color_smoothed_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>pos</b>:%{y:.3f}",
            name="y smoothed",
            showlegend=True,
            legendgroup="smoothed_pos2",
        )
        trace_smoothed_pos2_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((spos2_upper, spos2_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_smoothed_pattern.format(cb_transparency),
            line=dict(color=color_smoothed_pattern.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_pos2",
        )
        fig.add_trace(trace_filtered_pos1_cb)
        fig.add_trace(trace_smoothed_pos1_cb)
        fig.add_trace(trace_filtered_pos2_cb)
        fig.add_trace(trace_smoothed_pos2_cb)
        fig.add_trace(trace_measured_pos1)
        fig.add_trace(trace_filtered_pos1)
        fig.add_trace(trace_smoothed_pos1)
        fig.add_trace(trace_measured_pos2)
        fig.add_trace(trace_filtered_pos2)
        fig.add_trace(trace_smoothed_pos2)
        fig.update_layout(xaxis_title="time", yaxis_title="position",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')

    elif variable == "vel":
        y_vel_fd = np.zeros_like(y)
        y_vel_fd[:, 1:] = (y[:, 1:] - y[:, :-1])/dt
        y_vel_fd[:, 0] = y_vel_fd[:, 1]

        fvel1_means = filter_res["xnn"][1,0,:]
        fvel2_means = filter_res["xnn"][4,0,:]
        f_covs   = filter_res["Vnn"]
        f_vars   = np.diagonal(f_covs, axis1=0, axis2=1)
        fvel1_stds = np.sqrt(f_vars[:, 1])
        fvel1_upper = fvel1_means + 1.96 * fvel1_stds
        fvel1_lower = fvel1_means - 1.96 * fvel1_stds
        fvel2_stds = np.sqrt(f_vars[:, 4])
        fvel2_upper = fvel2_means + 1.96 * fvel2_stds
        fvel2_lower = fvel2_means - 1.96 * fvel2_stds

        svel1_means = smooth_res["xnN"][1,0,:]
        svel2_means = smooth_res["xnN"][4,0,:]
        s_covs   = smooth_res["VnN"]
        s_vars   = np.diagonal(s_covs, axis1=0, axis2=1)
        svel1_stds = np.sqrt(s_vars[:, 1])
        svel1_upper = svel1_means + 1.96 * svel1_stds
        svel1_lower = svel1_means - 1.96 * svel1_stds
        svel2_stds = np.sqrt(s_vars[:, 4])
        svel2_upper = svel2_means + 1.96 * svel2_stds
        svel2_lower = svel2_means - 1.96 * svel2_stds

        trace_fd_vel1 = go.Scatter(
            x=time, y=y_vel_fd[0,:],
            mode="lines+markers",
            marker={"color": color_fd},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>vel</b>:%{y:.3f}",
            name="x finite differences",
            showlegend=True,
        )
        trace_filtered_vel1 = go.Scatter(
            x=time, y=fvel1_means,
            mode="lines+markers",
            marker={"color": color_filtered_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>vel</b>:%{y:.3f}",
            name="x filtered",
            showlegend=True,
            legendgroup="filtered_vel1",
        )
        trace_filtered_vel1_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((fvel1_upper, fvel1_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_filtered_pattern.format(cb_transparency),
            line=dict(color=color_filtered_pattern.format(0.0)),
            showlegend=False,
            legendgroup="filtered_vel1",
        )
        trace_smoothed_vel1 = go.Scatter(
            x=time, y=svel1_means,
            mode="lines+markers",
            marker={"color": color_smoothed_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>vel</b>:%{y:.3f}",
            name="x smoothed",
            showlegend=True,
            legendgroup="smoothed_vel1",
        )
        trace_smoothed_vel1_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((svel1_upper, svel1_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_smoothed_pattern.format(cb_transparency),
            line=dict(color=color_smoothed_pattern.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_vel1",
        )
        trace_fd_vel2 = go.Scatter(
            x=time, y=y_vel_fd[1,:],
            mode="lines+markers",
            marker={"color": color_fd},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>vel</b>:%{y:.3f}",
            name="y finite differences",
            showlegend=True,
        )
        trace_filtered_vel2 = go.Scatter(
            x=time, y=fvel2_means,
            mode="lines+markers",
            marker={"color": color_filtered_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>vel</b>:%{y:.3f}",
            name="y filtered",
            showlegend=True,
            legendgroup="filtered_vel2",
        )
        trace_filtered_vel2_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((fvel2_upper, fvel2_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_filtered_pattern.format(cb_transparency),
            line=dict(color=color_filtered_pattern.format(0.0)),
            showlegend=False,
            legendgroup="filtered_vel2",
        )
        trace_smoothed_vel2 = go.Scatter(
            x=time, y=svel2_means,
            mode="lines+markers",
            marker={"color": color_smoothed_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>vel</b>:%{y:.3f}",
            name="y smoothed",
            showlegend=True,
            legendgroup="smoothed_vel2",
        )
        trace_smoothed_vel2_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((svel2_upper, svel2_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_smoothed_pattern.format(cb_transparency),
            line=dict(color=color_smoothed_pattern.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_vel2",
        )
        fig.add_trace(trace_filtered_vel1_cb)
        fig.add_trace(trace_smoothed_vel1_cb)
        fig.add_trace(trace_filtered_vel2_cb)
        fig.add_trace(trace_smoothed_vel2_cb)
        fig.add_trace(trace_fd_vel1)
        fig.add_trace(trace_filtered_vel1)
        fig.add_trace(trace_smoothed_vel1)
        fig.add_trace(trace_fd_vel2)
        fig.add_trace(trace_filtered_vel2)
        fig.add_trace(trace_smoothed_vel2)
        fig.update_layout(xaxis_title="time", yaxis_title="velocity",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "acc":
        y_vel_fd = np.zeros_like(y)
        y_vel_fd[:, 1:] = (y[:, 1:] - y[:, :-1])/dt
        y_vel_fd[:, 0] = y_vel_fd[:, 1]
        y_acc_fd = np.zeros_like(y_vel_fd)
        y_acc_fd[:, 1:] = (y_vel_fd[:, 1:] - y_vel_fd[:, :-1])/dt
        y_acc_fd[:, 0] = y_acc_fd[:, 1]

        facc1_means = filter_res["xnn"][2,0,:]
        facc2_means = filter_res["xnn"][5,0,:]
        f_covs   = filter_res["Vnn"]
        f_vars   = np.diagonal(f_covs, axis1=0, axis2=1)
        facc1_stds = np.sqrt(f_vars[:, 5])
        facc1_upper = facc1_means + 1.96 * facc1_stds
        facc1_lower = facc1_means - 1.96 * facc1_stds
        facc2_stds = np.sqrt(f_vars[:, 5])
        facc2_upper = facc2_means + 1.96 * facc2_stds
        facc2_lower = facc2_means - 1.96 * facc2_stds

        sacc1_means = smooth_res["xnN"][2,0,:]
        sacc2_means = smooth_res["xnN"][5,0,:]
        s_covs   = smooth_res["VnN"]
        s_vars   = np.diagonal(s_covs, axis1=0, axis2=1)
        sacc1_stds = np.sqrt(s_vars[:, 2])
        sacc1_upper = sacc1_means + 1.96 * sacc1_stds
        sacc1_lower = sacc1_means - 1.96 * sacc1_stds
        sacc2_stds = np.sqrt(s_vars[:, 5])
        sacc2_upper = sacc2_means + 1.96 * sacc2_stds
        sacc2_lower = sacc2_means - 1.96 * sacc2_stds

        trace_fd_acc1 = go.Scatter(
            x=time, y=y_acc_fd[0,:],
            mode="lines+markers",
            marker={"color": color_fd},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>acc</b>:%{y:.3f}<br>",
            name="x finite differences",
            showlegend=True,
        )
        trace_filtered_acc1 = go.Scatter(
            x=time, y=facc1_means,
            mode="lines+markers",
            marker={"color": color_filtered_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>acc</b>:%{y:.3f}<br>",
            name="x filtered",
            showlegend=True,
            legendgroup="filtered_acc1",
        )
        trace_filtered_acc1_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((facc1_upper, facc1_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_filtered_pattern.format(cb_transparency),
            line=dict(color=color_filtered_pattern.format(0.0)),
            showlegend=False,
            legendgroup="filtered_acc1",
        )
        trace_smoothed_acc1 = go.Scatter(
            x=time, y=sacc1_means,
            mode="lines+markers",
            marker={"color": color_smoothed_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>acc</b>:%{y:.3f}<br>",
            name="x smoothed",
            showlegend=True,
            legendgroup="smoothed_acc1",
        )
        trace_smoothed_acc1_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((sacc1_upper, sacc1_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_smoothed_pattern.format(cb_transparency),
            line=dict(color=color_smoothed_pattern.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_acc1",
        )
        trace_fd_acc2 = go.Scatter(
            x=time, y=y_acc_fd[1,:],
            mode="lines+markers",
            marker={"color": color_fd},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>acc</b>:%{y:.3f}<br>",
            name="y finite differences",
            showlegend=True,
        )
        trace_filtered_acc2 = go.Scatter(
            x=time, y=facc2_means,
            mode="lines+markers",
            marker={"color": color_filtered_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>acc</b>:%{y:.3f}<br>",
            name="y filtered",
            showlegend=True,
            legendgroup="filtered_acc2",
        )
        trace_filtered_acc2_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((facc2_upper, facc2_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_filtered_pattern.format(cb_transparency),
            line=dict(color=color_filtered_pattern.format(0.0)),
            showlegend=False,
            legendgroup="filtered_acc2",
        )
        trace_smoothed_acc2 = go.Scatter(
            x=time, y=sacc2_means,
            mode="lines+markers",
            marker={"color": color_smoothed_pattern.format(1.0)},
            hovertemplate="<b>time</b>:%{x:.3f}<br><b>acc</b>:%{y:.3f}<br>",
            name="y smoothed",
            showlegend=True,
            legendgroup="smoothed_acc2",
        )
        trace_smoothed_acc2_cb = go.Scatter(
            x=np.concatenate((time, time[::-1])),
            y=np.concatenate((sacc2_upper, sacc2_lower[::-1].squeeze())),
            fill="toself",
            fillcolor=color_smoothed_pattern.format(cb_transparency),
            line=dict(color=color_smoothed_pattern.format(0.0)),
            showlegend=False,
            legendgroup="smoothed_acc2",
        )
        fig.add_trace(trace_filtered_acc1_cb)
        fig.add_trace(trace_smoothed_acc1_cb)
        fig.add_trace(trace_filtered_acc2_cb)
        fig.add_trace(trace_smoothed_acc2_cb)
        fig.add_trace(trace_fd_acc1)
        fig.add_trace(trace_filtered_acc1)
        fig.add_trace(trace_smoothed_acc1)
        fig.add_trace(trace_fd_acc2)
        fig.add_trace(trace_filtered_acc2)
        fig.add_trace(trace_smoothed_acc2)
        fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))

    fig.write_image(fig_filename_pattern.format(variable, "png"))
    fig.write_html(fig_filename_pattern.format(variable, "html"))
    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
