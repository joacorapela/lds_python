'''
Utility functions for sphinx_gallery examples
==============================================

'''

import numpy as np
import plotly.graph_objects as go

def get_fig_kinematics_vs_time(
    time,
    true_x, true_y,
    measured_x, measured_y,
    estimated_mean_x, estimated_mean_y,
    estimated_ci_x_upper, estimated_ci_y_upper,
    estimated_ci_x_lower, estimated_ci_y_lower,
    cb_alpha,
    color_true,
    color_measured,
    color_estimated_pattern,
    xlabel, ylabel):

    fig = go.Figure()
    trace_true_x = go.Scatter(
        x=time, y=true_x,
        mode="markers",
        marker={"color": color_true},
        name="true x",
        showlegend=True,
    )
    fig.add_trace(trace_true_x)
    trace_true_y = go.Scatter(
        x=time, y=true_y,
        mode="markers",
        marker={"color": color_true},
        name="true y",
        showlegend=True,
    )
    fig.add_trace(trace_true_y)
    if measured_x is not None:
        trace_mes_x = go.Scatter(
            x=time, y=measured_x,
            mode="markers",
            marker={"color": color_measured},
            name="measured x",
            showlegend=True,
        )
        fig.add_trace(trace_mes_x)
    if measured_y is not None:
        trace_mes_y = go.Scatter(
            x=time, y=measured_y,
            mode="markers",
            marker={"color": color_measured},
            name="measured y",
            showlegend=True,
        )
        fig.add_trace(trace_mes_y)
    trace_est_x = go.Scatter(
        x=time, y=estimated_mean_x,
        mode="markers",
        marker={"color": color_estimated_pattern.format(1.0)},
        name="estimated x",
        showlegend=True,
        legendgroup="estimated_x",
    )
    fig.add_trace(trace_est_x)
    trace_est_x_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([estimated_ci_x_upper, estimated_ci_x_lower[::-1]]),
        fill="toself",
        fillcolor=color_estimated_pattern.format(cb_alpha),
        line=dict(color=color_estimated_pattern.format(0.0)),
        showlegend=False,
        legendgroup="estimated_x",
    )
    fig.add_trace(trace_est_x_cb)
    trace_est_y = go.Scatter(
        x=time, y=estimated_mean_y,
        mode="markers",
        marker={"color": color_estimated_pattern.format(1.0)},
        name="estimated y",
        showlegend=True,
        legendgroup="estimated_y",
    )
    fig.add_trace(trace_est_y)
    trace_est_y_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([estimated_ci_y_upper, estimated_ci_y_lower[::-1]]),
        fill="toself",
        fillcolor=color_estimated_pattern.format(cb_alpha),
        line=dict(color=color_estimated_pattern.format(0.0)),
        showlegend=False,
        legendgroup="estimated_y",
    )
    fig.add_trace(trace_est_y_cb)

    fig.update_layout(xaxis_title=xlabel,
                      yaxis_title=ylabel,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      yaxis_range=[estimated_mean_x.min(),
                                   estimated_mean_x.max()],
                     )
    return fig
