
import numpy as np
import plotly.graph_objects as go

def get_fig_mfs_positions_2D(time, measured_x=None, measured_y=None,
                             filtered_x=None, filtered_y=None,
                             smoothed_x=None, smoothed_y=None,
                             color_measured="black",
                             color_filtered="red",
                             color_smoothed="green",
                             xlabel="x (pixels)", ylabel="y (pixels)",
                             hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                            ):
    """Returns a figure with measured, measured and/or smoothed (mfs) positions
    with the x and y components plotted against each other.
    """
    fig = go.Figure()
    if measured_x is not None and measured_y is not None:
        trace = go.Scatter(x=measured_x, y=measured_y,
                          mode="lines+markers",
                          marker={"color": color_measured},
                          customdata=time,
                          hovertemplate=hovertemplate,
                          name="measured",
                          showlegend=True,
                          )
        fig.add_trace(trace)
    if filtered_x is not None and filtered_y is not None:
        trace = go.Scatter(x=filtered_x,
                           y=filtered_y,
                           mode="lines+markers",
                           marker={"color": color_filtered},
                           customdata=time,
                           hovertemplate=hovertemplate,
                           name="filtered",
                           showlegend=True,
                           )
        fig.add_trace(trace)
    if smoothed_x is not None and smoothed_y is not None:
        trace = go.Scatter(x=smoothed_x,
                           y=smoothed_y,
                           mode="lines+markers",
                           marker={"color": color_smoothed},
                           customdata=time,
                           hovertemplate=hovertemplate,
                           name="smoothed",
                           showlegend=True,
                           )
        fig.add_trace(trace)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig

def get_fig_mfdfs_kinematics_1D(time, yaxis_title,
                                measured_x=None, measured_y=None,
                                fd_x=None, fd_y=None,
                                filtered_x=None, filtered_y=None,
                                smoothed_x=None, smoothed_y=None,
                                color_measured="black",
                                color_fd="blue",
                                color_filtered="red",
                                color_smoothed="green",
                                symbol_x="circle", symbol_y="circle-open",
                                xlabel1D="x (pixels)", ylabel1D="y (pixels)",
                                xlabel2D="time (sec)", ylabel2D="position (pixels)",
                               ):
    """Returns a figure with measured, finite differences, measured and/or
    smoothed (mfs) kinematics (e.g., positions, velocities or accelerations)
    with the x and y components plotted against time.
    """
    fig = go.Figure()
    if fd_x is not None and fd_y is not None:
        trace_x = go.Scatter(x=time,
                             y=fd_x,
                             mode="lines+markers",
                             marker={"color": color_fd},
                             name="finite diff x",
                             showlegend=True,
                             )
        trace_y = go.Scatter(x=time,
                             y=fd_y,
                             mode="lines+markers",
                             marker={"color": color_fd},
                             name="finite diff y",
                             showlegend=True,
                             )
        fig.add_trace(trace_x)
        fig.add_trace(trace_y)
    if measured_x is not None and measured_y is not None:
        trace_x = go.Scatter(x=time,
                             y=measured_x,
                             mode="lines+markers",
                             marker={"color": color_measured,
                                     "symbol": symbol_x},
                             name="measured x",
                             showlegend=True,
                             )
        trace_y = go.Scatter(x=time,
                             y=measured_y,
                             mode="lines+markers",
                             marker={"color": color_measured,
                                     "symbol": symbol_y},
                             name="measured y",
                             showlegend=True,
                             )
        fig.add_trace(trace_x)
        fig.add_trace(trace_y)
    if filtered_x is not None and filtered_y is not None:
        trace_x = go.Scatter(x=time,
                             y=filtered_x,
                             mode="lines+markers",
                             marker={"color": color_filtered,
                                     "symbol": symbol_x},
                             name="filtered x",
                             showlegend=True,
                             )
        trace_y = go.Scatter(x=time,
                             y=filtered_y,
                             mode="lines+markers",
                             marker={"color": color_filtered,
                                     "symbol": symbol_y},
                             name="filtered y",
                             showlegend=True,
                             )
        fig.add_trace(trace_x)
        fig.add_trace(trace_y)
    if smoothed_x is not None and smoothed_y is not None:
        trace_x = go.Scatter(x=time,
                             y=smoothed_x,
                             mode="lines+markers",
                             marker={"color": color_smoothed,
                                     "symbol": symbol_x},
                             name="smoothed x",
                             showlegend=True,
                             )
        trace_y = go.Scatter(x=time,
                             y=smoothed_y,
                             mode="lines+markers",
                             marker={"color": color_smoothed,
                                     "symbol": symbol_y},
                             name="smoothed y",
                             showlegend=True,
                             )
        fig.add_trace(trace_x)
        fig.add_trace(trace_y)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_title="time (sec)",
                      yaxis_title=yaxis_title,
                     )
    return fig
