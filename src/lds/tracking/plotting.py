
import numpy as np
import plotly.graph_objects as go

def get_fig_mfs_positions_2D(time, measured_x=None, measured_y=None,
                             filtered_x=None, filtered_y=None,
                             smoothed_x=None, smoothed_y=None,
                             xlabel="x", ylabel="y",
                             color_measured="black",
                            ):
    """Returns a figure with measured, filtered and smoothed (mfs) positions
    """
    if measured_x is not None and measured_y is not None:
        trace_fd = go.Scatter(x=measured_x, y=measured_x,
                              mode="markers",
                              marker={"color": color_measured},
                              customdata=time,
                              hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                              name="measured",
                              showlegend=True,
                              )
    if filtered_x is not None and filtered_y is not None:
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
        fig.add_trace(trace_fd)
        fig.add_trace(trace_filtered)
        fig.add_trace(trace_smoothed)
        fig.update_layout(xaxis_title="x (pixels)", yaxis_title="y (pixels)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')

def get_fig_mfs_kinematics_1D(time, measured_x, measured_y, fd_x, fd_y,
                              filtered_x, filtered_y, smoothed_x, smoothed_y,
                              xlabel="x", ylabel="y"):
    trace_fd_x = go.Scatter(x=smoothed_data["time"],
                             y=y_acc_fd[0, :],
                             mode="lines+markers",
                             marker={"color": color_measured},
                             name="finite diff x",
                             showlegend=True,
                             )
    trace_fd_y = go.Scatter(x=smoothed_data["time"],
                             y=y_acc_fd[1, :],
                             mode="lines+markers",
                             marker={"color": color_measured},
                             name="finite diff y",
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
    fig.add_trace(trace_fd_x)
    fig.add_trace(trace_fd_y)
    fig.add_trace(trace_filtered_x)
    fig.add_trace(trace_filtered_y)
    fig.add_trace(trace_smoothed_x)
    fig.add_trace(trace_smoothed_y)
    fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')

