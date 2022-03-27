import sys
import pdb
import numpy as np
import argparse
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", type=int,
                        help="simulation result number")
    parser.add_argument("--variable", type=str, default="pos",
                        help="variable to plot: pos, vel, acc")
    parser.add_argument("--color_measured", type=str, default="black",
                        help="color for measured markers")
    parser.add_argument("--color_true", type=str, default="blue",
                        help="color for true markers")
    parser.add_argument("--symbol_x", type=str, default="circle",
                        help="color for x markers")
    parser.add_argument("--symbol_y", type=str, default="circle-open",
                        help="color for y markers")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_simulation_{:s}.{:s}")

    args = parser.parse_args()

    simRes_number = args.simRes_number
    variable = args.variable
    color_measured = args.color_measured
    color_true = args.color_true
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    simRes_filename_pattern = args.simRes_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

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
        fig.add_trace(trace_mes)
        fig.add_trace(trace_true)
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
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
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
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
        fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))


    fig.write_image(fig_filename_pattern.format(simRes_number, variable,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(simRes_number, variable,
                                               "html"))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
