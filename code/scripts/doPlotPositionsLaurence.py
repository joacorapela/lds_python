import sys
import pdb
import h5py
import numpy as np
import argparse
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str,
                        default="../../data/cam1DLC_resnet50_NPX_7May20shuffle1_500000.h5",
                        help="data filename")
    parser.add_argument("--first_sample", type=int, default=0,
                        help="first sample")
    parser.add_argument("--number_samples", type=int, default=10000,
                        help="number of samples to smooth")
    parser.add_argument("--sample_rate", type=float, default=1,
                        help="sample rate")
    parser.add_argument("--colorscale", type=str, default="Rainbow",
                        help="colorscale name")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/positions_fede_from{:d}_numberSamples{:d}.{:s}")

    args = parser.parse_args()

    data_filename = args.data_filename
    first_sample = args.first_sample
    number_samples = args.number_samples
    sample_rate = args.sample_rate
    colorscale = args.colorscale
    fig_filename_pattern = args.fig_filename_pattern

    with h5py.File(data_filename, "r") as f:
        some_keys = f.keys()
        breakpoint()

    pos = np.asarray([list(data_json["x"].values()), list(data_json["y"].values())])
    pos = pos[:, first_sample:(first_sample+number_samples)]
    N = pos.shape[1]
    dt = 1.0/sample_rate
    time = np.linspace(first_sample*dt, (first_sample+number_samples)*dt, dt)

    fig = go.Figure()
    trace_mes = go.Scatter(x=pos[0, :], y=pos[1, :],
                           mode="markers",
                           marker={"color": time,
                                   "colorscale": colorscale,
                                   "colorbar": {"title": "Time"},
                                  },
                           customdata=time,
                           hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                           name="measured",
                           showlegend=False,
                          )
    fig.add_trace(trace_mes)
    fig.update_layout(xaxis_title="x (pixels)",
                      yaxis_title="y (pixels)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    fig.write_image(fig_filename_pattern.format(first_sample, number_samples, "png"))
    fig.write_html(fig_filename_pattern.format(first_sample, number_samples, "html"))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
