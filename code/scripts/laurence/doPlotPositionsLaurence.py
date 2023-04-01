import sys
import pdb
import h5py
import numpy as np
import pandas as pd
import argparse
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str,
                        default="../../data/cam1DLC_resnet50_NPX_7May20shuffle1_500000.h5",
                        help="data filename")
    parser.add_argument("--bodypart", type=str, help="body part to track",
                        default="upper_back")
    parser.add_argument("--min_likelihood", type=float,
                        help=("positions with likelihood less than "
                              "min_likelihood are set to np.nan"),
                        default=0.99)
    parser.add_argument("--first_sample", type=int, default=0,
                        help="first sample")
    parser.add_argument("--number_samples", type=int, default=10000,
                        help="number of samples to smooth")
    parser.add_argument("--sample_rate", type=float, default=40,
                        help="sample rate (Hz)")
    parser.add_argument("--colorscale", type=str, default="Rainbow",
                        help="colorscale name")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/positions_laurence_from{:d}_numberSamples{:d}.{:s}")

    args = parser.parse_args()

    data_filename = args.data_filename
    bodypart = args.bodypart
    min_likelihood = args.min_likelihood
    first_sample = args.first_sample
    number_samples = args.number_samples
    sample_rate = args.sample_rate
    colorscale = args.colorscale
    fig_filename_pattern = args.fig_filename_pattern


    df = pd.read_hdf(data_filename)
    level0_values =  df.columns.levels[0]
    if len(level0_values) >1:
        raise ValueError(f"More than one level0 values in {data_filename}")
    data = df[level0_values[0]][bodypart]
    low_like_indices = data["likelihood"] < min_likelihood
    data.loc[low_like_indices, "x"] = np.nan
    data.loc[low_like_indices, "y"] = np.nan
    data = data.T
    data = data.iloc[:2].to_numpy()
    data = data[:, first_sample:(first_sample+number_samples)]
    N = data.shape[1]
    dt = 1.0/sample_rate
    time = np.arange(first_sample*dt, (first_sample+number_samples)*dt, dt)

    fig = go.Figure()
    trace_mes = go.Scatter(x=data[0, :], y=data[1, :],
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
