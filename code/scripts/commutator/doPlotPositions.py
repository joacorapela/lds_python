import sys
import pdb
import numpy as np
import pandas as pd
import argparse
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/commutator/tracking.csv")
    parser.add_argument("--bodypart", type=str, help="body part to track",
                        default="snout")
    parser.add_argument("--min_likelihood", type=float,
                        help=("positions with likelihood less than "
                              "min_likelihood are set to np.nan"),
                        default=0.00)
    parser.add_argument("--first_sample", type=int, help="first sample",
                        default=0)
    parser.add_argument("--number_samples", type=int,
                        help="number of samples to smooth", default=10000)
    parser.add_argument("--sample_rate", type=float, help="sample rate (Hz)",
                        default=30)
    parser.add_argument("--colorscale", type=str,
                        help="colorscale name", default="Rainbow")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern",
                        default=("../../../figures/commutator/positions_laurence_"
                                 "from{:d}_numberSamples{:d}.{:s}"))

    args = parser.parse_args()

    data_filename = args.data_filename
    bodypart = args.bodypart
    min_likelihood = args.min_likelihood
    first_sample = args.first_sample
    number_samples = args.number_samples
    sample_rate = args.sample_rate
    colorscale = args.colorscale
    fig_filename_pattern = args.fig_filename_pattern


    df = pd.read_csv(data_filename, header=None)
    column_start_index = None
    for i in range(0, df.shape[1], 3):
        if df.iloc[0,i] == bodypart:
            column_start_index = i+1
    if column_start_index == None:
        raise ValueError(f"{bodypart} not found in {data_filename}")
    data = df.iloc[:, range(column_start_index, column_start_index+3)]
    data.columns = ["x", "y", "likelihood"]
    low_like_indices = data["likelihood"] < min_likelihood
    data.loc[low_like_indices, "x"] = np.nan
    data.loc[low_like_indices, "y"] = np.nan
    data = data.T
    data = data.to_numpy()
    data = data[:, first_sample:(first_sample+number_samples)]

    N = data.shape[1]
    dt = 1.0/sample_rate
    time = np.arange(first_sample*dt, (first_sample+number_samples)*dt, dt)
    likelihood = data[2, :]
    customdata = np.stack((time, likelihood), axis=-1)

    fig = go.Figure()
    trace_mes = go.Scatter(x=data[0, :], y=data[1, :],
                           mode="markers",
                           marker={"color": time,
                                   "colorscale": colorscale,
                                   "colorbar": {"title": "Time"},
                                  },
                           customdata=customdata,
                           hovertemplate=("<b>x</b>: %{x:.3f}<br>"
                                          "<b>y</b>: %{y:.3f}<br>"
                                          "<b>time</b>: "
                                          "%{customdata[0]:.3f} sec<br>"
                                          "<b>likelihood</b>: "
                                          "%{customdata[1]:.3f}"),
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
