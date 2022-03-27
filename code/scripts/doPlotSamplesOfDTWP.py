
import sys
import argparse
import numpy as np
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_a",
                        help="standard deviation of acceleration discrete time Wiener process",
                        type=float, default=1)
    parser.add_argument("--N", help="number of samples of DTWP", type=int,
                        default=1000)
    parser.add_argument("--title_pattern", help="title pattern", type=str,
                        default="sigma_a={:.02f}")
    args = parser.parse_args()

    sigma_a = args.sigma_a
    N = args.N
    title_pattern = args.title_pattern

    w = np.random.normal(loc=0, scale=sigma_a, size=N)
    acceleration = np.zeros(shape=N)
    for n in range(1, N):
        acceleration[n] = acceleration[n-1] + w[n]

    fig = go.Figure()
    trace = go.Scatter(x=np.arange(N), y=acceleration)
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="sample number", yaxis_title="acceleration",
                      title=title_pattern.format(sigma_a))
    fig.show()

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
