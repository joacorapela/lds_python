import sys
import argparse
import configparser
import numpy as np
import plotly.graph_objects as go

sys.path.append("../../src")
import inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", type=int,
                        help="simulation result number")
    parser.add_argument("--noise_intensity_min", type=float, default=0.1,
                        help="minimum value for noise_intensity")
    parser.add_argument("--noise_intensity_max", type=float, default=2.1,
                        help="maximum value for noise_intensity")
    parser.add_argument("--noise_intensity_step", type=float, default=0.1,
                        help="step value for noise_intensity")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--filtering_params_filename", type=str,
                        default="", help="filtering parameters filename")
    parser.add_argument("--filtering_params_section", type=str,
                        default="initial_params",
                        help=("section of ini file containing the filtering "
                              "params"))
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../../figures/{:08d}_logLike_noise_intensity_sweep.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    noise_intensity_min = args.noise_intensity_min
    noise_intensity_max = args.noise_intensity_max
    noise_intensity_step = args.noise_intensity_step
    simRes_filename_pattern = args.simRes_filename_pattern
    filtering_params_filename = args.filtering_params_filename
    filtering_params_section = args.filtering_params_section
    fig_filename_pattern = args.fig_filename_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number, "npz")
    simRes = np.load(simRes_filename)

    if len(filtering_params_filename) > 0:
        smoothing_params = configparser.ConfigParser()
        smoothing_params.read(filtering_params_filename)
        pos_x0 = float(smoothing_params[filtering_params_section]["pos_x0"])
        pos_y0 = float(smoothing_params[filtering_params_section]["pos_y0"])
        vel_x0 = float(smoothing_params[filtering_params_section]["vel_x0"])
        vel_y0 = float(smoothing_params[filtering_params_section]["vel_x0"])
        acc_x0 = float(smoothing_params[filtering_params_section]["acc_x0"])
        acc_y0 = float(smoothing_params[filtering_params_section]["acc_x0"])
        # noise_intensityx = float(smoothing_params[filtering_params_section]["noise_intensityx"])
        # noise_intensityy = float(smoothing_params[filtering_params_section]["noise_intensityy"])
        sigma_x = float(smoothing_params[filtering_params_section]["sigma_x"])
        sigma_y = float(smoothing_params[filtering_params_section]["sigma_y"])
        sqrt_diag_V0_value = float(smoothing_params[filtering_params_section]
                                                   ["sqrt_diag_V0_value"])

        m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                      dtype=np.double)
        V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
        R = np.diag([sigma_x**2, sigma_y**2])
    else:
        # noise_intensityx = simRes["noise_intensityx"]
        # noise_intensityy = simRes["noise_intensityy"]
        # noise_intensityx = simRes["noise_intensity"]
        # noise_intensityy = simRes["noise_intensity"]
        m0 = simRes["m0"]
        V0 = simRes["V0"]
        R = simRes["R"]
        Qe = simRes["Qe"]

    noise_intensities = np.arange(noise_intensity_min, noise_intensity_max, noise_intensity_step)
    log_likes = np.empty(len(noise_intensities))
    for i, noise_intensity in enumerate(noise_intensities):
        Q = Qe * noise_intensity
        filterRes = inference.filterLDS_SS_withMissingValues_np(
            y=simRes["y"], B=simRes["B"], Q=Q, m0=m0, V0=V0, Z=simRes["Z"],
            R=R)
        log_likes[i] = filterRes["logLike"]
        print(f"log likelihood for noise_intensity={noise_intensity:.02f}: {log_likes[i]}")
    fig = go.Figure()
    trace = go.Scatter(x=noise_intensities, y=log_likes, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_layout(xaxis_title=r"$Noise Intensity$",
                      yaxis_title="Log Likelihood")
    fig.write_image(fig_filename_pattern.format(simRes_number, "png"))
    fig.write_html(fig_filename_pattern.format(simRes_number, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
