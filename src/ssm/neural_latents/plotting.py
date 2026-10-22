import math
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import webcolors

import ssm.neural_latents.utils


def get2DPlotLatentsAcrossTrials(
        trials_times, means, covs, bins_centers, Z, trials_ids,
        orthonormalize=True,
        latents_to_plot=[0, 1],
        align_event_times=None,
        epoch_times=None,
        marked_events_times=None,
        marked_events_colors=None,
        marked_events_markers=None,
        marked_events_labels=None,
        marked_size=10,
        line_width=2,
        trials_annotations=None,
        trials_colors=None, default_trial_color="gray",
        xyLabelsPattern="Latent {:d}", title=""):
    # trials_times[r] \in trial_n_samples
    # latents_means \in n_latents x 1 x n_samples
    # latents_covs \in n_latents x n_latents x n_samples

    if orthonormalize:
        if Z is None:
            raise ValueError("If orthonormalize=True then Z should be "
                             "specified as a parameter")
        means, covs = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
            means, covs, Z)

    # eo_latents_means[r] \in n_latents x 1 x trial_n_samples
    means = epochLatents(latents=means, trials_times=trials_times,
                         bins_centers=bins_centers)
    n_trials = len(means)


    hover_texts = [["Trial: {:02d}<br>Time: {:f}".format(trial_id, trial_time)
                    for i, trial_time in enumerate(trials_times[r])]
                   for r, trial_id in enumerate(trials_ids)]
    for r in range(n_trials):
        n_times = len(trials_times[r])
        an_annotation = ""
        if trials_annotations is not None:
            for trial_annotation_key in trials_annotations:
                an_annotation += "<br>{:s}: {}".format(
                    trial_annotation_key,
                    trials_annotations[trial_annotation_key][r])
            for i in range(n_times):
                hover_texts[r][i] = hover_texts[r][i] + an_annotation

    if trials_colors is None:
        trials_colors = px.colors.qualitative.Plotly
    num_colors = len(trials_colors)

    fig = go.Figure()
    for r in range(n_trials):
        latent_color = trials_colors[r % num_colors]

        if epoch_times is not None:
            legend = "trial {:02d}, eTime {:02f}".format(trials_ids[r],
                                                         epoch_times[r])
        else:
            legend = "trial {:02d}".format(trials_ids[r])

        trace_latent_mean = go.Scatter(
            x=means[r][latents_to_plot[0], 0, :],
            y=means[r][latents_to_plot[1], 0, :],
            mode="lines+markers",
            line=dict(color=latent_color, width=line_width),
            name=legend,
            legendgroup="trial{:02d}".format(trials_ids[r]),
            showlegend=True,
            hoverinfo="text",
            text=hover_texts[r],
        )
        fig.add_trace(trace_latent_mean)

        if marked_events_times is not None and \
           marked_events_colors is not None and \
           marked_events_markers is not None and \
           marked_events_labels is not None and \
           align_event_times is not None:
            n_marked_events = len(marked_events_times[r])
            marked_events_times_centered = [
                marked_events_times[r][i]-align_event_times[r]
                for i in range(len(marked_events_times[r]))
            ]
            for i in range(n_marked_events):
                if not math.isnan(marked_events_times_centered[i]):
                    marked_index = np.argmin(np.abs(
                        trials_times[r]-marked_events_times_centered[i]))

                    trace_marker = go.Scatter(
                        x=[means[r][latents_to_plot[0], 0, marked_index]],
                        y=[means[r][latents_to_plot[1], 0, marked_index]],
                        marker=dict(color=marked_events_colors[r][i],
                                    symbol=marked_events_markers[r][i],
                                    size=marked_size),
                        customdata=[[trials_ids[r],marked_events_times[r][i],marked_events_labels[r][i]]],
                        hovertemplate="x=%{x}<br>" + "y=%{y}<br>" + "trial=%{customdata[0]}<br>" + "time=%{customdata[1]} sec<br>" + "event=%{customdata[2]}",
                        mode="markers",
                        legendgroup="trial{:02d}".format(trials_ids[r]),
                        showlegend=False)
                    fig.add_trace(trace_marker)

    fig.update_xaxes(title=xyLabelsPattern.format(latents_to_plot[0]))
    fig.update_yaxes(title=xyLabelsPattern.format(latents_to_plot[1]))
    return fig


def get3DPlotLatentsAcrossTrials(
        trials_times, means, covs, bins_centers, Z, trials_ids,
        orthonormalize=True,
        latents_to_plot=[0, 1, 2],
        align_event_times=None,
        epoch_times=None,
        events_names=None,
        marked_events_times=None,
        marked_events_colors=None,
        marked_events_markers=None,
        marked_events_labels=None,
        marked_size=3,
        trials_annotations=None,
        trials_colors=None, default_trial_color="gray",
        xyzLabelsPattern="Latent {:d}", title=""):

    if orthonormalize:
        if Z is None:
            raise ValueError("If orthonormalize=True then Z should be "
                             "specified as a parameter")
        means, _ = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
            means, covs, Z)

    # eo_latents_means[r] \in n_latents x 1 x trial_n_samples
    means = epochLatents(latents=means, trials_times=trials_times,
                         bins_centers=bins_centers)
    n_trials = len(means)

    hover_texts = [["Trial: {:02d}<br>Time: {:f}".format(trial_id, trial_time)
                    for i, trial_time in enumerate(trials_times[r])]
                   for r, trial_id in enumerate(trials_ids)]
    for r in range(n_trials):
        n_times = len(trials_times)
        an_annotation = ""
        if trials_annotations is not None:
            for trial_annotation_key in trials_annotations:
                an_annotation += "<br>{:s}: {}".format(
                    trial_annotation_key,
                    trials_annotations[trial_annotation_key][r])
            for i in range(n_times):
                hover_texts[r][i] = hover_texts[r][i] + an_annotation

    if trials_colors is None:
        trials_colors = px.colors.qualitative.Plotly
    num_colors = len(trials_colors)

    fig = go.Figure()
    for r in range(n_trials):
        latent_color = trials_colors[r % num_colors]

        if epoch_times is not None:
            legend = "trial {:02d}, eTime {:02f}".format(trials_ids[r],
                                                         epoch_times[r])
        else:
            legend = "trial {:02d}".format(trials_ids[r])

        trace_latent_mean = go.Scatter3d(
            x=means[r][latents_to_plot[0], 0, :],
            y=means[r][latents_to_plot[1], 0, :],
            z=means[r][latents_to_plot[2], 0, :],
            mode="lines",
            line=dict(color=latent_color, width=3),
            name=legend,
            legendgroup="trial{:02d}".format(trials_ids[r]),
            showlegend=True,
            hoverinfo="text",
            text=hover_texts[r],
        )
        fig.add_trace(trace_latent_mean)

        if marked_events_times is not None and \
           marked_events_colors is not None and \
           marked_events_markers is not None and \
           align_event_times is not None:
            n_marked_events = len(marked_events_times[r])
            marked_events_times_centered = [
                marked_events_times[r][i]-align_event_times[r]
                for i in range(len(marked_events_times[r]))
            ]
            for i in range(n_marked_events):
                if not math.isnan(marked_events_times_centered[i]):
                    marked_index = np.argmin(np.abs(
                        trials_times[r]-marked_events_times_centered[i]))

                    trace_marker = go.Scatter3d(
                        x=[means[r][latents_to_plot[0], 0, marked_index]],
                        y=[means[r][latents_to_plot[1], 0, marked_index]],
                        z=[means[r][latents_to_plot[2], 0, marked_index]],
                        marker=dict(color=marked_events_colors[r][i],
                                    symbol=marked_events_markers[r][i],
                                    size=marked_size),
                        customdata=[[trials_ids[r],marked_events_times[r][i],marked_events_labels[r][i]]],
                        hovertemplate="x=%{x}<br>" + "y=%{y}<br>" + "z=%{z}<br>" + "trial=%{customdata[0]}<br>" + "time=%{customdata[1]} sec<br>" + "event=%{customdata[2]}",
                        mode="markers",
                        legendgroup="trial{:02d}".format(trials_ids[r]),
                        showlegend=False)
                    fig.add_trace(trace_marker)

    fig.update_layout(scene=dict(
        xaxis_title=xyzLabelsPattern.format(latents_to_plot[0]),
        yaxis_title=xyzLabelsPattern.format(latents_to_plot[1]),
        zaxis_title=xyzLabelsPattern.format(latents_to_plot[2])),
    )
    return fig


def getPlotLatents(means, covs, bins_centers, Z=None,
                   orthonormalize=True, legend_pattern="latent {:d}",
                   cb_alpha=0.3, colors=[],
                   xlabel="Time (sec)", ylabel="Latent Value"):
    if orthonormalize:
        if Z is None:
            raise ValueError("If orthonormalize=True then Z should be "
                             "specified as a parameter")
        means, covs = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
            means, covs, Z)
    if len(colors) == 0:
        colors = px.colors.qualitative.Plotly
    num_colors = len(colors)
    fig = go.Figure()
    n_states = means.shape[0]
    for i in range(n_states):
        color_rgb = webcolors.hex_to_rgb(colors[i % num_colors])
        color_pattern = \
            f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},{{:f}})"
        filter_means = means[i, 0, :]
        filter_stds = np.sqrt(covs[i, i, :])
        filter_ci_upper = filter_means + 1.96*filter_stds
        filter_ci_lower = filter_means - 1.96*filter_stds

        trace = go.Scatter(
            x=bins_centers, y=filter_means,
            mode="lines+markers",
            marker={"color": color_pattern.format(1.0)},
            name=legend_pattern.format(i),
            showlegend=True,
            legendgroup=legend_pattern.format(i),
        )
        trace_cb = go.Scatter(
            x=np.concatenate([bins_centers, bins_centers[::-1]]),
            y=np.concatenate([filter_ci_upper, filter_ci_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern.format(cb_alpha),
            line=dict(color=color_pattern.format(0.0)),
            showlegend=False,
            legendgroup=legend_pattern.format(i),
        )
        fig.add_trace(trace)
        fig.add_trace(trace_cb)

    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return fig

def getPlotUnitsFiringRates(times, preIntensityMeans, preIntensitySTDs,
                                    cbAlpha=0.2,
                                    default_trial_color_pattern="rgba(128,128,128,{:f})",
                                    cb_transparency=0.3, mean_transparency=1.0,
                                    events_names=None,
                                    marked_events_times=None,
                                    marked_events_colors=None,
                                    marked_events_markers=None,
                                    marked_size=10,
                                    xlabel="Time (sec)",
                                    ylabel="Value",
                                    title=""):
    # times = times.detach().numpy()
    # preIntensityMeans = preIntensityMeans.detach().numpy()
    # preIntensitySTDs = preIntensitySTDs.detach().numpy()

    n_trials = preIntensityMeans.shape[0]
    if trials_ids is None:
        trials_ids = np.arange(n_trials)
    # pio.renderers.default = "browser"
    fig = go.Figure()
    for r in range(n_trials):
        trial_times = times[r, :, 0]
        meanToPlot = preIntensityMeans[r, :]
        stdToPlot = preIntensitySTDs[r, :]
        ciToPlot = 1.96*stdToPlot
        if trials_colors_patterns is not None:
            trial_color_pattern = trials_colors_patterns[r]
        else:
            trial_color_pattern = default_trial_color_pattern

        # pdb.set_trace()
#         import matplotlib
#         matplotlib.use('TkAgg')
#         import matplotlib.pyplot as plt
#         plt.plot(times, meanToPlot)
#         plt.show()
#         pdb.set_trace()

        x = trial_times
        y = meanToPlot
        y_upper = y + ciToPlot
        y_lower = y - ciToPlot

        traceCB = go.Scatter(
            x=np.concatenate((x, x[::-1])),
            y=np.concatenate((y_upper, y_lower[::-1])),
            fill="toself",
            fillcolor=trial_color_pattern.format(cb_transparency),
            line=dict(color=trial_color_pattern.format(0.0)),
            showlegend=False,
            # name="trial CB {:d}".format(r),
            legendgroup="trial{:02d}".format(trials_ids[r])
        )
        traceMean = go.Scatter(
            x=x,
            y=y,
            line=dict(color=trial_color_pattern.format(mean_transparency)),
            mode="lines",
            name="trial {:d}".format(r),
            legendgroup="trial{:02d}".format(trials_ids[r]),
            showlegend=True,
        )
        fig.add_trace(traceCB)
        fig.add_trace(traceMean)

        # add markers to trials
        if events_names is not None and\
           marked_events_times is not None and \
           marked_events_colors is not None and \
           marked_events_markers is not None and \
           align_event_times is not None:
            n_marked_events = len(marked_events_times[r])
            marked_events_times_centered = marked_events_times[r]-align_event_times[r]
            for i in range(n_marked_events):
                if not math.isnan(marked_events_times_centered[i]):
                    marked_index = np.argmin(np.abs(
                        times[r, :, 0]-marked_events_times_centered[i]))
                    trace_marker = go.Scatter(
                        x=[times[r, marked_index, 0]],
                        y=[meanToPlot[marked_index]],
                        marker=dict(color=marked_events_colors[r][i],
                                    symbol=marked_events_markers[r][i],
                                    size=marked_size),
                        mode="markers",
                        text=[events_names[i]],
                        hovertemplate="x=%{x}<br>" + "y=%{y}<br>" + "event=%{text}",
                        legendgroup="trial{:02d}".format(trials_ids[r]),
                        showlegend=False)
                    fig.add_trace(trace_marker)

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title_text=title)
    return fig


def epochLatents(latents, trials_times, bins_centers):
    # latents \in n_latents x 1 x T
    # bins_centers \in T
    # trials_times[r] \in trial_n_samples
    # answer[r] \in n_latents x 1 x T_r

    n_trials = len(trials_times)
    answer = [None] * n_trials
    for r in range(n_trials):
        answer[r] = latents[:, :, np.logical_and(
            trials_times[r][0] <= bins_centers,
            bins_centers <= trials_times[r][-1])]
        assert(answer[r].shape[2] == len(trials_times[r]))
    return answer
