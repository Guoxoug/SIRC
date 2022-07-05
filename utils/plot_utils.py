import numpy as np
from numpy.core.function_base import linspace
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
from torch.nn.functional import softmax
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
import seaborn as sns
import os
from matplotlib.lines import Line2D
from torchvision import datasets
from utils.train_utils import get_filename
from utils.ood_eval_utils import get_metric_name
sns.set_theme()


def plot_ptq_noise_histogram(
    difference: np.array, key, dataset_name, config, gaussian=True, suffix=""
):
    """"Plot and save a histogram of the noise from post training quantization.
    
    For the output logits, should be over a dataset. 
    """
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename 
    filename = get_filename(config, seed=config["seed"]) + "_" \
        + key.replace(", ", "") + "_" + dataset_name + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.hist(
        difference, 
        density=True, 
        bins=50, 
        label=key + suffix, 
        color="lightgray", 
        edgecolor="gray"
    )
    if gaussian:
        mean = difference.mean()
        std = difference.std()
        x = np.linspace(np.min(difference), np.max(difference), 1000)
        dist = scipy.stats.norm(loc=mean, scale=std)
        y = dist.pdf(x)
        ax.plot(
            x,y, color="indianred", label=f"$N(x;{mean:.2f}, {std:.2f}^2)$"
        )
    ax.set_xlim(
        (
            np.percentile(difference, 0.5),
            np.percentile(difference, 99.5)
        )
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(path)

def plot_ptq_comp_histogram(
    fp_logits, quant_logits, 
    key, dataset_name, config, 
    gaussian=True, suffix=""
):
    """Plot and save a histogram of the logit distributions before and after"""
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) + "_" \
        + key.replace(", ", "") + "_" + dataset_name + \
            "_" + "comp" + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(
        fp_logits,
        density=True,
        bins=50,
        label="floating point" + suffix,
        color="darkorange",
        alpha=.5
    )
    if gaussian:
        mean = fp_logits.mean()
        std = fp_logits.std()
        x = np.linspace(np.min(fp_logits), np.max(fp_logits), 1000)
        dist = scipy.stats.norm(loc=mean, scale=std)
        y = dist.pdf(x)
        ax.plot(
            x, y, color="darkorange", label=f"$N(x;{mean:.2f}, {std:.2f}^2)$"
        )
    ax.hist(
        quant_logits,
        density=True,
        bins=50,
        label=key + suffix,
        color="steelblue",
        alpha=.5
    )
    if gaussian:
        mean = quant_logits.mean()
        std = quant_logits.std()
        x = np.linspace(np.min(quant_logits), np.max(quant_logits), 1000)
        dist = scipy.stats.norm(loc=mean, scale=std)
        y = dist.pdf(x)
        ax.plot(
            x, y, color="steelblue", label=f"$N(x;{mean:.2f}, {std:.2f}^2)$"
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(path)


def plot_ptq_hist_over_bitwidths(
    diff_dict, dataset_name, config, 
    num_bins=500, suffix="", gaussian=True
):
    """Plot a series of histograms of logit noise PTQ for different bits."""
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) + "_" \
         + dataset_name + \
        "_" + "over_precision" + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # unpack dict into two lists
    labels, noise = zip(*diff_dict.items())

    # Computed quantities to aid plotting

    # y range
    hist_range = (np.min(noise), np.max(noise))

    # x range
    # share the same range over all  histrograms
    binned_data_sets = [
        np.histogram(
            n, range=hist_range, bins=num_bins, density=True
        )[0]
        for n in noise
    ]
    print(len(binned_data_sets))
    binned_maximums = np.max(binned_data_sets, axis=1)
    print(binned_maximums)
    x_locations = np.linspace(
        0, 
        (len(binned_maximums) -1) * np.max(binned_maximums),
        len(binned_maximums)
    )
    print(x_locations)

    # The bin_edges are the same for all of the histograms
    bin_edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)

    # Cycle through and plot each histogram
    fig, ax = plt.subplots(figsize=(6,4))
    for i, (x_loc, binned_data) in enumerate(zip(x_locations, binned_data_sets)):
        lefts = x_loc # - 0.5 * binned_data
        ax.barh(
            centers, binned_data, height=heights, left=lefts,
            color="navy", alpha=.5
        )
        if gaussian:
            mean = noise[i].mean()
            std = noise[i].std()
            
            # rotated
            y = np.linspace(np.min(noise), np.max(noise), 1000) 
            dist = scipy.stats.norm(loc=mean, scale=std)
            x = dist.pdf(y) + x_loc
            ax.plot(
                x, y, color="grey", alpha=0.5
            )


    ax.set_xticks(x_locations)
    ax.set_xticklabels(labels)

    ax.set_ylabel("$v_{ptq} - v$")
    ax.set_xlabel("precision")
    ax.set_ylim((
        np.percentile(noise[-1], 2),
        np.percentile(noise[-1], 98)
    ))

    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


def plot_ptq_stats_over_bitwidths(
    diff_dict, dataset_name, config, suffix=""
):
    """Plot the mean and std over bitwidth of ptq logit noise"""
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) + "_" \
        + dataset_name + \
        "_" + "stats_over_precision" + suffix + ".pdf"
    path = os.path.join(save_dir, filename)


    # unpack dict into two lists
    labels, noise = zip(*diff_dict.items())

    fig, ax1 = plt.subplots(figsize=(6,4))

    ax1.set_xlabel('precision')
    ax1.set_ylabel('mean')
    ax1.plot(labels, np.mean(noise, axis=-1), color="indianred")
    ax1.tick_params(axis='y', labelcolor="indianred")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "navy"
    ax2.set_ylabel('log standard deviation', color=color)  # we already handled the x-label with ax1
    ax2.plot(labels, np.log(np.std(noise, axis=-1)), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


def plot_ptq_err_swap_hist(
    delta_err_dict, 
    conf_dict,
    data_size,
    dataset_name,
    config, suffix="", num_bins=50
):
    """Plot histograms of swap confidence and change in error over bitwidth."""
    assert delta_err_dict.keys() == conf_dict.keys(), (
        "delta error and conf keys must match (should be in form afp, w4)"
    )
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) + "_" \
        + dataset_name +\
        "_" + "err_conf_swaps" + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # unpack dict into two lists
    labels, confs = zip(*conf_dict.items())
    _, err = zip(*delta_err_dict.items())
    err = np.array(err)

    # Computed quantities to aid plotting

    # y range
    hist_range = (0.0, 1.0)

    # x range
    # share the same range over all  histrograms
    binned_data_sets = [
        np.histogram(conf, range=hist_range, bins=num_bins)[0]
        for conf in confs
    ]

    binned_maximums = np.max(binned_data_sets, axis=1)

    x_locations = np.linspace(
        0,
        len(binned_maximums) * np.max(binned_maximums) * 0.825,
        len(binned_maximums)
    ) + np.max(binned_maximums) * 0.25
   

    # The bin_edges are the same for all of the histograms
    bin_edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)

    # Cycle through and plot each histogram
    fig, axes = plt.subplots(3,1, sharex=True, figsize=(6, 6))
    for x_loc, binned_data in zip(x_locations, binned_data_sets):
        lefts = x_loc - 0.5 * binned_data 
        axes[2].barh(
            centers, binned_data, height=heights, left=lefts,
            color="navy", alpha=.5
        )
    axes[2].set_xticks(x_locations)
    axes[2].set_xticklabels(labels)

    axes[2].set_ylabel("original confidence")
    axes[2].set_xlabel("precision")
    axes[2].set_ylim((
        -0.1,
        1
    ))

    # # add some text to legend
    axes[2].plot(
        [], [],linestyle="-", color="navy", alpha=.5, linewidth=2,
        label=f"distribution of swapped predictions"
    )
    axes[2].legend()

    proportion_swapped = np.array([
        len(conf_dict[k])/data_size for k in conf_dict.keys()
    ])

    axes[0].plot(
        x_locations, proportion_swapped * 100,
        color="navy", linestyle="dashed", alpha=.7, marker="o",
        label="% swapped"
    )

    axes[0].plot(
        x_locations, err * 100, 
        color="indianred", linestyle="dashed", alpha=.7, marker="o",
        label="$\Delta$% error rate"
    )

    axes[0].set_ylabel("percentage")

    axes[0].legend()

    axes[1].plot(
        x_locations, err/proportion_swapped,
        color="black", linestyle="dashed", alpha=.7, marker="o",
        label="$\Delta$% error rate / % swapped"
    )

    axes[1].set_ylabel("ratio")
    axes[1].legend()
    
    # set limits for presentation

    axes[0].set_ylim((-5, 35))
    axes[1].set_ylim((-0.05, 0.4))
    axes[2].set_ylim((0,1))
    # minor gridlines need this extra code
    for ax in axes:
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='minor', color='w', linewidth=0.075)

        # need this so that shift is seen
        ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to: {path}")


def plot_reliability_curve(
    logits, labels, legend_labels=[], file_path=None, n_bins=15, title="reliability curve",
    histogram=True, one_color=True
):
    """Plot a reliability curve given logits and labels.
    
    Also can optionally have a histogram of confidences.
    """

    # expect to iterate through a list
    if type(logits) == np.ndarray:
        logits = [logits]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    acc = np.array([
        accuracy_score(labels, data.argmax(axis=-1)) for data in logits 
    ]).mean() * 100

    ax.plot(
        [0, 1], [0, 1],
        linestyle='--', color="black", label="perfect calibration"
    )

    # put overall acc in title
    ax.set_title(f"overall accuracy: {acc:.2f}%")

    # color cycle is gradual
    n_plots = len(logits)
    color = plt.cm.magma(np.linspace(0, 0.8, n_plots))
    color[:, -1] = 0.5  # alpha
    new_cycler = cycler.cycler(color=color)
    ax.set_prop_cycle(new_cycler)
    overall_probs = []
    for i, data in enumerate(logits):
        if type(data) != np.ndarray:
            data = np.array(data)
        probs = softmax(torch.tensor(data), dim=-1).numpy()
        overall_probs.append(probs)
        fop, mpv = calibration_curve(
            (labels == np.argmax(probs, axis=-1)),
            np.max(probs, axis=-1),
            n_bins=n_bins,
            strategy="quantile"
        )
        if legend_labels:
            legend_label = legend_labels[i]
        else:
            legend_label = None
        if one_color:
            ax.plot(mpv, fop, color="indianred", alpha=0.5, label=legend_label)
        else:
            ax.plot(mpv, fop, label=legend_label)

    if one_color:
        ax.set_ylabel('accuracy', color="indianred")
    else:
        ax.set_ylabel('accuracy')
    ax.set_xlabel('confidence')
    if title is not None:
        ax.set_title(title)

    # histogram shows density of predictions wrt confidence
    if histogram:
        overall_probs = np.concatenate(overall_probs)
        confs = overall_probs.max(axis=-1)
        ax2 = ax.twinx()
        ax2.hist(
            confs,
            density=True,
            bins=20,
            alpha=0.2,
            color="navy",
            range=(0,1)
        )
        ax2.set_ylabel("density", color="navy")
        ax2.grid(False)
    ax.legend()
    fig.tight_layout()
    if file_path is not None:
        fig.savefig(file_path)
    else:
        plt.show()



def plot_cum_density_over_bitwidth(
    data_dict, dataset_name, config, metric:str, suffix=""
):
    """Plot a cumulative density plot over bitwidths."""
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
         "_" + dataset_name + f"_{metric}_cum_density_"\
         + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # reset the color cycle style
    sns.set_theme()
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    
    # change color cycle to be gradient
    n_plots = len(data_dict)
    color = plt.cm.magma(np.linspace(0, 0.8, n_plots))
    color[:, -1] = 0.5 # alpha
    new_cycler = cycler.cycler(color=color)
    ax.set_prop_cycle(new_cycler)
    for precision in data_dict:
        sorted_metric = np.sort(data_dict[precision])
        # filter so that there aren't too many points
        idx = np.round(
            np.linspace(0, len(sorted_metric) - 1, 1000)
        ).astype(int)
        sorted_metric = sorted_metric[idx]
        cum_density = np.arange(1, len(sorted_metric) + 1) / len(sorted_metric)
        ax.plot(
            sorted_metric, cum_density,
            label=f"{precision}",
        )
    
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")




def plot_comp_cum_density_over_bitwidth(
    id_dict, ood_dict, id_name, ood_name, config, metric: str, suffix=""
):
    """Plot a cumulative density plot over bitwidths."""
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        "_" + id_name +"_" + ood_name + f"_{metric}_cum_density_"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # change color cycle to be gradient
    n_plots = len(id_dict)
    color = plt.cm.magma(np.linspace(0, 0.8, n_plots))
    color[:, -1] = 0.5  # alpha


    new_cycler = cycler.cycler(color=color)
    ax.set_prop_cycle(new_cycler)

    for precision in id_dict:
        sorted_metric = np.sort(id_dict[precision])
        # filter so that there aren't too many points
        idx = np.round(
            np.linspace(0, len(sorted_metric) - 1, 1000)
        ).astype(int)
        sorted_metric = sorted_metric[idx]
        cum_density = np.arange(1, len(sorted_metric) + 1) / len(sorted_metric)
        ax.plot(
            sorted_metric, cum_density,
            label=f"{precision}",
            alpha=0.5
        )
    # reset colors
    ax.set_prop_cycle(new_cycler)

    for precision in ood_dict:
        sorted_metric = np.sort(ood_dict[precision])
        # filter so that there aren't too many points
        idx = np.round(
            np.linspace(0, len(sorted_metric) - 1, 1000)
        ).astype(int)
        sorted_metric = sorted_metric[idx]
        cum_density = np.arange(1, len(sorted_metric) + 1) / len(sorted_metric)
        ax.plot(
            sorted_metric, cum_density, linestyle="dotted",
            alpha=.5
        )
    handles, labels = ax.get_legend_handles_labels()
    id_line = Line2D([0], [0], label=id_name)
    ood_line = Line2D([0], [0], linestyle="dotted", label=ood_name)
    handles += [id_line, ood_line]

    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")

    # reset the color cycle style
    sns.set_theme()

def plot_conf_change_over_bitwidth(
    conf_dict,
    dataset_name, config,
    histogram=True,
    suffix=""
):
    """Plot average change in confidence over bitwidth.
    
    Changes are binned over the original confidence at FP.
    """
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        "_" + dataset_name  + f"conf_change_precision"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # params to use later for plotting
    bins = 20
    range = (0, 1)
    # get change in confidences for each precision
    delta_confs = {
        k: conf_dict[k] - conf_dict["afp, wfp"] 
        for k in conf_dict if k != "afp, wfp"
    }

    # reset the color cycle style
    # sns.set_theme()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if histogram:
        ax2 = ax.twinx()
        ax2.hist(
            conf_dict["afp, wfp"],
            density=True,
            range=range,
            bins=bins,
            alpha=0.2,
            color="navy"
        )
        ax2.set_ylabel("density", color="navy")
        ax2.grid(False)
     

    # change color cycle to be gradient
    n_plots = len(delta_confs)
    color = plt.cm.magma(np.linspace(0, 0.8, n_plots))
    color[:, -1] = 0.5  # alpha
    new_cycler = cycler.cycler(color=color)
    ax.set_prop_cycle(new_cycler)

    bin_centres = np.linspace(
        range[0] + (range[1] - range[0])/25/2, 
        range[1] - (range[1] - range[0])/25/2,
        bins
    )

    for precision in delta_confs:
        means, _, _ = scipy.stats.binned_statistic(
            conf_dict["afp, wfp"], delta_confs[precision],
            "mean", 
            bins=bins, range=(0,1)
        )

        ax.plot(
            bin_centres, means, label=precision, alpha = 0.5
        )

    ax.legend(loc="lower left")
    ax.grid(True, which="major")
    ax.grid(b=True, which='minor', linewidth=0.075)
    ax.minorticks_on()
    ax.set_ylim((-0.5, 0.5))
    ax.set_xlim(range)
    ax.set_xlabel("fp confidence")
    ax.set_ylabel("mean $\Delta$ confidence")
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


def plot_acc_change_over_bitwidth(
    correct_dict,
    conf_dict,
    dataset_name, config,
    histogram=True,
    suffix=""
):
    """Plot average change in confidence over bitwidth.
    
    Changes are binned over the original confidence at FP.
    """
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        "_" + dataset_name + f"_acc_change_precision"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # params to use later for plotting
    bins = 20
    range = (0, 1)
    # get change in confidences for each precision


    # dictionary should have 1 or 0 for each prediction
    delta_acc = {
        k: correct_dict[k] - correct_dict["afp, wfp"]
        for k in correct_dict if k != "afp, wfp"
    }

    # reset the color cycle style
    # sns.set_theme()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if histogram:
        ax2 = ax.twinx()
        ax2.hist(
            conf_dict["afp, wfp"],
            density=True,
            range=range,
            bins=bins,
            alpha=0.2,
            color="navy"
        )
        ax2.set_ylabel("density", color="navy")
        ax2.grid(False)

    # change color cycle to be gradient
    n_plots = len(delta_acc)
    color = plt.cm.magma(np.linspace(0, 0.8, n_plots))
    color[:, -1] = 0.5  # alpha
    new_cycler = cycler.cycler(color=color)
    ax.set_prop_cycle(new_cycler)

    bin_centres = np.linspace(
        range[0] + (range[1] - range[0])/25/2,
        range[1] - (range[1] - range[0])/25/2,
        bins
    )

    for precision in delta_acc:
        means, _, _ = scipy.stats.binned_statistic(
            conf_dict["afp, wfp"], delta_acc[precision],
            "mean",
            bins=bins, range=(0, 1)
        )

        ax.plot(
            bin_centres, means, label=precision, alpha=0.5
        )

    ax.legend(loc="lower left")
    ax.grid(True, which="major")
    ax.grid(b=True, which='minor', linewidth=0.075)
    ax.minorticks_on()
    ax.set_ylim((-0.5, 0.5))
    ax.set_xlim(range)
    ax.set_xlabel("fp confidence")
    ax.set_ylabel("mean $\Delta$ accuracy")
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


def plot_2D_unc_scatter(
    data: dict, config, suffix, 
    scales=[1,1],
    data_names=None,
    uncertainties=None, contours=None, n_samples=500,
    xlims=None, ylims=None
):
    """Plot a 2D scatter of 2 different uncertainty metrics over datasets.
    
    Contours should be a function."""

    if uncertainties is None:
        data_name = list(data.keys())[0]

        # get 1st two uncertainties
        uncertainties = [unc for unc in list(data[data_name].keys())][:2]

    # specify filename
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)



    data_names = list(data.keys()) if data_names is None else data_names


    if uncertainties is None:
        data_name = list(data.keys())[0]

        # get 1st two uncertainties
        uncertainties = [unc for unc in list(data[data_name].keys())][:2]
 
    for i, data_name in enumerate(data):
        if data_name in data_names:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            x = data[data_name][uncertainties[0]] * scales[0]
            y = data[data_name][uncertainties[1]] * scales[1]

            # randomly sample to declutter plot
            x = np.random.choice(np.array(x), size=n_samples) 
            if uncertainties[0] == "feature_norm":
                x = -x
            y = np.random.choice(np.array(y), size=n_samples)
            if i == 0:
                alpha = 1
            else:
                alpha = .3
            ax.scatter(
                x, y, label=data_name, alpha=0.5, marker="x", s=10
            )

            if xlims is not None:
                ax.set_xlim(xlims)
            if ylims is not None:
                ax.set_ylim(ylims)

    
            if contours is not None:
                # x = linspace(ax.get_xlim()[0]+1e-3 + 200, ax.get_xlim()[1], 1000)
                # y = linspace(ax.get_ylim()[0], ax.get_ylim()[1], 1000)
                # x = linspace(0.001,1000.0,1000)
                # y = linspace(0.0,5.0,1000)
                x = linspace(ax.get_xlim()[0]+5, ax.get_xlim()[1], 1000)
                y = linspace(ax.get_ylim()[0], ax.get_ylim()[1], 1000)

                print(ax.get_xlim(), ax.get_ylim())
                X, Y = np.meshgrid(x, y)
                Z = contours(X, Y)
                ax.contour(X, Y, Z, 10, extend="both")

            ax.set_xlabel(uncertainties[0])
            ax.set_ylabel(uncertainties[1])


            ax.legend()
            fig.tight_layout()

            # suffix is there for custom filename
            filename = get_filename(config, seed=config["seed"]) +  \
                f"_{data_name}_{uncertainties[0]}_{uncertainties[1]}" + f"_2D"\
                + suffix + ".pdf"
            path = os.path.join(save_dir, filename)
            fig.savefig(path)
            print(f"figure saved to:\n{path}")


def plot_unc_over_datasets(
    data: dict, config, suffix, 
    scale=1,
    data_names=None,
    uncertainty=None, 
):
    """Plot a 1D KDE of an uncertainty metric over datasets.
    
    Decision boundary contours should be a function."""

    # specify filename
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)



    data_names = list(data.keys()) if data_names is None else data_names


    df = pd.DataFrame()
    # construct DataFrame for Seaborn
    for i, data_name in enumerate(data):
        if data_name in data_names:
            dataset_df = pd.DataFrame(
                {
                    uncertainty: data[data_name][uncertainty] * scale
                }
            )
            dataset_df["dataset"] = data_name 
            df = pd.concat([df, dataset_df], ignore_index=True)

    fig = plt.figure(figsize=(6,4))
    print("plotting")
    ax = sns.kdeplot(
        data=df, 
        x=uncertainty,
        hue="dataset",
        palette="muted",
        alpha=0.2,
        common_norm=False,
        fill=True,
        # bw_adjust=.5
    )
    

    fig.tight_layout()

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_over_data_{uncertainty}"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    print(f"figure saved to:\n{path}")

def plot_2D_unc_over_datasets(
    data: dict, config, suffix, 
    scales=[1,1],
    data_names=None,
    uncertainties=None, contours=None,
):
    """Plot a 2D scatter of 2 different uncertainty metrics over datasets.
    
    Decision boundary contours should be a function."""

    # specify filename
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)



    data_names = list(data.keys()) if data_names is None else data_names


    if uncertainties is None:
        data_name = list(data.keys())[0]

        # get 1st two uncertainties
        uncertainties = [unc for unc in list(data[data_name].keys())][:2]

    df = pd.DataFrame()
    # construct DataFrame for Seaborn
    for i, data_name in enumerate(data):
        if data_name in data_names:
            dataset_df = pd.DataFrame(
                {
                    uncertainties[0]: data[data_name][uncertainties[0]] * scales[0],
                    uncertainties[1]: data[data_name][uncertainties[1]] * scales[1]
                }
            )
            dataset_df["dataset"] = data_name 
            df = pd.concat([df, dataset_df], ignore_index=True)

    fig = plt.figure(figsize=(6,4))
    print("plotting")
    ax = sns.kdeplot(
        data=df, 
        x=uncertainties[0],
        y=uncertainties[1],
        hue="dataset",
        alpha=0.7,
        levels=6,
        common_norm=False,
        palette="muted",
    )
    
    if contours is not None:
        # x = linspace(ax.get_xlim()[0]+1e-3 + 200, ax.get_xlim()[1], 1000)
        # y = linspace(ax.get_ylim()[0], ax.get_ylim()[1], 1000)
        # x = linspace(0.001,1000.0,1000)
        # y = linspace(0.0,5.0,1000)
        x = linspace(ax.get_xlim()[1]/10, ax.get_xlim()[1], 1000) 
        y = linspace(ax.get_ylim()[0], ax.get_ylim()[1], 1000)

        print(ax.get_xlim(), ax.get_ylim())
        X, Y = np.meshgrid(x, y)
        Z = contours(X, Y)
        ax.contour(X, Y, Z, 10, extend="both", alpha=0.7)

    # ax.set_xlabel(uncertainties[0])
    # ax.set_ylabel(uncertainties[1])

    fig.tight_layout()

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_over_data_{uncertainties[0]}_{uncertainties[1]}" + f"_2D"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


def plot_conf_T(
    data: dict, config, suffix, labels, preds,
    data_names=None
):
    """Plot a 1D KDE of an uncertainty metric over datasets.
    
    Decision boundary contours should be a function."""

    # specify filename
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # data has keys of each T at highest level
    data_names = list(data[1].keys()) if data_names is None else data_names
    id_data_name = config["id_dataset"]["name"]
    n = len(data)
    df = pd.DataFrame()

    correct_idx = (preds==labels).nonzero().squeeze(-1)
    incorrect_idx = (preds!=labels).nonzero().squeeze(-1)
    print(correct_idx.shape)
    print("plotting")
    fig, axes = plt.subplots(n, 1, figsize=(6, 2*n), sharex=True)
    for j, T in enumerate(data.keys()):
        for i, data_name in enumerate(data[T]):
            if data_name in data_names:

                # separate correct ID predictions from incorrect predictions
                if data_name == id_data_name:
                    dataset_df = pd.DataFrame(
                        {
                            "max prob": data[T][data_name]["confidence"][correct_idx] 
                        }
                    )
                    dataset_df["dataset"] = data_name + " (ID) - correct"
                    df = pd.concat([df, dataset_df], ignore_index=True)


                    dataset_df = pd.DataFrame(
                        {
                            "max prob": data[T][data_name]["confidence"][incorrect_idx] 
                        }
                    )
                    dataset_df["dataset"] = data_name + " (ID) - incorrect"
                    df = pd.concat([df, dataset_df], ignore_index=True)

                # OOD data
                else:
                    dataset_df = pd.DataFrame(
                        {
                            "max prob": data[T][data_name]["confidence"] 
                        }
                    )
                    dataset_df["dataset"] = data_name + " (OOD)"
                    df = pd.concat([df, dataset_df], ignore_index=True)

    
        legend = True if T == 1 else False
        sns.kdeplot(
            ax=axes[j],
            data=df,
            x="max prob",
            hue="dataset",
            palette="muted",
            alpha=0.2,
            common_norm=False,
            fill=True,
            bw_adjust=.25,
            legend=legend
        )
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        axes[j].text(0.1, 0.9, f"$T={T}$",
            transform=axes[j].transAxes, verticalalignment='top', bbox=props,
        )
        axes[j].set_ylabel("density")


    fig.tight_layout()

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_conf_over_T"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    print(f"figure saved to:\n{path}")

def plot_comp_acc(df, config, suffix=""):
    """Plot various computational costs against accuracy."""

    # specify filename
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    costs = ["FLOPS", "latency"]

    # we want pairs of the same colour
    # cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # cycle = [cycle[i//2] for i in range(len(cycle)*2)]
    # ax.set_prop_cycle(cycle)
    for cost in costs:
        if cost in df.columns:
            relative_cost = df[cost].iloc[:-1]/df[cost].iloc[-1]
            ax.plot(
                df["top1"].iloc[:-1], 
                relative_cost,
                linestyle=":",
                marker="x",
                label=cost
            )
            for i, t in enumerate(list(df["threshold"].iloc[:-1])):
                ax.annotate(
                    str(t), (df["top1"].iloc[i], relative_cost.iloc[i]),
                    fontsize="x-small", color="darkgray"
                )

    ax.scatter(
        [df["top1"].iloc[-1]], [1.0],
        marker="+",
        label=f"backbone",
        color="indianred"
    )
    ax.set_xlabel("% error rate")
    ax.set_ylabel("relative cost")
    ax.legend()
    fig.tight_layout()

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_comp_acc"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


def plot_comp_ood(
    data, config, id_data_name, ood_data_name, unc,
    suffix=""
):
    """Plot various computational costs against accuracy."""
    # print(data[ood_data_name].keys())
    # specify filename
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(
        [], [], 
        label=f"$t_c = {data[ood_data_name][unc]['t_c']}$", linestyle=""
    )
    ax.plot(
        data[ood_data_name][unc]["recall"], 
        data[ood_data_name][unc]["precision"],
        color="indianred", label="early exit precision"
    )

    ax.plot(
        data[ood_data_name]["backbone " + unc]["recall"], 
        data[ood_data_name]["backbone " + unc]["precision"],
        color="black", label="backbone precision"
    )

    idx = data[ood_data_name][unc]["idx"]
    ax.plot(
        data[ood_data_name][unc]["recall"][idx], 
        data[ood_data_name][unc]["FLOPS"]/data[ood_data_name]["backbone FLOPS"],
        color="indianred", linestyle=":", label="relative FLOPS"
    )

    ax.plot(
        data[ood_data_name][unc]["recall"][idx], 
        data[ood_data_name][unc]["latency"]/data[ood_data_name]["backbone latency"],
        color="indianred", linestyle="--", label="relative latency"
    )

    ax.hlines(
        [1], xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
        label="backbone relative cost", linestyle="-.", color="black" 
    )
    
    ax.set_xlabel("recall")
    ax.set_ylim([0.0, 1.1])
    ax.set_title(f"{id_data_name} - {ood_data_name} - {unc}")
    ax.legend()
    fig.tight_layout()

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_{id_data_name}_{ood_data_name}_{unc}_comp_acc"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


def plot_ood_prop(
    data, config, id_data_name, ood_data_names, unc,
    suffix=""
):
    """Plot various computational costs against accuracy."""
    # print(data[ood_data_name].keys())
    # specify filename
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.plot(
        [], [], 
        label=f"$t_c = {data[ood_data_names[0]][0.0][unc]['t_c']}$", linestyle=""
    )

    ax2.plot(
        [], [], 
        label=f"$t_c = {data[ood_data_names[0]][0.0][unc]['t_c']}$", linestyle=""
    )

    for ood_data_name in ood_data_names:
        ood_data = data[ood_data_name]
        FLOPS = []
        lat = []
        props = []
        for prop in ood_data:
            if "backbone" in str(prop):
                break
            props.append(prop)
            lat.append(ood_data[prop][unc]["latency"])
            FLOPS.append(ood_data[prop][unc]["FLOPS"])

        FLOPS, lat, props = np.array(FLOPS), np.array(lat), np.array(props)


        ax1.plot(
            props, 
            FLOPS/data[ood_data_name]["backbone FLOPS"],
            linestyle=":", label=ood_data_name
        )

        ax2.plot(
            props, 
            lat/data[ood_data_name]["backbone latency"],
            linestyle="--", label=ood_data_name
        )

    ax1.hlines(
        [1], xmin=ax1.get_xlim()[0], xmax=ax1.get_xlim()[1],
        label="backbone relative cost", linestyle="-.", color="black" 
    )

    ax2.hlines(
        [1], xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1],
        label="backbone relative cost", linestyle="-.", color="black" 
    )
    
    
    ax1.set_xlabel("proportion of OOD data")
    ax2.set_xlabel("proportion of OOD data")
    ax1.set_ylim([0.0, 1.1])
    ax1.set_title(f"{id_data_name} - FLOPS")
    ax1.legend()
    fig1.tight_layout()

    ax2.set_ylim([0.0, 1.1])
    ax2.set_title(f"{id_data_name} - latency")
    ax2.legend()
    fig2.tight_layout()

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_{id_data_name}_over_data_prop_ood_flops"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig1.savefig(path)
    print(f"figure saved to:\n{path}")

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_{id_data_name}_over_data_prop_ood_latency"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig2.savefig(path)
    print(f"figure saved to:\n{path}")




def plot_features_weights(features, W: torch.Tensor, label_list, b=None):
    """Plot the features, alongside the closest weight vector."""
    if b == None:
        b = torch.zeros(W.shape[0])

    logits = W @ features + b
    pred = logits.argmax()

    # weight vector associated with top logit
    w = W[pred]
    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    ent = -(probs*log_probs).sum()
    energy = torch.logsumexp(logits, dim=-1)

    fig, axes = plt.subplots(4, 1, figsize=(6, 6))
    x = np.arange(len(w))
    axes[0].set_title(f"prediction: {label_list[pred]}")
    axes[0].bar(x, w)
    axes[0].set_xticks([])
    # for minor ticks
    axes[0].set_xticks([], minor=True)

    feature_norm = torch.sum(features)
    axes[1].plot([], [], ' ', label=f"$||z||_1 = {feature_norm:.2f}$")
    axes[1].bar(x, features)
    axes[1].set_xticks([])
    # for minor ticks
    axes[1].set_xticks([], minor=True)
    axes[1].legend()

    axes[2].bar(x, features * w)
    axes[2].set_xticks([])
    # for minor ticks
    axes[2].set_xticks([], minor=True)

    axes[3].plot([], [], ' ', label=f"$H = {ent:.2f}$")
    axes[3].plot([], [], ' ', label=f"energy$ = {energy:.2f}$")
    axes[3].bar(np.arange(len(logits)), logits)
    axes[3].set_xticks([])
    # for minor ticks
    axes[3].set_xticks([], minor=True)
    axes[3].legend()
    fig.tight_layout()
    plt.show()



def plot_sc_ood_ratio(
    sc_perf, ratio, config,  eval_metric, 
    suffix="", uncs=None, var_param="alpha"
):
    """Plot a cumulative density plot over bitwidths."""
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
         "_" f"_sc_ood_ratio_{eval_metric}"\
         + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # reset the color cycle style
    sns.set_theme()
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    
    for metric in sc_perf:
        if uncs is not None and metric not in uncs:
            continue
        unc_perf = torch.tensor(sc_perf[metric])
        metric = get_metric_name(metric).replace("\\b", "")
        if unc_perf.shape[0] > 1:
            mean = unc_perf.mean(axis=0)
            std = unc_perf.std(axis=0)
            ax.plot(ratio, mean, label=f"{metric}")
            ax.fill_between(ratio, mean-1*std, mean + 1*std, alpha=0.15)
        else:
            ax.plot(ratio, unc_perf[0], label=f"{metric}")
    ax.set_xlabel("ratio of OOD:ID data")
    ax.set_ylabel(eval_metric)
    ax.legend()
    ax.grid(b=True, which='both', color="w")
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


def plot_uncs_conditional(
    uncs,
    unc_names,
    config,
    unc_range = [0,1], 
    suffix=""
):
    """Plot histograms of unc2|unc1.
    """
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_{unc_names[0]}_{unc_names[1]}"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # params to use later for plotting
    bins = 20

    # the two uncertainties
    unc1, unc2 = np.array(uncs[unc_names[0]]), np.array(uncs[unc_names[1]])


    # reset the color cycle style
    # sns.set_theme()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    H, xedges, yedges = np.histogram2d(
        unc1, unc2, range=unc_range,density=True, bins=bins
    )

    H = np.stack(
        [H[i]/H[i].sum() if H[i].sum() > 0 else H[i] for i in range(len(H))]
    )
    # H = np.log(H)
    H = H.T # for display purposes
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H, cmap="Blues")
    ax.set_xlabel(unc_names[0])
    ax.set_ylabel(unc_names[1])
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")

def plot_uncs_conditional_together(
    uncs_list,
    data_names,
    unc_names,
    config,
    unc_range = [0,1], 
    suffix=""
):
    """Plot histograms of unc2|unc1.
    """
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_{unc_names[0]}_{unc_names[1]}"\
        + suffix + "_tog.pdf"
    path = os.path.join(save_dir, filename)

    # params to use later for plotting
    bins = 20
    n = len(data_names)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4*n))
    # the two uncertainties
    for i, ax in enumerate(axes):
        unc1= np.array(uncs_list[i][unc_names[0]])
        unc2= np.array(uncs_list[i][unc_names[1]])


        H, xedges, yedges = np.histogram2d(
            unc1, unc2, range=unc_range,density=True, bins=bins
        )

        H = np.stack(
            [H[i]/H[i].sum() if H[i].sum() > 0 else H[i] for i in range(len(H))]
        )
        # H = np.log(H)
        H = H.T # for display purposes
        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, H, cmap="Blues")
        ax.set_xlabel(unc_names[0])
        ax.set_ylabel(unc_names[1])
        ax.set_title(data_names[i].replace("_", " "))
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")