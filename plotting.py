import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_colormap():
    # colorblind-friendly color map taken from https://personal.sron.nl/~pault/
    # rgb_colors = [[68,119,170], [102,204,238], [34,136,51], [204,187,68], [238,102,119], [170,51,119], [187,187,187]]
    # re-ordered colors to group the whole-program/func. summaries variants together
    rgb_colors = [
        [238, 102, 119],
        [68, 119, 170],
        [102, 204, 238],
        [204, 187, 68],
        [34, 136, 51],
        [238, 102, 119],
        [170, 51, 119],
        [187, 187, 187],
    ]
    rgba_colors = [x + [256] for x in rgb_colors]
    rgba_colors_norm = np.vstack(np.array([np.array(x) / 256.0 for x in rgba_colors]))
    colormap = matplotlib.colors.ListedColormap(rgba_colors_norm)
    return colormap


def plot_p(p, ax, cmap):
    """Plot analysis precisions"""
    ez = p.reset_index()["Enz"] * 100.0
    wp = p.reset_index()["WP"] * 100.0
    fs = p.reset_index()["FS"] * 100.0
    width = 0.5
    offset = 0.03
    ez_xs = np.arange(len(wp)) * 3 - width - offset
    wp_xs = np.arange(len(wp)) * 3
    fs_xs = np.arange(len(wp)) * 3 + width + offset
    ax.bar(
        ez_xs,
        ez,
        zorder=3,
        width=width,
        edgecolor="#444",
        label="Informal",
        color=cmap(0),
    )
    ax.bar(
        wp_xs,
        wp,
        zorder=3,
        width=width,
        edgecolor="#444",
        label="Whole Program",
        color=cmap(1),
    )
    ax.bar(
        fs_xs,
        fs,
        zorder=3,
        width=width,
        edgecolor="#444",
        label="Func. Summaries",
        color=cmap(2),
    )
    ax.grid(axis="y", zorder=1)
    ax.set_yticks(np.arange(0, 101, 25), np.arange(0, 101, 25))
    ax.set_xticks(np.arange(len(p.index)) * 3.0, p.index)


def plot_precision(cpu_precision_results, gpu_precision_results, output: str):
    (fig, (ax1, ax2)) = plt.subplots(ncols=2, width_ratios=[0.55, 0.45])
    ax1.set_ylabel("Inactive Instructions (%)")
    ax1.set_title("CPU")
    ax2.set_title("GPU")
    fig.set_size_inches(9, 2.2)
    cmap = get_colormap()
    plot_p(cpu_precision_results, ax1, cmap)
    plot_p(gpu_precision_results, ax2, cmap)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncols=3,
        frameon=False,
    )
    fig.tight_layout()
    plt.savefig(output, bbox_inches="tight")


def plot_speedups_on_axis(ax, df: pd.DataFrame, cmap, yticks):
    xs = np.arange(df.shape[0]) * df.shape[1]
    width = 0.5
    offset = 0.03
    for i, c in enumerate(df.columns):
        ax.bar(
            xs + width * float(i) + (i - 1) * offset,
            df[c],
            width=width,
            zorder=3,
            edgecolor="#444",
            label=c,
            color=cmap(i),
        )

    plot_baseline_in_front = False
    ax.plot(
        [-1, xs[-1] + 3],
        [1.0] * 2,
        color="black",
        linewidth=2,
        zorder=3 if plot_baseline_in_front else 2,
        label="No Activity Analysis",
    )
    ax.grid(axis="y", zorder=1)
    ax.set_xticks(xs + width * ((df.shape[1] - 1) / 2.0), df.index)
    ax.set_xlim(-1, df.shape[0] * df.shape[1] - 1.5)

    ax.set_yscale("log")
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(yticks)


def plot_speedups(cpu_results, gpu_results, output: str):
    (fig, (ax1, ax2)) = plt.subplots(ncols=2, width_ratios=[0.55, 0.45])
    fig.set_size_inches(9, 2.3)
    ax1.set_title("CPU")
    ax1.set_ylabel("Speedup (log scale)")
    ax2.set_title("GPU")
    cmap = get_colormap()
    plot_speedups_on_axis(
        ax1, cpu_results, cmap, yticks=[0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    )
    gpu_ticks = [0.0, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    plot_speedups_on_axis(ax2, gpu_results, cmap, yticks=gpu_ticks)
    ax1.set_ylim(0.95, 2.1)
    ax2.set_ylim(0.91, 3.6)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncols=5,
        frameon=False,
    )
    fig.tight_layout()
    plt.savefig(output, bbox_inches="tight")
