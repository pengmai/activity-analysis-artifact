from ninjawrap.gen_build import OPT, ENZYME_DYLIB, HOME

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pathlib
import subprocess
import re

BENCH_BUILD_DIR = pathlib.Path(HOME) / "build"
gpu_ir_files = {
    "XSBench": {
        "Enz": "informal/xsbench/Simulation.dev.ll",
        "WP": "whole_program/xsbench/Simulation.lower.ll",
        "FS": "relative/xsbench/Simulation.lower.ll",
    },
    "LULESH": {
        "Enz": "informal/lulesh/lulesh.lower.ll",
        "WP": "whole_program/lulesh/lulesh.lower.ll",
        "FS": "relative/lulesh/lulesh.lower.ll",
    },
    "RSBench": {
        "Enz": "informal/rsbench/simulation.lower.ll",
        "WP": "whole_program/rsbench/simulation.lower.ll",
        "FS": "relative/rsbench/simulation.lower.ll",
    },
    "LBM": {
        "Enz": "informal/lbm/lbm.lower.ll",
        "WP": "whole_program/lbm/lbm.lower.ll",
        "FS": "relative/lbm/lbm.lower.ll",
    },
}

cpu_ir_files = {
    "Hand": {
        "Enz": "informal/hand/hand.ll",
        "WP": "whole_program/hand/hand.lower.ll",
        "FS": "relative/hand/hand.lower.ll",
    },
    "BUDE": {
        "Enz": "informal/bude/bude.ll",
        "WP": "whole_program/bude/bude.lower.ll",
        "FS": "relative/bude/bude.lower.ll",
    },
    "BA": {
        "Enz": "informal/ba/ba.ll",
        "WP": "whole_program/ba/ba.lower.ll",
        "FS": "relative/ba/ba.lower.ll",
    },
    "GMM": {
        "Enz": "informal/gmm/gmm.ll",
        "WP": "whole_program/gmm/gmm.lower.ll",
        "FS": "relative/gmm/gmm.lower.ll",
    },
    "LSTM": {
        "Enz": "informal/lstm/lstm.ll",
        "WP": "whole_program/lstm/lstm.lower.ll",
        "FS": "relative/lstm/lstm.lower.ll",
    },
}


def get_precision(llvm_file: str):
    opt_ps = subprocess.run(
        [
            OPT,
            "-S",
            llvm_file,
            f"-load-pass-plugin={ENZYME_DYLIB}",
            "-passes=enzyme",
            "-enzyme-print-active-instructions",
            "-o",
            "/dev/null",
        ],
        check=True,
        capture_output=True,
    )
    output = subprocess.run(
        ["grep", "Running totals"], input=opt_ps.stderr, check=True, capture_output=True
    )

    # Get the most recent running total as the overall total
    last_line = output.stdout.decode("utf-8").splitlines()[-1]
    m = re.match(r"Running totals:.+\(([\d.]+)\)", last_line)
    return float(m.group(1))


def collect_precisions_for_dict(ir_files_dict):
    return {
        bench: {
            variant: get_precision(BENCH_BUILD_DIR / llvm_file)
            for variant, llvm_file in inner_dict.items()
        }
        for bench, inner_dict in ir_files_dict.items()
    }


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


if __name__ == "__main__":
    gpu_precision_results = pd.DataFrame(collect_precisions_for_dict(gpu_ir_files)).T
    cpu_precision_results = pd.DataFrame(collect_precisions_for_dict(cpu_ir_files)).T
    plot_precision(cpu_precision_results, gpu_precision_results, "activity_precision.pdf")
