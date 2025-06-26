from ninjawrap.gen_build import HOME

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import json
import re
import subprocess

from plotting import plot_speedups

BENCH_BUILD_DIR = HOME / "build"
RESULTS_FILE = "runtimes.tsv"
GPU_BENCHMARKS = ["XSBench", "LULESH", "RSBench", "LBM"]
CPU_BENCHMARKS = ["Hand", "BUDE", "BA", "GMM", "LSTM"]

gpu_runflags = {
    "XSBench": (["-m", "event", "-k", "0", "-l", "17000000"], "Runtime"),
    "LULESH": (["-s", "60"], "Elapsed"),
    "RSBench": (["-m", "event", "-l", "10200"], "Runtime"),
    "LBM": (
        [
            "-i",
            HOME
            / "gpu"
            / "lbm"
            / "datasets"
            / "lbm"
            / "short"
            / "input"
            / "120_120_150_ldc.of",
            "-o",
            "ref.dat",
            "--",
            "150",
        ],
        "Kernel   ",
    ),
    # BUDE is not a GPU benchmark, but its output is formatted similarly to one (and must be filtered by grep)
    "BUDE": (
        ["-n", "4096", "--deck", HOME / "cpu" / "bude" / "data" / "bm1"],
        "Average time",
    ),
}

cpu_runflags = {
    "Hand": [
        HOME / "cpu" / "hand" / "data" / "model",
        HOME / "cpu" / "hand" / "data" / "hand1_t26_c100.txt",
    ],
    "BA": [HOME / "cpu" / "ba" / "data" / "ba10_n1197_m126327_p563734.txt"],
    "GMM": [HOME / "cpu" / "gmm" / "data" / "gmm_d64_K50.txt"],
    "LSTM": [HOME / "cpu" / "lstm" / "data" / "lstm_l4_c4096.txt"],
}

all_benchmarks = list(gpu_runflags.keys() | cpu_runflags.keys())
VARIANTS = ["all_active", "informal", "whole_program", "relative", "gdce"]


def get_runtime_dataframe(result_file):
    def make_dataframe():
        columns = [f"run{i+1}" for i in range(6)]
        idx = pd.MultiIndex.from_product((all_benchmarks, VARIANTS))
        df = pd.DataFrame(
            data=np.zeros((len(idx), len(columns))), columns=columns, index=idx
        )
        return df

    try:
        return pd.read_csv(result_file, index_col=[0, 1], sep="\t")
    except FileNotFoundError:
        return make_dataframe()


# Map the lowercase benchmark names to the capitalized benchmark names.
capitalized = {key.lower(): key for key in all_benchmarks}


def collect_cpu(bench, key):
    flags = cpu_runflags[key]
    bench_ps = subprocess.run([bench] + flags, capture_output=True)
    return json.loads(bench_ps.stdout.decode("utf-8"))


def run_gpu_benchmark(bench, key):
    flags, timestr = gpu_runflags[key]
    bench_ps = subprocess.run([bench] + flags, capture_output=True)
    grep_ps = subprocess.run(
        ["grep", timestr], input=bench_ps.stdout, check=True, capture_output=True
    )
    # Extract the elapsed kernel time as reported by the benchmark
    m = re.match(r".+[:=]\s+([\d.]+).*", grep_ps.stdout.decode("utf-8"))
    elapsed = float(m.group(1))
    return elapsed


def collect_gpu(bench, key):
    return [run_gpu_benchmark(bench, key) for _ in range(6)]


def collect_all(df: pd.DataFrame, args):
    for benchmark in all_benchmarks:
        lbench = benchmark.lower()
        if args.benchmark != "all" and args.benchmark != lbench:
            continue
        if args.benchmark == "plot-only":
            continue

        gpu_benchmark = benchmark in gpu_runflags.keys()
        bench_key = benchmark

        for variant in VARIANTS:
            if args.variant != "all" and args.variant != variant:
                continue

            print(f"Collecting {benchmark} variant {variant}")
            bench_bin = BENCH_BUILD_DIR / variant / lbench / lbench
            if gpu_benchmark:
                results = collect_gpu(bench_bin, bench_key)
            else:
                results = collect_cpu(bench_bin, bench_key)

            df.loc[benchmark, variant] = results
            df.to_csv(args.result_file, sep="\t")


def prepare_speedups(runtimes: pd.DataFrame):
    # Discard the first run as a warmup and take the median of the remaining 5
    medians = runtimes.drop(columns="run1").median(axis=1).unstack(level=1)
    speedups = (medians["all_active"] / medians.T).T

    # Sort and rename variants to match the figure
    speedups = speedups[VARIANTS].drop(columns="all_active")
    speedups.columns = [
        "Informal",
        "Whole Program",
        "Func. Summaries",
        "No Activity + gDCE",
    ]
    cpu_results = speedups.loc[CPU_BENCHMARKS]
    gpu_results = speedups.loc[GPU_BENCHMARKS]
    return cpu_results, gpu_results


import argparse


def main(args):
    runtimes = get_runtime_dataframe(args.result_file)
    collect_all(runtimes, args)
    cpu_results, gpu_results = prepare_speedups(runtimes)
    plot_speedups(cpu_results, gpu_results, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        choices=["all", "plot-only"] + [b.lower() for b in all_benchmarks],
        help="Which benchmarks to run. 'all' will run all available benchmarks, while 'plot-only' will run none and instead just generate plots using the saved results.",
    )
    parser.add_argument(
        "--variant",
        choices=["all"] + VARIANTS,
        default="all",
    )
    parser.add_argument(
        "--result-file",
        default="runtimes.tsv",
        help="TSV file to save (intermediate) run-time results.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="runtimes.pdf",
        help="The location to save the generated plot.",
    )
    main(parser.parse_args())
