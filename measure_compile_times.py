from ninjawrap.gen_build import HOME, ENZYME_MLIR_OPT

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import argparse
import pathlib
import subprocess
import timeit
import sys

from plotting import plot_compile_times

BENCH_BUILD_DIR = pathlib.Path(HOME) / "build"
gpu_ir_files = {
    "XSBench": {
        "WP": "whole_program/xsbench/Simulation.private.mlir",
        "FS": "relative/xsbench/Simulation.private.mlir",
    },
    "LULESH": {
        "WP": "whole_program/lulesh/lulesh.private.mlir",
        "FS": "relative/lulesh/lulesh.private.mlir",
    },
    "RSBench": {
        "WP": "whole_program/rsbench/simulation.private.mlir",
        "FS": "relative/rsbench/simulation.private.mlir",
    },
    "LBM": {
        "WP": "whole_program/lbm/lbm.mlir",
        "FS": "relative/lbm/lbm.mlir",
    },
}

cpu_ir_files = {
    "Hand": {
        "WP": "whole_program/hand/hand.private.mlir",
        "FS": "relative/hand/hand.private.mlir",
    },
    "BUDE": {
        "WP": "whole_program/bude/bude.private.mlir",
        "FS": "relative/bude/bude.private.mlir",
    },
    "BA": {
        "WP": "whole_program/ba/ba.private.mlir",
        "FS": "relative/ba/ba.private.mlir",
    },
    "GMM": {
        "WP": "whole_program/gmm/gmm.private.mlir",
        "FS": "relative/gmm/gmm.private.mlir",
    },
    "LSTM": {
        "WP": "whole_program/lstm/lstm.private.mlir",
        "FS": "relative/lstm/lstm.private.mlir",
    },
}


def get_compile_time_dataframe(result_file):
    def make_dataframe():
        columns = [f"run{i+1}" for i in range(6)]
        all_benchmarks = list(gpu_ir_files.keys()) + list(cpu_ir_files.keys())
        variants = ["WP", "FS"]
        idx = pd.MultiIndex.from_product((all_benchmarks, variants))
        df = pd.DataFrame(
            data=np.zeros((len(idx), len(columns))), columns=columns, index=idx
        )
        return df

    try:
        return pd.read_csv(result_file, sep="\t", index_col=[0, 1])
    except FileNotFoundError:
        return make_dataframe()


def get_analysis_time(mlir_file: str, whole_program: bool):
    analysis_pass = "--print-activity-analysis=infer annotate"
    if not whole_program:
        analysis_pass += " relative"

    def work():
        try:
            subprocess.run(
                [ENZYME_MLIR_OPT, mlir_file, "-o", "/dev/null", analysis_pass],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(e.stdout)
            print(e.stderr)
            sys.exit(e.returncode)

    return timeit.repeat(work, repeat=6, number=1)


def collect_all(df: pd.DataFrame, args):
    for benchmark, inner_dict in (gpu_ir_files | cpu_ir_files).items():
        lbench = benchmark.lower()
        if args.benchmark != "all" and args.benchmark != lbench:
            continue
        if args.benchmark == "plot-only":
            continue

        print(f"Collecting analysis time for {benchmark}")
        for variant, ir_file in inner_dict.items():
            results = get_analysis_time(
                BENCH_BUILD_DIR / ir_file, whole_program=variant == "WP"
            )
            print(f"Variant {variant} results:", results)
            df.loc[(benchmark, variant)] = results
            df.to_csv(args.result_file, sep="\t")


def prepare_compile_df(compile_times):
    medians = compile_times.drop(columns="run1").median(axis=1).unstack(level=1)
    medians = medians[["WP", "FS"]]

    speedups = (medians["WP"] / medians.T).T
    speedups = speedups.drop(columns="WP")
    speedups.columns = ["Func. Summaries"]
    medians.columns = ["Whole Program", "Func. Summaries"]

    cpu_benchmarks = ["Hand", "BUDE", "BA", "GMM", "LSTM"]
    gpu_benchmarks = ["XSBench", "LULESH", "RSBench", "LBM"]
    cpu_compile_speedups = speedups.loc[cpu_benchmarks]
    gpu_compile_speedups = speedups.loc[gpu_benchmarks]
    return cpu_compile_speedups, gpu_compile_speedups


def main(args):
    compile_times = get_compile_time_dataframe(args.result_file)
    collect_all(compile_times, args)
    cpu_results, gpu_results = prepare_compile_df(compile_times)
    if args.print:
        print(cpu_results)
        print(gpu_results)
    plot_compile_times(cpu_results, gpu_results, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        choices=["all", "plot-only"]
        + [b.lower() for b in gpu_ir_files]
        + [b.lower() for b in cpu_ir_files],
        help="Which benchmarks to run. 'all' will run all available benchmarks, while 'plot-only' will run none and instead just generate plots using the saved results.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="compile_times.pdf",
        help="The location to save the generated plot.",
    )
    parser.add_argument(
        "--result-file",
        default="compile_times.tsv",
        help="TSV file to save (intermediate) compile-time results.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the speedup dataframe.",
    )
    main(parser.parse_args())
