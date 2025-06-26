from ninjawrap.gen_build import OPT, ENZYME_DYLIB, HOME

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


# print(collect_precisions_for_dict(cpu_ir_files))
print(collect_precisions_for_dict(gpu_ir_files))
