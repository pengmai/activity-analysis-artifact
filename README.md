# Artifact for "Sound and Modular Activity Analysis for Automatic Differentiation in MLIR"

## Introduction

This artifact supports the experimental evaluation of the whole-program and modular activity analyses in the paper.

Claims:
- Activity analysis improves performance of differentiated code compared to not using activity analysis.
  - We observe a geometric mean speedup of 1.24x (up to 2x) on CPU and geomean 1.7x (up to 3x) on GPU
- The modular activity analysis is just as precise as the whole-program activity analysis.
- Running constant propagation and dead code elimination after differentiation is unable to recover the performance benefit of activity analysis.

All claims are supported by this artifact.

## Planned Changes

In response to reviewers, the revision will include an additional experiment that compares the compile/analysis times of the whole-program and modular activity analyses.
A Python script will be added that automates the measuring and plotting of this experiment, but nothing else in the artifact will change.

## Hardware Dependencies

| Component   | Minimum                 | Recommended                                                |
| ----------- | ----------------------- | ---------------------------------------------------------- |
| **CPU**     | 6 cores                 | 32 cores                                                   |
| **GPU**     | NVIDIA GPU (CUDA 12.6+) | NVIDIA GPU with 12 GiB+ VRAM (Ampere or newer, CUDA 12.6+) |
| **Storage** | 10GB                    | 10GB                                                       |

We recommend a high CPU core count due to the overhead of building LLVM.

The evaluation in the paper was performed on an Ubuntu 22.0.4 workstation with an NVIDIA GeForce RTX 3060 GPU with 12 GiB VRAM and a 3.8 GHz AMD Ryzen 5 7600 6-Core CPU with 32 GB of RAM.

## Getting Started

We assume the target machine as a valid CUDA and GPU driver installation with CUDA 12.6+.

0. **Install NVIDIA container toolkit**

The following commands may be used to install the container toolkit on Ubuntu.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
   nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For more details and instructions for other Linux distros, see [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. **Pull and build the docker container**

If needed, the source code for the artifact is available on GitHub:

```sh
git clone https://github.com/pengmai/activity-analysis-artifact.git
cd activity-analysis-artifact
```

Once obtained, the following command will build the container:

```sh
docker build -t activity-artifact .
```

> [!NOTE]
> If you run into permission issues, run the docker commands with `sudo` or add the current user to the [`docker` group](https://docs.docker.com/engine/install/linux-postinstall/).

Building the image for the first time will compile LLVM from source, which can take significant time. The AMD Ryzen 5 takes about 40 minutes to complete this process.

This process will also compile Enzyme and the benchmarks, which includes running the activity analyses.

2. **Verify the installation**

Start the built container with the following command:

```sh
docker run --name activity-ubuntu --gpus all --rm -i -t activity-artifact bash
```

On a machine with multiple GPUs, we recommend selecting a specific GPU when entering the container:

```sh
# Using GPU 1
docker run --name activity-ubuntu --gpus '"device=1"' --rm -i -t activity-artifact bash
```

In the container, you can verify that compilation was successful by running two of the faster benchmarks:
```sh
# A CPU benchmark
python measure_runtimes.py lstm --variant relative

# A GPU benchmark
python measure_runtimes.py rsbench --variant relative
```

Each of these benchmarks should take under 10 seconds to execute. Once complete, the script will output the runtime executions of each benchmark in the newly-created `runtimes.tsv` file.

```sh
cat runtimes.tsv
# Example output:

# 		run1	run2	run3	run4	run5	run6
# XSBench	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# XSBench	informal	0.0	0.0	0.0	0.0	0.0	0.0
# XSBench	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# XSBench	relative	0.0	0.0	0.0	0.0	0.0	0.0
# XSBench	gdce	0.0	0.0	0.0	0.0	0.0	0.0
# LULESH	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# LULESH	informal	0.0	0.0	0.0	0.0	0.0	0.0
# LULESH	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# LULESH	relative	0.0	0.0	0.0	0.0	0.0	0.0
# LULESH	gdce	0.0	0.0	0.0	0.0	0.0	0.0
# LBM	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# LBM	informal	0.0	0.0	0.0	0.0	0.0	0.0
# LBM	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# LBM	relative	0.0	0.0	0.0	0.0	0.0	0.0
# LBM	gdce	0.0	0.0	0.0	0.0	0.0	0.0
# GMM	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# GMM	informal	0.0	0.0	0.0	0.0	0.0	0.0
# GMM	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# GMM	relative	0.0	0.0	0.0	0.0	0.0	0.0
# GMM	gdce	0.0	0.0	0.0	0.0	0.0	0.0
# BA	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# BA	informal	0.0	0.0	0.0	0.0	0.0	0.0
# BA	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# BA	relative	0.0	0.0	0.0	0.0	0.0	0.0
# BA	gdce	0.0	0.0	0.0	0.0	0.0	0.0
# BUDE	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# BUDE	informal	0.0	0.0	0.0	0.0	0.0	0.0
# BUDE	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# BUDE	relative	0.0	0.0	0.0	0.0	0.0	0.0
# BUDE	gdce	0.0	0.0	0.0	0.0	0.0	0.0
# RSBench	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# RSBench	informal	0.0	0.0	0.0	0.0	0.0	0.0
# RSBench	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# RSBench	relative	0.276	0.267	0.266	0.265	0.266	0.268
# RSBench	gdce	0.0	0.0	0.0	0.0	0.0	0.0
# Hand	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# Hand	informal	0.0	0.0	0.0	0.0	0.0	0.0
# Hand	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# Hand	relative	0.0	0.0	0.0	0.0	0.0	0.0
# Hand	gdce	0.0	0.0	0.0	0.0	0.0	0.0
# LSTM	all_active	0.0	0.0	0.0	0.0	0.0	0.0
# LSTM	informal	0.0	0.0	0.0	0.0	0.0	0.0
# LSTM	whole_program	0.0	0.0	0.0	0.0	0.0	0.0
# LSTM	relative	124771.0	115010.0	116142.0	115250.0	116363.0	117689.0
# LSTM	gdce	0.0	0.0	0.0	0.0	0.0	0.0
```

Each benchmark is run 6 times. You should see nonzero entries for the benchmarks and variants that were run, and zeros elsewhere.
Subsequent runs of the same benchmark/variant will overwrite previous results.

> [!NOTE]
> Different benchmarks store their execution time with different units. The GPU benchmarks use seconds, while most CPU benchmarks use microseconds (except BUDE, which uses milliseconds). As each benchmark uses consistent units among its variants, this does not impact the resulting speedups.

## Step by Step Instructions

There are two main Python scripts that reproduce experiments:
* `measure_precision.py` produces Figure 11 by measuring the number of inactive instructions for each benchmark.
* `measure_runtimes.py` produces Figure 12 by measuring the execution time of the differentiated benchmark for the five variants examined in this work.

Both scripts can be run with `python measure_<...> --help` to see a full list of available arguments.

The five variants are as follows:
1. `all_active`: the baseline differentiated code that is run without activity analysis.
2. `informal`: the existing activity analysis in Enzyme that lacks a correctness argument.
3. `whole_program`: the whole-program analysis of this work.
4. `relative`: the function summary-based modular analysis of this work.
5. `gdce`: a variant where differentiation is performed without activity analysis, then a custom dead code elimination is subsequently run that removes all instructions that only (transitively) depend on gradients of inactive arguments.

### Analysis Precision

The following script invokes Enzyme to differentiate each benchmark and count the number of inactive instructions found by the three activity analysis variants (`informal`, `whole_program`, and `relative`).
```sh
python measure_precision.py [--output precision.pdf]
```

The script should take fewer than 30 seconds to complete.
The resulting plot can then be copied out the docker container for easier viewing:

```sh
docker cp activity-ubuntu:/home/evaluator/precision.pdf .
```

#### Interpretation

The precision plot should be *exactly identical* to Figure 11 in the paper, modulo the result for LBM as noted below.
The key claims are that the modular analysis (Func. Summaries) finds the same number of inactive instructions as the whole-program variant, and that both variants find the same or slightly more inactive instructions than the prior informal analysis.

> [!NOTE]
> Due to a mistake in the submitted version, the LBM benchmark precision should be 32%, not 23%, for all analysis variants. However, the claim that all three analyses produce the same number of inactive instructions for this benchmark is unchanged.

### Run-time performance

The `measure_runtimes.py` script can be used

```sh
# Run all benchmarks and variants
python measure_runtimes.py all

# Run all variants of a specific benchmark
python measure_runtimes.py rsbench

# Run a specific benchmark and variant
python measure_runtimes.py rsbench --variant all_active
```

Running all benchmarks and variants should take approximately 15-20 minutes.

## Reusability Guide
