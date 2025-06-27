# Artifact for "Sound and Modular Activity Analysis for Automatic Differentiation in MLIR"

## Introduction

This artifact supports the experimental evaluation of the whole-program and modular activity analyses in the paper.

Claims:
- Activity analysis improves performance of differentiated code compared to not using activity analysis.
  - We observe a geometric mean speedup of 1.24x (up to 2x) on CPU and geomean 1.7x (up to 3x) on GPU
- The modular activity analysis is just as precise as the whole-program activity analysis. Furthermore, both the modular and whole-program analyses find as many or more inactive instructions than Enzyme's informal analysis.
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

### Analysis Precision (Section 8.2)

The following script invokes Enzyme to differentiate each benchmark and count the number of inactive instructions found by the three activity analysis variants (`informal`, `whole_program`, and `relative`).
```sh
python measure_precision.py [--output precision.pdf] [--print]
```

The script should take fewer than 30 seconds to complete.
The resulting plot can then be copied out the docker container for easier viewing:

```sh
docker cp activity-ubuntu:/home/evaluator/precision.pdf .
```

#### Interpretation

The precision plot should be *exactly identical* to Figure 11 in the paper, modulo the result for LBM as noted below.

> [!NOTE]
> Due to a mistake in the submitted version, the LBM benchmark precision should be 32%, not 23%, for all analysis variants. However, the claim that all three analyses produce the same number of inactive instructions for this benchmark is unchanged.

> Claim: The modular activity analysis is just as precise as the whole-program activity analysis. Furthermore, both the modular and whole-program analyses find as many or more inactive instructions than Enzyme's informal analysis.

The Whole Program and Func. Summaries should find exactly the same number of inactive instructions for all benchmarks.
For all benchmarks, both analyses should find equally many or more inactive instructions than the Informal analysis.

### Run-time performance (Section 8.3)

The `measure_runtimes.py` script can be used to run all or specific benchmarks and variants.
The script merely invokes the given compiled benchmark with appropriate arguments, parses the reported kernel execution time, and serializes the results.

After each benchmark/variant, intermediate runtimes are saved to a TSV file (`runtimes.tsv` by default), and Figure 12 from the paper is produced (`runtimes.pdf` by default).
Both the TSV file and plot location can be overwritten via the CLI.
Naturally, the produced figure will only be meaningful once results for all benchmarks/variants have been collected.

```sh
# Run all benchmarks and variants
python measure_runtimes.py all

# Run all variants of a specific benchmark
python measure_runtimes.py rsbench

# Run a specific benchmark and variant
python measure_runtimes.py rsbench --variant all_active

# Run no benchmarks, just generate plots (and print the resulting speedups)
python measure_runtimes.py plot-only --print
```

Running all benchmarks and variants should take approximately 15-20 minutes.

#### Interpretation

> Claim: Activity analysis improves performance of differentiated code compared to not using activity analysis.

Results will vary depending on the specific hardware and machine noise.
- The resulting speedup plot should show significant speedups for the Hand (2x), BUDE (1.2x), XSBench (3x), LULESH (2x), and RSBench (1.2x) benchmarks, as well as a modest speedup (1.1x) for the GMM benchmark for the three activity analysis variants.
- The relative performance of the three activity analyses should be approximately the same.
- The use of activity analysis should never harm performance relative to the baseline.

> Claim: Running constant propagation and dead code elimination after differentiation is unable to recover the performance benefit of activity analysis.

With the exception of the slight speedup for GMM, the `No Activity + gDCE` variant should not improve performance relative to the baseline.

## Reusability Guide

The two activity analyses of this work are implemented as data-flow analyses using the MLIR data-flow framework.
- The whole-program analysis is implemented in `~/Enzyme/enzyme/Enzyme/MLIR/Analysis/DataFlowActivityAnalysis.{h,cpp}`
- The modular analysis is implemented in `~/Enzyme/enzyme/Enzyme/MLIR/Analysis/ActivityAnnotations.{h,cpp}`

The whole program and modular analyses are each composed of four collaborating sub-analyses (sparse forward, dense forward, sparse backward, and dense backward).
- The sparse analyses reason about data flow between operands and results, while the dense analyses reason about memory operations.
- The forward and backward analyses correspond to the forward and backward activity analyses described in the paper.

Each individual sub-analysis is relatively simple, but their collaboration enables the resulting analysis to be general enough for the benchmarks in this artifact.

Both the whole program and modular analyses are accessible via the `print-activity-analysis` pass in `enzymemlir-opt`, which includes several flags to configure their behavior.

### Running the analyses on new MLIR modules

`enzymemlir-opt` is available by default at `~/Enzyme/build/Enzyme/MLIR/enzymemlir-opt`. It can be used to annotate the activities of arbitrary MLIR modules, such as this simple example:

```mlir
func.func @callee(%val: f64, %out: !llvm.ptr) {
  llvm.store %val, %out : f64, !llvm.ptr
  return
}

func.func @caller(%unused: i32, %val: f64, %out: !llvm.ptr) {
  %square = arith.mulf %val, %val : f64
  call @callee(%square, %out) : (f64, !llvm.ptr) -> ()
  return
}
```

The below command will run the modular activity analysis and annotate the IR with attributes indicating if they are inactive values (`enzyme.icv`) and inactive instructions (`enzyme.ici`)
```sh
enzymemlir-opt <file.mlir> --print-activity-analysis='funcs=caller relative annotate'
```

Which should yield the following output:
```mlir
#distinct = distinct[0]<#enzyme.pseudoclass<@callee(1, 0)>>
#distinct1 = distinct[1]<#enzyme.pseudoclass<@caller(2, 0)>>
module {
  func.func @callee(%arg0: f64, %arg1: !llvm.ptr) attributes {enzyme.alias = [], enzyme.denseactive = [[#distinct, [#enzyme.argorigin<@callee(0)>, #enzyme.argorigin<@callee(1)>]]], enzyme.p2p = [], enzyme.sparseactive = [], enzyme.visited} {
    llvm.store %arg0, %arg1 {enzyme.ici = false, enzyme.icv = true} : f64, !llvm.ptr
    return {enzyme.ici = true, enzyme.icv = true}
  }
  func.func @caller(%arg0: i32, %arg1: f64, %arg2: !llvm.ptr) attributes {enzyme.alias = [], enzyme.denseactive = [[#distinct1, [#enzyme.argorigin<@caller(1)>, #enzyme.argorigin<@caller(2)>]]], enzyme.p2p = [], enzyme.sparseactive = [], enzyme.visited} {
    %0 = arith.mulf %arg1, %arg1 {enzyme.ici = false, enzyme.icv = false} : f64
    call @callee(%0, %arg2) {enzyme.ici = false, enzyme.icv = true} : (f64, !llvm.ptr) -> ()
    return {enzyme.ici = true, enzyme.icv = true}
  }
}
```

In this example, `%0` is a value with a differential dependency on `%arg1` that is then stored into `%arg2`, meaning it is an active value (`enzyme.icv = false`) and the `arith.mulf` op that produces it is an active instruction (`enzyme.ici = false`).
By default, float and pointer arguments are assumed to be active by the pass. This behavior can be changed by passing the `infer` flag to `print-activity-analysis` on a module set up with an `__enzyme_autodiff` call per [Enzyme's calling conventions](https://enzyme.mit.edu/getting_started/CallingConvention/#types).

Additional examples can be found in `~/Enzyme/enzyme/test/MLIR/ActivityAnalysis`. These are set up as unit tests and can be run by invoking the test runner:
```sh
cmake --build Enzyme/build --target check-enzymemlir
```

### Incorporating the analyses with Automatic Differentiation

At time of writing, the easiest way to incorporate the activity analyses together with Enzyme's differentiation is via the C/C++/CUDA pipeline in this artifact.
This artifact includes `ninjawrap`, a small Python library that emits a Ninja file with the appropriate commands to run the activity analyses and emit annotated LLVM IR such that Enzyme will use the activity results during differentiation.
The relevant code can be found in `build_benchmarks.py` and the various `{cpu,gpu}/<benchmark>/build.py` files that configures each benchmark.

### Extending the analyses to new dialects and operations

The analyses are implemented on both built-in and differentiation-specific [interfaces](https://mlir.llvm.org/docs/Interfaces/), and thus work as-is on arbitrary operations that correctly implement these interfaces.

The standard MLIR interfaces are as follows:
* Operations that implement functions and function calls should implement the respective `FunctionOpInterface` and `CallOpInterface`
* Operations that implement control flow should implement `BranchOpInterface` and `RegionBranchOpInterface` as appropriate
* All operations should implement `MemoryEffectOpInterface`. This interface describes operations that allocate, read, and write to memory, all of which are used by the analyses.
  * Crucially, operations that do not touch memory should be marked `Pure` or `NoMemoryEffect`.

The differentiation-specific interface that must be implemented is the `ActivityOpInterface`, which specifies when operations are either completely inactive (meaning there is no differential data dependency from any input to any output), or specific arguments are inactive (meaning there is no differential data dependency between that argument and any result).
All other results are conservatively assumed to be differentially dependent on all operands.

[The following example](https://github.com/EnzymeAD/Enzyme/blob/437f09ac322a424b79700d18614b910401d020d8/enzyme/Enzyme/MLIR/Implementations/LLVMAutoDiffOpInterfaceImpl.cpp#L39) shows how to express that the condition of a `select` is always inactive:
```cpp
struct SelectActivityInterface
    : public ActivityOpInterface::ExternalModel<SelectActivityInterface,
                                                LLVM::SelectOp> {
  bool isInactive(Operation *op) const { return false; }
  bool isArgInactive(Operation *op, size_t idx) const {
    // llvm.select is not inactive in general, but the condition is always
    // inactive.
    auto selectOp = cast<LLVM::SelectOp>(op);
    return selectOp.getCondition() == selectOp.getOperand(idx);
  }
};

void mlir::enzyme::registerLLVMDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    // ...
    LLVM::SelectOp::attachInterface<SelectActivityInterface>(*context);
    // ...
  });
}
```

[Another example](https://github.com/EnzymeAD/Enzyme-JAX/blob/90eb0bbd1949388c795c37277ff40d8a988d5a96/src/enzyme_ad/jax/Implementations/StableHLOAutoDiffOpInterfaceImpl.cpp#L2231) shows the registration of the `ActivityOpInterface` for an operation in the StableHLO dialect.
If defining operations with TableGen, [Enzyme includes examples of how to indicate that an operation is always fully inactive](https://github.com/EnzymeAD/Enzyme/blob/437f09ac322a424b79700d18614b910401d020d8/enzyme/Enzyme/MLIR/Implementations/LLVMDerivatives.td#L4).
