import pathlib
from ninjawrap.gen_build import abs_glob, NWrapWriter


def configure_rsbench(
    writer: NWrapWriter,
    outbin: str,
    prefix="",
    all_active=False,
    dataflow=False,
    whole_program=False,
    custom_dce=False,
    verify=False,
):
    SOURCE_DIR = pathlib.Path(__file__).parent
    support = ["main.cu", "io.cu", "init.cu", "material.cu", "utils.cu"]
    support = [abspath for pat in support for abspath in abs_glob(SOURCE_DIR / pat)]
    inputs = abs_glob(SOURCE_DIR / "simulation.cu")

    flags = [
        "-DNDEBUG",
        "-fno-exceptions",
        "--cuda-path=/usr/local/cuda-12.6",
        "--cuda-gpu-arch=sm_60",
        f"-DALL_ACTIVE={int(all_active)}",
        f"-DALWAYS_INLINE=1",
        "-O3",
    ]
    if verify:
        flags += ["-DVERIFY"]

    dflags = [
        "-force-nvvm",
        "-enzyme-loop-invariant-cache=1",
        "-enzyme-phi-restructure=1",
    ]

    public_symbols = [
        "_Z18calculate_macro_xsPdid5InputPiS1_iS_S1_S_P6WindowP4Poleii",
        "_Z25xs_lookup_kernel_baseline5Input14SimulationData",
    ]
    dce_func = "diffe" + public_symbols[0] if custom_dce else None
    dce_indices = "11" if custom_dce else None

    kernel_objs = writer.compile_cuda_llvm(
        inputs,
        cflags=flags,
        prefix=prefix,
        dflags=["-force-nvvm"],
        # dflags=dflags,
        public_symbols=public_symbols,
        use_mlir=True,
        dataflow=dataflow,
        emit_clang_enzyme=True,
        clang_ad=False,
        relative=not whole_program,
        dce_func=dce_func,
        dce_indices=dce_indices,
    )
    support_objs = writer.compile_cuda(support, cflags=flags, prefix=prefix)
    writer.link_executable(prefix + outbin, kernel_objs + support_objs)
