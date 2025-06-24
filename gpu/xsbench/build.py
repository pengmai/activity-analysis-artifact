import pathlib
from ninjawrap.gen_build import abs_glob, NWrapWriter


def configure_xsbench(
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
    support = ["Main.cu", "io.cu", "GridInit.cu", "XSutils.cu", "Materials.cu"]
    support = [abspath for pat in support for abspath in abs_glob(SOURCE_DIR / pat)]
    inputs = abs_glob(SOURCE_DIR / "Simulation.cu")

    flags = [
        # "-I",
        # str(pathlib.Path(__file__).parent.absolute()),
        "-DNDEBUG",
        "-fno-exceptions",
        "--cuda-path=/usr/local/cuda-12.6",
        "--cuda-gpu-arch=sm_60",
        # "--no-cuda-version-check",
        f"-DALL_ACTIVE={int(all_active)}",
        "-O3",
    ]

    if verify:
        flags += ["-DPRINT=1"]

    dflags = [
        "-enzyme-new-cache=1",
        "-enzyme-mincut-cache=1",
        "-enzyme-loop-invariant-cache=1",
        "-enzyme-phi-restructure=1",
        "-enzyme-coalese",
        "-enzyme-force-malloc",
    ]

    public_symbols = [
        "_Z18calculate_macro_xsdillPiPdS0_S_P16NuclideGridPointS_S0_iii",
        "_Z25xs_lookup_kernel_baseline6Inputs14SimulationData",
    ]
    dce_func = "diffe" + public_symbols[0] if custom_dce else None
    dce_indices = "6,8" if custom_dce else None

    kernel_objs = writer.compile_cuda_llvm(
        inputs,
        cflags=flags,
        dflags=dflags,
        prefix=prefix,
        public_symbols=public_symbols,
        use_mlir=dataflow,
        dataflow=dataflow,
        relative=not whole_program,
        dce_func=dce_func,
        dce_indices=dce_indices,
    )
    support_objs = writer.compile_cuda(support, cflags=flags, prefix=prefix)
    writer.link_executable(prefix + outbin, kernel_objs + support_objs)
