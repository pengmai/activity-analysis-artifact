import pathlib
from ninjawrap.gen_build import abs_glob, NWrapWriter


def configure_lbm(
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
    support = ["main.cc", "parboil_cuda.c", "args.c"]
    support = [abspath for pat in support for abspath in abs_glob(SOURCE_DIR / pat)]
    inputs = abs_glob(SOURCE_DIR / "lbm.cu")

    flags = [
        "-I",
        str(SOURCE_DIR.absolute()),
        "--cuda-path=/usr/local/cuda-12.6",
        "--cuda-gpu-arch=sm_60",
        "-mllvm",
        "-max-heap-to-stack-size=1000000",
        "-I/usr/local/cuda-12.6/include",
        "-DABI",
        "-DSIZE=20",
        # "--no-cuda-version-check",
        "-DALLOW_AD=1",
        "-DALLOCATOR",
        "-O3",
    ]

    if verify:
        flags += ["-DVERIFY"]

    dflags = [
        "-enzyme-new-cache=1",
        "-enzyme-mincut-cache=1",
        "-enzyme-loop-invariant-cache=1",
        "-enzyme-phi-restructure=1",
        "-enzyme-coalese",
        "-enzyme-force-malloc",
    ]

    kernel_objs = writer.compile_cuda_llvm(
        inputs,
        cflags=flags,
        dflags=dflags,
        prefix=prefix,
        use_mlir=True,
        dataflow=dataflow,
        relative=not whole_program,
    )
    support_objs = writer.compile_cuda(support, cflags=flags, prefix=prefix)
    writer.link_executable(prefix + outbin, kernel_objs + support_objs)
