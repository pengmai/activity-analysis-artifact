import pathlib
from ninjawrap.gen_build import abs_glob, NWrapWriter


def configure_lulesh(
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
    support = ["allocator.cu", "lulesh-comms.cu", "lulesh-comms-gpu.cu"]
    support = [abspath for pat in support for abspath in abs_glob(SOURCE_DIR / pat)]
    inputs = abs_glob(SOURCE_DIR / "lulesh.cu")

    flags = [
        "-I",
        str(SOURCE_DIR.absolute()),
        "-DNDEBUG",
        "-fno-exceptions",
        "--cuda-path=/usr/local/cuda-12.6",
        "--cuda-gpu-arch=sm_60",
        # "--no-cuda-version-check",
        "-DRESTRICT=1",
        f"-DALL_ACTIVE={int(all_active)}",
        "-DNormal_forward=0",
        "-O3",
    ]
    public_symbols = [
        "_Z51Inner_ApplyMaterialPropertiesAndUpdateVolume_kernelidddPdS_S_S_dddddPiS_S_S_S_dS_dS0_iPKiS2_i",
        "_Z45ApplyMaterialPropertiesAndUpdateVolume_kernelidddPdS_S_S_S_S_S_dddddPiS_S_S_S_S_S_S_dS_dS0_iPKiS2_i",
    ]
    dce_func = "diffe" + public_symbols[0] if custom_dce else None
    dce_indices = "5,7,10,21,23" if custom_dce else None
    if verify:
        flags += ["-DVERIFY"]

    kernel_objs = writer.compile_cuda_llvm(
        inputs,
        cflags=flags,
        prefix=prefix,
        public_symbols=public_symbols,
        use_mlir=True,
        dataflow=dataflow,
        relative=not whole_program,
        force_intraproc=whole_program,
        dce_func=dce_func,
        dce_indices=dce_indices,
    )
    support_objs = writer.compile_cuda(support, cflags=flags, prefix=prefix)
    writer.link_executable(prefix + outbin, kernel_objs + support_objs)
