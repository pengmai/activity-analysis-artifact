import pathlib
from ninjawrap import abs_glob, NWrapWriter


def configure_bude(
    writer: NWrapWriter,
    outbin: str,
    prefix="",
    all_active=False,
    dataflow=False,
    whole_program=False,
    custom_dce=False,
):
    SOURCE_DIR = pathlib.Path(__file__).parent / "src"
    cflags = [
        f"-I{SOURCE_DIR}",
        f"-I{pathlib.Path(__file__).parents[1] / 'shared_include'}",
        "-O2",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-unroll-loops",
        "-ffast-math",
        "-DWGSIZE=256",
        "-march=native",
    ]

    if all_active:
        cflags += ["-DALL_ACTIVE=1"]
    dce_func = "diffeonecompute" if custom_dce else None
    dce_indices = "3,5,7,9,11,13,15,19" if custom_dce else None

    support_objs = writer.compile_c(
        abs_glob(SOURCE_DIR / "driver.c"), cflags=cflags, prefix=prefix
    )
    # Appears to be some GVN related issue that prevents this from compiling with clang-20
    kernel_objs = writer.compile_enzyme_cpu(
        abs_glob(SOURCE_DIR / "bude.c"),
        cflags,
        prefix=prefix,
        public_symbols=["onecompute", "done_compute"],
        cc="clang-18",
        dataflow=dataflow,
        relative=not whole_program,
        dce_func=dce_func,
        dce_indices=dce_indices,
    )
    writer.clang_link(prefix + outbin, support_objs + kernel_objs, ldflags=["-lm"])
