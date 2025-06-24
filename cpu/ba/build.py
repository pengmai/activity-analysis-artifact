import pathlib
from ninjawrap import abs_glob, NWrapWriter


def configure_ba(
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
    ]

    if all_active:
        cflags += ["-DALL_ACTIVE=1"]

    dce_func = "diffeecompute_reproj_error" if custom_dce else None
    dce_indices = "7" if custom_dce else None

    support_objs = writer.compile_c(
        abs_glob(SOURCE_DIR / "driver.c"), cflags=cflags, prefix=prefix
    )
    kernel_objs = writer.compile_enzyme_cpu(
        abs_glob(SOURCE_DIR / "ba.c"),
        cflags,
        prefix=prefix,
        public_symbols=[
            "ecompute_reproj_error",
            "dcompute_reproj_error",
            "ecompute_zach_weight_error",
            "dcompute_w_error",
        ],
        dataflow=dataflow,
        relative=not whole_program,
        dce_func=dce_func,
        dce_indices=dce_indices,
    )
    writer.clang_link(prefix + outbin, support_objs + kernel_objs, ldflags=["-lm"])
