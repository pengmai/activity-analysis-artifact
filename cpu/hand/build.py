import argparse
import pathlib
from ninjawrap import abs_glob, NWrapWriter


def configure_hand(
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
    ]
    preopts = [
        "-O2",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-unroll-loops",
        "-ffast-math",
    ]

    if all_active:
        cflags += ["-DALL_ACTIVE=1"]
    dce_func = "diffehand_objective" if custom_dce else None
    dce_indices = "6,8,10,12,18" if custom_dce else None

    support_objs = writer.compile_c(
        abs_glob(SOURCE_DIR / "driver.c") + abs_glob(SOURCE_DIR / "hand_io.c"),
        prefix=prefix,
        cflags=cflags + preopts,
    )
    kernel_objs = writer.compile_enzyme_cpu(
        abs_glob(SOURCE_DIR / "hand.c"),
        cflags + preopts,
        prefix=prefix,
        public_symbols=["hand_objective", "dhand_objective"],
        dataflow=dataflow,
        relative=not whole_program,
        dce_func=dce_func,
        dce_indices=dce_indices,
    )
    writer.clang_link(prefix + outbin, support_objs + kernel_objs, ldflags=["-lm"])
