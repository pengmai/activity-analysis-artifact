import pathlib
from ninjawrap import abs_glob, NWrapWriter


def configure_lstm(
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
    dce_func = "diffelstm_objective" if custom_dce else None
    dce_indices = "10" if custom_dce else None

    support_objs = writer.compile_c(
        abs_glob(SOURCE_DIR / "driver.c") + abs_glob(SOURCE_DIR / "lstm_io.c"),
        prefix=prefix,
        cflags=cflags,
    )
    kernel_objs = writer.compile_enzyme_cpu(
        abs_glob(SOURCE_DIR / "lstm.c"),
        cflags,
        public_symbols=[
            "lstm_objective",
            "dlstm_objective",
        ],
        prefix=prefix,
        dataflow=dataflow,
        relative=not whole_program,
        dce_func=dce_func,
        dce_indices=dce_indices,
    )
    writer.clang_link(prefix + outbin, support_objs + kernel_objs, ldflags=["-lm"])
