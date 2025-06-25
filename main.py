from cpu.ba.build import configure_ba
from cpu.bude.build import configure_bude
from cpu.lstm.build import configure_lstm
from cpu.hand.build import configure_hand
from cpu.gmm.build import configure_gmm

from gpu.lbm.build import configure_lbm
from gpu.lulesh.build import configure_lulesh
from gpu.rsbench.build import configure_rsbench
from gpu.xsbench.build import configure_xsbench

from ninjawrap import NWrapWriter


def collect_cpu_runtimes():
    """"""
    # Bude: ./build/all_active/bude -n 4096 --deck cpu/bude/data/bm1
    pass


def collect_gpu_runtimes():
    # LBM: -i datasets/lbm/short/input/120_120_150_ldc.of -o ref.dat -- 150 | grep "Kernel    "
    # LULESH: -s 60
    # RSBench: -m event -l 10200 | grep Runtime
    # XSBench: ./build/all_active/xsbench -m event -k 0 -l 17000000 | grep Runtime
    pass


def build_all():
    with open("build/build.ninja", "w") as f:
        writer = NWrapWriter(f)

        for stem, configure in [
            ("ba", configure_ba),
            ("bude", configure_bude),
            ("lstm", configure_lstm),
            ("hand", configure_hand),
            ("gmm", configure_gmm),
            ("lbm", configure_lbm),
            ("lulesh", configure_lulesh),
            ("rsbench", configure_rsbench),
            ("xsbench", configure_xsbench),
        ]:
            configure(
                writer,
                stem,
                prefix=f"relative/",
                all_active=False,
                dataflow=True,
                whole_program=False,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"informal/",
                all_active=False,
                dataflow=False,
                whole_program=False,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"gdce/",
                all_active=False,
                dataflow=False,
                whole_program=False,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"whole_program/",
                all_active=False,
                dataflow=True,
                whole_program=True,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"all_active/",
                all_active=True,
                dataflow=False,
                whole_program=False,
                custom_dce=False,
            )


if __name__ == "__main__":
    build_all()
