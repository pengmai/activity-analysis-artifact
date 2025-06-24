from cpu.ba.build import configure_ba
from cpu.bude.build import configure_bude
from cpu.lstm.build import configure_lstm
from cpu.hand.build import configure_hand
from cpu.gmm.build import configure_gmm

from ninjawrap import NWrapWriter


def collect_cpu_runtimes():
    """"""
    # Bude: ./build/all_active/bude -n 4096 --deck cpu/bude/data/bm1
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
        ]:
            configure(
                writer,
                stem,
                prefix=f"{stem}/relative/",
                all_active=False,
                dataflow=True,
                whole_program=False,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"{stem}/informal/",
                all_active=False,
                dataflow=False,
                whole_program=False,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"{stem}/gdce/",
                all_active=False,
                dataflow=False,
                whole_program=False,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"{stem}/whole_program/",
                all_active=False,
                dataflow=True,
                whole_program=True,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"{stem}/all_active/",
                all_active=True,
                dataflow=False,
                whole_program=False,
                custom_dce=False,
            )


if __name__ == "__main__":
    build_all()
