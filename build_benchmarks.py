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
                prefix=f"relative/{stem}/",
                all_active=False,
                dataflow=True,
                whole_program=False,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"informal/{stem}/",
                all_active=False,
                dataflow=False,
                whole_program=False,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"gdce/{stem}/",
                all_active=True,
                dataflow=False,
                whole_program=False,
                custom_dce=True,
            )
            configure(
                writer,
                stem,
                prefix=f"whole_program/{stem}/",
                all_active=False,
                dataflow=True,
                whole_program=True,
                custom_dce=False,
            )
            configure(
                writer,
                stem,
                prefix=f"all_active/{stem}/",
                all_active=True,
                dataflow=False,
                whole_program=False,
                custom_dce=False,
            )


if __name__ == "__main__":
    build_all()
