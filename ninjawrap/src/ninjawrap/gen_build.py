import pathlib
import subprocess
import os
from io import TextIOWrapper

from . import ninja_syntax

LLVM_VER = 20
HOME = pathlib.Path(os.environ["WORKDIR"])
LLVM_BUILD_DIR = HOME / "llvm-project" / "build"

ENZYME_BUILD_DIR = HOME / "Enzyme" / "build"
print("Using Enzyme build dir", ENZYME_BUILD_DIR)
ENZYME_DYLIB = str(ENZYME_BUILD_DIR / "Enzyme" / f"LLVMEnzyme-{LLVM_VER}.so")
CLANG_ENZYME = str(ENZYME_BUILD_DIR / "Enzyme" / f"ClangEnzyme-{LLVM_VER}.so")
ENZYME_MLIR_OPT = str(ENZYME_BUILD_DIR / "Enzyme" / "MLIR" / "enzymemlir-opt")
ENZYME_TRANSLATE = str(
    ENZYME_BUILD_DIR
    / "Enzyme"
    / "MLIR"
    / "enzymemlir-translate"
    / "enzymemlir-translate"
)

CLANG = str(LLVM_BUILD_DIR / "bin" / f"clang")
CLANGPP = str(LLVM_BUILD_DIR / "bin" / f"clang++")
LLC = str(LLVM_BUILD_DIR / "bin" / "llc")
MLIR_OPT = str(LLVM_BUILD_DIR / "bin" / "mlir-opt")
MLIR_TRANSLATE = str(LLVM_BUILD_DIR / "bin" / "mlir-translate")
OPT = str(LLVM_BUILD_DIR / "bin" / "opt")


def abs_glob(pattern):
    if isinstance(pattern, pathlib.Path):
        return [pattern.absolute()]
    return [p.absolute() for p in pathlib.Path().glob(pattern)]


class NWrapWriter:
    def __init__(self, output: TextIOWrapper, build_dir="build"):
        self.writer = ninja_syntax.Writer(output)
        self.build_dir = build_dir
        self.emit_rules()
        self._cache = set()

    def build(self, key, *args, **kwargs):
        if key not in self._cache:
            self.writer.build(key, *args, **kwargs)
            self._cache.add(key)

    def emit_rules(self):
        writer = self.writer
        writer.variable("cc", CLANGPP)
        writer.variable("llc", LLC)
        writer.variable("emliropt", ENZYME_MLIR_OPT)
        writer.variable("mliropt", MLIR_OPT)
        writer.variable("etranslate", ENZYME_TRANSLATE)
        writer.variable("enzyme_lib", ENZYME_DYLIB)
        writer.variable("clang_enzyme", CLANG_ENZYME)
        writer.variable("opt", OPT)
        writer.variable("sm_ver", "sm_60")
        writer.variable("compute_ver", "compute_60")
        LDFLAGS = "-L/usr/local/cuda-12.6/lib64 -lcudart_static -ldl -lrt -lpthread -lm"
        writer.variable("ldflags", LDFLAGS)

        writer.rule("cc", "$cc $cflags -o $out $in")
        writer.rule("opt", "$opt $optflags $in -o $out")
        writer.rule("emliropt", "$emliropt $optflags $in -o $out")
        writer.rule("mliropt", "$mliropt $optflags $in -o $out")
        writer.rule(
            "cchost",
            "$cc $cflags -c -o $out $in_cu --cuda-host-only -relocatable-pch -Xclang -fcuda-include-gpubinary -Xclang $in_fatbin",
        )
        writer.rule(
            "emitllvmdev", "$cc $cflags $in -o $out -S -emit-llvm --cuda-device-only"
        )
        writer.rule(
            "emitllvmdevclang",
            # "$cc $cflags $in -Xclang -load -Xclang $clang_enzyme -o $out -S -emit-llvm --cuda-device-only",
            "$cc $cflags $in -Xclang -load -Xclang $clang_enzyme -mllvm --enzyme-dump-module -o /dev/null -S -emit-llvm --cuda-device-only > $out",
        )
        writer.rule("raisemlir", "$etranslate $in -o $out -import-llvm")
        writer.rule("lowermlir", "$etranslate $in -o $out -activity-to-llvm")
        writer.rule(
            "enzymead",
            "$opt -S -load-pass-plugin=$enzyme_lib -passes='enzyme,preserve-nvvm' $dflags $in -o $out",
        )
        writer.rule(
            "emitptx", "$llc -march=nvptx64 -mcpu=$sm_ver -mattr=+ptx64 $in -o $out"
        )
        writer.rule("emitptxo", "ptxas -m64 --gpu-name=$sm_ver $in -o $out")
        writer.rule(
            "fatbin",
            "fatbinary --64 --create $out --image=profile=$sm_ver,file=$in_ptxo --image=profile=$compute_ver,file=$in_ptx -link",
        )
        writer.rule(
            "dlink",
            "nvcc $in -gencode arch=$compute_ver,code=$sm_ver -dlink -o $out -lcudart -lcudart_static -lcudadevrt",
        )
        writer.rule(
            "ld",
            "nvcc $in -o $out $ldflags -arch=$sm_ver -lcudart -lcudart_static -lcudadevrt",
        )

    def compile_c(self, inputs: list[pathlib.Path], prefix="", cflags: list[str] = []):
        results = []
        for input_file in inputs:
            stem = prefix + input_file.stem
            input_file = str(input_file)

            result = f"{stem}.o"
            self.build(
                f"{stem}.o",
                "cc",
                input_file,
                variables={"cflags": " ".join(["-c"] + cflags), "cc": CLANG},
            )
            results.append(result)
        return results

    def compile_enzyme_cpu(
        self,
        inputs: list[pathlib.Path],
        cflags: list[str],
        dflags: list[str] = [],
        public_symbols: list[str] = [],
        prefix="",
        mlir=True,
        dataflow=True,
        relative=True,
        cc="",
        dce_func=None,
        dce_indices=None,
    ):
        writer = self.writer
        if dce_func and not dce_indices:
            raise ValueError("must specify both dce_func and dce_indices")
        c_compiler = cc or CLANG
        results = []
        for input_file in inputs:
            stem = prefix + input_file.stem
            input_file = str(input_file)

            self.build(
                f"{stem}.ll",
                "cc",
                input_file,
                variables={
                    "cflags": " ".join(cflags + ["-S", "-emit-llvm"]),
                    "cc": c_compiler,
                },
            )
            if dataflow:
                self.build(
                    f"{stem}.mlir",
                    "raisemlir",
                    f"{stem}.ll",
                    implicit=ENZYME_TRANSLATE,
                )
                should_privatize = len(public_symbols) != 0
                if should_privatize:
                    self.build(
                        f"{stem}.private.mlir",
                        "mliropt",
                        f"{stem}.mlir",
                        variables={
                            "optflags": f"-symbol-privatize=exclude={','.join(public_symbols)}"
                        },
                    )
                privatized = (
                    f"{stem}.private.mlir" if should_privatize else f"{stem}.mlir"
                )
                optflags = ["infer", "annotate"]
                if relative:
                    optflags += ["relative"]
                self.build(
                    f"{stem}.analyzed.mlir",
                    "emliropt",
                    privatized,
                    variables={
                        "optflags": f"-print-activity-analysis='{' '.join(optflags)}'"
                    },
                )
                self.build(
                    f"{stem}.lower.ll",
                    "lowermlir",
                    f"{stem}.analyzed.mlir",
                    implicit=ENZYME_TRANSLATE,
                )
            lowered = f"{stem}.lower.ll" if dataflow else f"{stem}.ll"
            differentiated = f"{stem}.diff.ll"
            self.build(
                differentiated,
                "enzymead",
                lowered,
                implicit=str(ENZYME_DYLIB),
                variables={"dflags": " ".join(dflags)},
            )
            if dce_func:
                self.build(
                    f"{stem}.dce.ll",
                    "opt",
                    differentiated,
                    variables=[
                        (
                            "optflags",
                            f"-load-pass-plugin=$enzyme_lib -passes='custom-dce' -custom-dce-func={dce_func} -custom-dce-indices={dce_indices}",
                        )
                    ],
                )
                differentiated = f"{stem}.dce.ll"
            final_flags = ["-c", "-O3"]
            self.build(
                f"{stem}.o",
                "cc",
                differentiated,
                variables={"cflags": " ".join(final_flags), "cc": c_compiler},
            )
            results.append(f"{stem}.o")
        return results

    def clang_link(self, executable: str, inputs: list[str], ldflags: list[str] = []):
        self.writer.build(
            executable,
            "cc",
            inputs,
            variables={"cflags": " ".join(ldflags), "cc": CLANG},
        )

    def compile_cuda(
        self, inputs: list[pathlib.Path], prefix="", cflags: list[str] = []
    ):
        results = []
        flags = ["-c", "--cuda-path=/usr/local/cuda-12.6", "--cuda-gpu-arch=sm_60"]
        diff_flags = ["-Rpass=enzyme", "-Xclang", "-load", "-Xclang", CLANG_ENZYME]
        for input_file in inputs:
            stem = prefix + input_file.stem
            input_file = str(input_file)

            result = f"{stem}.o"
            self.writer.build(
                f"{stem}.o",
                "cc",
                input_file,
                variables={"cflags": " ".join(cflags + flags + diff_flags)},
            )
            results.append(result)
        return results

    def link_executable(
        self, executable: str, inputs: list[str], ldflags: list[str] = []
    ):
        self.writer.build(executable, "ld", inputs)

    def compile_cuda_llvm(
        self,
        inputs: list[pathlib.Path],
        cflags: list[str],
        prefix="",
        dflags: list[str] = [],
        public_symbols: list[str] = [],
        use_mlir=False,
        dataflow=False,
        relative=True,
        emit_clang_enzyme=True,
        clang_ad=False,
        dce_func=None,
        dce_indices=None,
    ):
        writer = self.writer
        if dce_func and not dce_indices:
            raise ValueError("must specify both dce_func and dce_indices")
        if dataflow and not use_mlir:
            raise ValueError("cannot use dataflow without mlir")
        if clang_ad and emit_clang_enzyme:
            raise ValueError("clang_ad and emit_clang enzyme cannot both be set")

        results = []
        for input_file in inputs:
            stem = prefix + input_file.stem
            input_file = str(input_file)

            clang_flags = []
            if clang_ad:
                clang_flags = ["-Xclang -load -Xclang", CLANG_ENZYME, "-Rpass=enzyme"]

            # Host side
            writer.build(
                f"{stem}_host.o",
                "cchost",
                [input_file] + [f"{stem}.fatbin"],
                variables=[
                    (
                        "cflags",
                        " ".join(
                            cflags
                            + ["-Xclang -load -Xclang", CLANG_ENZYME, "-Rpass=enzyme"]
                        ),
                    ),
                    ("in_cu", input_file),
                    ("in_fatbin", f"{stem}.fatbin"),
                ],
            )

            # Device side
            writer.build(
                f"{stem}.dev.ll",
                "emitllvmdevclang" if emit_clang_enzyme else "emitllvmdev",
                input_file,
                variables=[("cflags", " ".join(cflags + clang_flags))],
            )
            if use_mlir:
                writer.build(
                    f"{stem}.mlir",
                    "raisemlir",
                    f"{stem}.dev.ll",
                    implicit=ENZYME_TRANSLATE,
                )
                should_privatize = len(public_symbols) != 0
                if should_privatize:
                    writer.build(
                        f"{stem}.private.mlir",
                        "mliropt",
                        f"{stem}.mlir",
                        variables={
                            "optflags": f"-symbol-privatize=exclude={','.join(public_symbols)}"
                        },
                    )
                privatized = (
                    f"{stem}.private.mlir" if should_privatize else f"{stem}.mlir"
                )
                if dataflow:
                    optflags = ["infer", "annotate"]
                    if relative:
                        optflags += ["relative"]
                    writer.build(
                        f"{stem}.analyzed.mlir",
                        "emliropt",
                        privatized,
                        variables={
                            "optflags": f"-print-activity-analysis='{' '.join(optflags)}'"
                        },
                    )
                to_lower = f"{stem}.analyzed.mlir" if dataflow else privatized
                writer.build(
                    f"{stem}.lower.ll",
                    "lowermlir",
                    to_lower,
                    implicit=ENZYME_TRANSLATE,
                )
            lowered = f"{stem}.lower.ll" if use_mlir else f"{stem}.dev.ll"
            differentiated = f"{stem}.diff.ll"
            writer.build(
                differentiated,
                "enzymead",
                lowered,
                implicit=str(ENZYME_DYLIB),
                variables={"dflags": " ".join(dflags)},
            )
            if dce_func:
                writer.build(
                    f"{stem}.dce.ll",
                    "opt",
                    differentiated,
                    variables=[
                        (
                            "optflags",
                            f"-load-pass-plugin=$enzyme_lib -passes='custom-dce' -custom-dce-func={dce_func} -custom-dce-indices={dce_indices}",
                        )
                    ],
                )
                differentiated = f"{stem}.dce.ll"
            writer.build(f"{stem}.ptx", "emitptx", differentiated)
            writer.build(f"{stem}.ptx.o", "emitptxo", f"{stem}.ptx")
            writer.build(
                f"{stem}.fatbin",
                "fatbin",
                [f"{stem}.ptx", f"{stem}.ptx.o"],
                variables=[
                    ("in_ptx", f"{stem}.ptx"),
                    ("in_ptxo", f"{stem}.ptx.o"),
                ],
            )
            writer.build(f"{stem}_dlink.o", "dlink", f"{stem}.fatbin")
            results.extend([f"{stem}_host.o", f"{stem}_dlink.o"])
        return results

    # def build(self):
    #     subprocess.run(["ninja", "-C", self.build_dir], check=True)
