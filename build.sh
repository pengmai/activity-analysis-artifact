# sudo chown `whoami` -R .

# Build LLVM
mkdir llvm-project/build && cd llvm-project/build
cmake ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_PROJECTS='clang;mlir' -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" -DLLVM_ENABLE_LLD=ON
ninja
cd ../..

# Build Enzyme
mkdir Enzyme/build && cd Enzyme/build
cmake ../enzyme -G Ninja -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm -DENZYME_MLIR=ON
ninja
cd ../..
