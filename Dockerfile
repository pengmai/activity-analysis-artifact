FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG USERNAME=evaluator

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/$USERNAME
ENV WORKDIR=$HOME

WORKDIR $HOME

RUN groupadd -r $USERNAME && \
  useradd -r -g $USERNAME -d /home/$USERNAME -m $USERNAME

RUN apt-get -q update \
  && apt-get install -y --no-install-recommends ca-certificates software-properties-common curl gnupg2 \
    cmake gcc g++ clang-18 lld python3-venv python3-pip ninja-build git time sudo

RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

COPY . $HOME

# Build Clang and MLIR
RUN mkdir llvm-project/build && cd llvm-project/build \
  && cmake ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DLLVM_ENABLE_PROJECTS='clang;mlir' -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" -DLLVM_ENABLE_LLD=ON \
  && ninja \
  && cd ../..

# Build Enzyme
RUN mkdir Enzyme/build && cd Enzyme/build \
  && cmake ../enzyme -G Ninja -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm -DENZYME_MLIR=ON \
  && ninja \
  && cd ../..

RUN python3 -m venv eval-env

ENV PATH="$HOME/eval-env/bin:$PATH"

RUN pip install -r requirements.txt \
  && pip install -e ./ninjawrap

# Build benchmarks
RUN mkdir build && python main.py && ninja -C build
# LULESH and LBM write their outputs to files. The enclosing directories must be owned by the user.
RUN chown $USERNAME -R cpu gpu build && chown $USERNAME .

USER $USERNAME
