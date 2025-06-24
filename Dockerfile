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

RUN echo "Building LLVM and Enzyme" && ./build.sh

RUN python3 -m venv eval-env \
  && source eval-env/bin/activate \
  && pip install -r requirements.pip \
  && pip install -e ./ninjawrap

USER $USERNAME
