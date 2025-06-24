ARG LLVM_VERSION=20

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG USERNAME=evaluator
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/$USERNAME

WORKDIR $HOME

RUN groupadd -r $USERNAME && \
  useradd -r -g $USERNAME -d /home/$USERNAME -m $USERNAME

RUN apt-get -q update \
  && apt-get install -y --no-install-recommends ca-certificates software-properties-common curl gnupg2 \
    cmake gcc g++ ninja-build git time sudo
  # && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  # && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
