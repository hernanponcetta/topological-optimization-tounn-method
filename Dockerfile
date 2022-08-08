FROM docker.io/library/ubuntu@sha256:b5a61709a9a44284d88fb12e5c48db0409cfad5b69d4ff8224077c57302df9cf

RUN apt-get update && apt-get upgrade -y

# FEniCS
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:fenics-packages/fenics
RUN apt-get update
RUN apt-get install fenics -y

# Git
RUN apt install git -y

# wget
RUN apt install wget -y

# pip
RUN apt install python3-pip -y
RUN python3 -m pip install --upgrade pip

# scikit-learn
RUN pip3 install -U scikit-learn

# Numpy
RUN pip install numpy==1.22.1

# PyTorch
RUN pip3 install torch==1.10.1

# Meshio
RUN pip3 install meshio==4.4.6

# Pytest
RUN pip install -U pytest

# h5py
RUN pip install h5py==2.10.0

# Code formater
RUN pip3 install autopep8
RUN pip install yapf

# Zsh
RUN apt install zsh -y
RUN sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

WORKDIR /home/ps-fenics-torch

COPY . /home/ps-fenics-torch

CMD [ "zsh" ]