FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

COPY requirements.txt /workspace/
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
    && apt-get update \ 
    && apt install redis-server vim git sudo -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r /workspace/requirements.txt --use-feature=2020-resolver \
    && rm /workspace/requirements.txt 
RUN git config --global --add safe.directory /workspace

RUN groupadd -r -g 1000 dream \
    && useradd -r -g dream -u 1000 -m dream
USER dream