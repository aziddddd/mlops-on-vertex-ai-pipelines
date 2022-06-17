FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Upgrade Python to 3.7.12
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libbz2-dev \
    libfreetype6-dev \
    libpng-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    liblzma-dev \
    libffi-dev \
    tk-dev
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
RUN /root/.pyenv/bin/pyenv install 3.7.12
ENV PATH /root/.pyenv/versions/3.7.12/bin:$PATH

RUN pip3 install --upgrade pip
COPY test_requirements.txt ./
RUN pip3 install -r test_requirements.txt

COPY ./ ./

COPY config.py ml_components/config.py
