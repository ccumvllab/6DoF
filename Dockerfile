FROM nvidia/cuda:9.0-base-ubuntu16.04

LABEL maintainer="ccumvllab" version="1.1"

SHELL ["/bin/bash", "-c"] 

# Install Miniconda3
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN source ~/.bashrc && \
    conda create -n torch-1.3 matplotlib python=3.6 pytorch==1.3.1 torchvision cudatoolkit=9.0 -c pytorch -y

RUN echo "conda activate torch-1.3" >> ~/.bashrc

# 創建工作目錄
RUN mkdir /6Dof
WORKDIR /6Dof

# 複製目前目錄下的內容，放進 Docker 容器中
ADD . /6Dof

VOLUME ["/6Dof/Dockershare"]
