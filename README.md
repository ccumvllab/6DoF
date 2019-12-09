# 教學手冊

訓練模型

```bash
cd /path/to/6DoF/Dockershare/tools/
python Bi-RNN.py
```

訓練環境設定，可選擇：

1. 自行安裝環境
2. 使用 docker 啟動訓練環境

## 系統需求

- OS: Windows 10 or Ubuntu 18.04

### 使用 GPU

- 顯示卡 compute capability >= 3.0
- NVIDIA Driver 410+
- CUDA 9.0 Runtime Library

## 安裝環境

環境需求：

1. CUDA 9.0
2. Miniconda3
3. Tensorflow 1.9

以下指令在 Ubuntu 18.04 測試通過。

### CUDA 9.0

#### 安裝 CUDA 9.0

從 [官網](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal) 下載 CUDA 安裝檔並安裝。

```bash
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sudo sh cuda_9.0.176_384.81_linux-run
```

**NOTE**: 如果已經安裝 NVIDIA Driver，CUDA 安裝過程只需要安裝 CUDA Runtime Library。

#### 設定 CUDA 環境變數

將環境變數加入 ~/.bashrc

```bash
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
```

### Miniconda3

#### 安裝 Miniconda3

請參考：https://docs.conda.io/en/latest/miniconda.html

#### 建立 conda 虛擬環境

```bash
conda create -n tf-1.9 python=3.6
```

#### 啟動 conda 虛擬環境

```bash
source activate tf-1.9
```

### Tensorflow

#### (啟動環境後) 利用 conda 安裝 tensorflow

選擇安裝 CPU 版或 GPU 版

##### CPU 版

```bash
conda install tensorflow==1.9 matplotlib
```

##### GPU 版

```bash
conda install tensorflow-gpu==1.9 matplotlib
```

## 使用 Docker 啟動訓練環境 (Linux Only)

需求：

1. Docker
2. NVIDIA Container Toolkit

以下指令在 Ubuntu 18.04 測試通過。

### 安裝 Docker

請參考：https://docs.docker.com/install/linux/docker-ce/ubuntu/

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

將使用者加入 docker 群組，避免 docker 指令需要 sudo 權限

```bash
sudo usermod -aG docker your-user
```

### 安裝 NVIDIA Container Toolkit

請參考：https://github.com/NVIDIA/nvidia-docker

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
```

#### 測試 NVIDIA Container Toolkit 是否安裝成功

```bash
docker run --gpus all --rm nvidia/cuda:9.0-base nvidia-smi
```

### 取得 docker image

```bash
docker pull ccumvllab/5g-1
```

### 啟動 docker container

```bash
docker run --gpus all -it -v /absolute/path/to/6DoF/Dockershare:/6DoF/Dockershare ccumvllab/5g-1
```

參數說明：

- `--gpus`: 指定 docker container 要使用哪些 GPU ID。
  - 參考：https://github.com/NVIDIA/nvidia-docker#usage
- `-v`: 將外部絕對路徑掛載至容器內的資料夾。
