# 教學手冊

可選擇自行安裝環境或使用 docker 啟動訓練環境。

## 系統需求

- OS: Windows 10 or Ubuntu 18.04

### 使用 GPU

- 顯示卡 compute capability >= 3.0
- 已安裝 NVIDIA Driver 410+

## 安裝環境

以下指令在 Ubuntu 18.04 測試通過。

### 安裝 CUDA 9.0

```bash
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sudo sh cuda_9.0.176_384.81_linux-run
```

### 設定 CUDA 環境變數

將環境變數加入 ~/.bashrc

```bash
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
```

### 安裝 Miniconda

請參考: https://docs.conda.io/en/latest/miniconda.html

### 建立 conda 虛擬環境

```bash
conda create -n tf-1.9 python=3.6
```

### 啟動 conda 虛擬環境

```bash
source activate tf-1.9
```

### (啟動環境後) 利用 conda 安裝 tensorflow

選擇安裝 CPU 版或 GPU 版

#### CPU 版

```bash
conda install tensorflow==1.9 matplotlib
```

#### GPU 版

```bash
conda install tensorflow-gpu==1.9 matplotlib
```

## 使用 docker image 啟動訓練環境 (Linux Only)

以下指令在 Ubuntu 18.04 測試通過。

### 安裝 docker

請參考: https://docs.docker.com/install/linux/docker-ce/ubuntu/

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

將使用者加入 docker 群組，避免 docker 指令需要 sudo 權限

```bash
sudo usermod -aG docker your-user
```

### 安裝 nvidia-docker

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
```

#### 測試 nvidia-docker 是否安裝成功

```bash
docker run --gpus all --rm nvidia/cuda:9.0-base nvidia-smi
```

### 建置 docker image

```bash
docker build --rm -t ccumvllab/5g-1 ./
```

### 啟動 docker container

```bash
docker run --gpus all -it -v /path/to/6DoF/Dockershare:/6DoF/Dockershare ccumvllab/5g-1
```

**Note**: 參數 `--gpus` 可以指定 docker container 要使用哪些 GPU ID。

請參考：https://github.com/NVIDIA/nvidia-docker#usage

## 訓練模型

```bash
cd /path/to/6DoF/Dockershare/tools/
python Bi-RNN.py
```
