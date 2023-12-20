# Master-thesis


## Installation 

### Install Miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

conda config --add channels conda-forge
conda config --set channel_priority strict
```

### Select kernel and python version 
You may find the jupyter extension in vscode 

### Install the requirements
```bash
pip install -r requirements.txt
spacy download  en_core_web_sm
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

### Create python kernel 
```bash
python -m ipykernel install --user --name build_central --display-name "gpu-test"
```

### Install cuda-toolkit
```bash 
conda install -c nvidia cuda-toolkit
```

apt-get install aptitude
aptitude update && sudo aptitude safe-upgrade

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get -y install cuda

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.1-545.23.08-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.1-545.23.08-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-3
```



cd llama.cpp
export CUDA_HOME=/usr/local/lib/python3.10/dist-packages/triton/third_party/cuda
export PATH=${CUDA_HOME}/bin:$PATH
export LLAMA_CUBLAS=on
make clean
make libllama.so

export LLAMA_CPP_LIB=~/miniconda3/lib/python3.11/site-packages/llama_cpp/libllama.so
export LLAMA_CPP_LIB=/usr/local/lib/python3.10/dist-packages/llama_cpp/libllama.so
CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.48

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda