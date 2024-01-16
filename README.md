# Master-thesis


## Installation 
Add the path of the right python version
```bash
export PATH=/myhome/miniconda3/bin:$PATH
```

If it does not work, add `export PATH=/myhome/miniconda3/bin:$PATH` in `~/.bashrc` with `vim` and run `source ~/.bashrc`
Install `vim` with `apt-get install vim`
And restart your terminal and verify with `python -V` the version of python that you have. 

If you ahve problem with `apt-get` run 

```bash
apt-get clean
rm -rf /var/lib/apt/lists/*
apt-get clean
apt-get update 
apt-get upgrade
```

### Install the requirements
```bash
pip install -r requirements.txt
spacy download  en_core_web_sm
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

### Setup git 
```bash 
cp /myhome/.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
cp /myhome/.ssh/id_rsa /root/.ssh/id_rsa
git config --global user.email "mathieu.desponds@ketl.ch"
git config --global user.name "Mathieu Desponds"
```

### Add jupyter extension on vscode
Add the jupyter notebook extension on vscode

### Run the set up script

We need to reload the dataset that were on the cache run `python setup.py`

### Direct run of everything
```bash
export PATH=/myhome/miniconda3/bin:$PATH
pip install -r requirements.txt
spacy download  en_core_web_sm
git config --global user.email "mathieu.desponds@ketl.ch"
git config --global user.name "Mathieu Desponds"
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
cp /myhome/.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
cp /myhome/.ssh/id_rsa /root/.ssh/id_rsa
python setup.py
```
Then, restart the VM so that the changes are added 

### To use the application from the servers 
```bash
cp /myhome/default.conf /etc/nginx/conf.d/default.conf
nginx -s reload
apt-get update -y
apt-get install -y sqlite3 
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
cd /myhome/Master-thesis/
python app.py
```



#### Explenations

You want to setup the nginx server and override `/etc/nginx/conf.d/default.conf` then make `nginx -s reload`
Install sqlite3 `apt-get install sqlite3`
You can copy form `myhome/` with `cp /myhome/default.conf /etc/nginx/conf.d/default.conf`

```bash
apt-get clean
rm -rf /var/lib/apt/lists/*
apt-get clean
apt-get update -y
apt-get upgrade -y
apt-get install sqlite3 -y
cp /myhome/default.conf /etc/nginx/conf.d/default.conf
nginx -s reload
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```


```bash 
server {
    listen 80;
    server_name compute.datascience.ch;
    location /custom-nginx/my-app/ {
        proxy_pass  http://127.0.0.1:45505;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Prefix /;
        proxy_read_timeout 120;
        proxy_connect_timeout 120;
        proxy_send_timeout 120; 
    }
}
```

apt update
apt upgrade
add-apt-repository ppa:ubuntu-toolchain-r/test
apt update
apt install gcc-11 g++-11

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