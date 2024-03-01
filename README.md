# Master-thesis

This is the repository of my Master Thesis. The final report can be found in the folder under `Master-Thesis.pdf`.

## Abstract

Ketl is an AI-driven company specializing in the automatic extraction of in- formation from official documents to facilitate efficient document retrieval using filters. However, the company encounters the challenge of maintaining client doc- ument privacy, prohibiting document sharing between clients while needing data to train their models.

Traditionally, clients classified manually documents that was then used for training a classification and a named entity classification models. With the emer- gence of Large Language Models (LLM), new possibilities arose. LLMs demon- strate impressive document understanding, enabling effective retrieval of critical information without the need for training. However, Ketl refrains from sending client documents to external APIs like OpenAI. Hence, one objective of this study is to assess the feasibility of deploying a smaller, on-premise model.

This thesis investigates LLM usage for named entity retrieval, exploring various prompt techniques, few-shot selection methods, output restriction strategies, and fine-tuning techniques. Due to the absence of labeled official document data, initial focus is directed towards traditional NER tasks, later extending these findings to evaluate LLM performance for Ketl. The models evaluated include deep-learning models like Flair, SpaCy and Wikineural-multinlingual-ner for comparison with the LLMs like ChatGPT and MistralAI. 

Key findings reveal that, in the traditional NER task, while ChatGPT ini- tially outperformed the raw Mistral-7B model, fine-tuning enhanced Mistral-7B’s performance above those of ChatGPT. In comparison with deep-learning mod- els, Flair and SpaCy consistently surpasses LLMs. However, for official document named entity extraction, ChatGPT yields superior results, followed by a fine-tuned MistralAI-7B-Instruct-v0.2 model, and deep-learning models exhibiting only lim- ited capabilities.

Furthermore, a study on LLM confidence assessment proposes training logistic regression on model logits before token generation for the desired output. Given the rapid advancement in the domain of LLM and the associated tools, the poten- tial for integrating LLM models into Ketl’s operations and ease the life of many secretary that need to classify documents manually is real.


## Main results 

You can run the `results.ipynb` to show the main results on the NER part of the thesis.

## Installation 

### To use it on the SDSC machines

First, download miniconda and set it under `/myhome/`

Then run the following :
```bash
export PATH=/myhome/miniconda3/bin:$PATH
pip install -r requirements.txt
spacy download  en_core_web_sm
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
git config --global user.email "mathieu.desponds@ketl.ch"
git config --global user.name "Mathieu Desponds"
cp /myhome/.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
cp /myhome/.ssh/id_rsa /root/.ssh/id_rsa
python setup.py
```
Add the jupyter notebook extension on vscode

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

The default config should be changed so that we have a proxy

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