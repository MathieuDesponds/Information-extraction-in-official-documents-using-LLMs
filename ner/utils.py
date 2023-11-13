import nltk
from nltk.corpus import stopwords
import string

from sentence_transformers import SentenceTransformer
import torch

import os
import logging

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from transformers import AutoTokenizer, AutoModelForTokenClassification

import numpy as np
# model_path_chat = os.getenv("MODEL_PATH_CHAT")

# if not model_path_chat :
#     logging.error("MODEL_PATH environment variable not set")
#     exit(1)
language = "en"
# Download the stopwords dataset if you haven't already
nltk.download('stopwords')
# Define a set of stopwords and punctuation
punctuation = set(string.punctuation)
if language == 'en' :
    model_name_ner = "dslim/bert-base-NER"
    stop_words = set(stopwords.words('english'))
elif language == 'fr' :
    model_name_ner = "Jean-Baptiste/camembert-ner"
    stop_words = set(stopwords.words('french'))
tokenizer_ner = AutoTokenizer.from_pretrained(model_name_ner)
model_ner = AutoModelForTokenClassification.from_pretrained(model_name_ner)
stop_words.update(['[CLS]', '[SEP]'])

sentence_transformer = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')



# Load pre-trained BERT NER model and tokenizer
def get_embbeding(sentence, idx = -1) :
    # Tokenize the sentence
    tokens = tokenizer_ner.tokenize(tokenizer_ner.decode(tokenizer_ner.encode(sentence)))

    # Remove stopwords and punctuation
    filtered_tokens = [word for word in tokens if word.lower() and word not in punctuation and word not in ['[CLS]', '[SEP]']]

    if not filtered_tokens:
            return []
    # Convert tokens to IDs
    input_ids = tokenizer_ner.convert_tokens_to_ids(filtered_tokens)

    # Convert input_ids to a tensor
    input_tensor = torch.tensor([input_ids])
    # Forward pass through the model
    with torch.no_grad():
        outputs = model_ner(input_tensor)

    # Get the hidden states or embeddings for each token
    hidden_states = outputs[0]

    # Extract embeddings for each token
    token_embeddings = hidden_states[0]

    return [{'embedding' : emb, 'token' :tok, 'idx' : idx} for emb, tok in zip(token_embeddings, filtered_tokens)]

def get_llm(model_path_chat):
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=model_path_chat,
            temperature=0,
            max_tokens=4096,
            n_ctx = 4096,
            n_batch=512,
            n_threads=12,
            top_p=1,
            n_gpu_layers=1000,
            callback_manager=callback_manager,
            repeat_penalty=1.0,
            verbose = False
        )
        return llm

import scipy.stats as stats
import numpy as np

def get_student_conf_interval(scores):
    # Calculate the mean and standard error of the mean (SEM)
    sample_mean = np.mean(scores)
    sample_size = len(scores)
    sem = stats.sem(scores)

    # Degrees of freedom for a small sample
    degrees_of_freedom = sample_size - 1

    # Confidence level
    confidence_level = 0.95

    # Calculate the t-score for the desired confidence level and degrees of freedom
    if degrees_of_freedom : 
        t_score = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

        # Calculate the margin of error
        margin_of_error = t_score * (sem / (sample_size ** 0.5))

    else :
        margin_of_error = np.nan

    # Calculate the confidence interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return round(sample_mean, 3), "({:.3f}, {:.3f})".format(lower_bound, upper_bound)

import subprocess
def run_command(command):
    # Use subprocess.run() to execute the command
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check the result
    if result.returncode == 0:
        print("Command executed successfully")
        print("Output:")
        print(result.stdout)
    else:
        print("Command failed")
        print("Error:")
        print(result.stderr)

import pickle 
def dump(obj, file_path):
    file_splitted = file_path.split('/')
    directory_path = '/'.join(file_splitted[:-1])+'/'
    file_name = file_splitted[-1]
    # Ensure that the directory exists; create it if it doesn't
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open(file_path, 'wb') as f :
        pickle.dump(obj, f) 

def load(file_path):
    try:
        with open(file_path, 'rb') as f :
            return pickle.load(f) 
    except Exception as e:
        # Handle any exceptions that may occur during loading
        print(f"Error loading file {file_path}: {str(e)}")