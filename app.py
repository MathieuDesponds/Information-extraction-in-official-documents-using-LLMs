import logging
import time
from flask import Flask, request
from llm.LLMModel import MistralAI, MistralAIInstruct, NoLLM
from llm.LlamaLoader import Llama_Langchain, Llama_LlamaCpp

from ner.utils import dump, load

from datetime import datetime

BASE_PATH = '/custom-nginx/my-app'

BASE_PROMPT = 'hello can you send me your description in a json ?'
BASE_PROMPT_INSTRUCT = [{'role' : 'user', 'content' : BASE_PROMPT}]

prompts = []

doc_type_from_file_name = load("ketl/mongo/data/doc_type_from_file_name.pkl")

model_available = True

def get_mistral_instruct_model():
    model = MistralAIInstruct(base_model_id = "llm/mistralai/Mistral-7B-Instruct-v0.2ketl_training/2024-02-06_14:35:37_mistral-instruct_ft-data-quora-alpaca-ketl/merged_ggml_q8_0.bin", quantization="Q8_0", llm_loader = Llama_LlamaCpp)
    model.set_grammar("json")
    print(model(BASE_PROMPT_INSTRUCT))
    return model

def get_mistral_model():
    model = MistralAI(quantization="Q8_0", llm_loader =Llama_LlamaCpp)
    model.set_grammar("json")
    print(model(BASE_PROMPT))
    return model

app = Flask(__name__)

model = get_mistral_instruct_model()

@app.route(BASE_PATH+'/', methods=['GET'])
def base():
    return 'Welcome'

@app.route('/', methods=['GET'])
def basebase():
    return 'Welcome Base'


@app.route(BASE_PATH+'/hello', methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route(BASE_PATH+'/run_prompt', methods=['POST'])
def run_prompt():
    global model_available
    data = request.get_json()  # Assuming JSON payload, adjust as needed
    if 'messages' not in data:
        return 'Missing "prompt" in the POST request data.'
    messages = data['messages']
    label = get_prompt_type_from_messages(messages)
    if label == "Type de document" :
        model.set_grammar('doc_type')
    else :
        model.set_grammar('string_json')


    while(not model_available):
        time.sleep(1)
    model_available = False
    answer =  model(messages)
    model_available =True
    print(answer)
    logging.debug(f"logging answer { answer}")
    return answer

@app.route(BASE_PATH+'/saving_prompt', methods=['POST'])
def saving_prompt():
    data = request.get_json()  # Assuming JSON payload, adjust as needed
    if 'messages' not in data:
        return 'Missing "prompt" in the POST request data.'
    
    messages = data['messages']
    label = get_prompt_type_from_messages(messages)
    file_name = get_filename_from_messages(messages)
    print(f"filenname :{file_name}")
    
    if file_name not in doc_type_from_file_name :
        print(f"Filename not found for '{file_name}'")

    if label == 'Type de document' and file_name in doc_type_from_file_name :
        output = "{" + f""" "Type de document" : "{doc_type_from_file_name[file_name]}" """ + '}'
    else : 
        output = "{}"

    prompts.append({
        "messages" : messages,
        "label" : label,
        "file_name" : file_name,
        "fields_name_key" : data['fields_name_key'],
        "output" : output
    })
    return output

@app.route(BASE_PATH+'/save_prompts', methods=['POST'])
def save_prompts():
    global prompts
    dump(prompts, f"ketl/mongo/data/saved_prompts/prompts_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv""")
    output = f"{len(prompts)} prompts saved" 
    prompts = []
    return output


def get_prompt_type_from_messages(messages):
    messages_string = " ".join([m['content'] for m in messages])
    label = "Type de document" if "Type de document"  in messages_string else ("Client" if "from the client" in messages_string else "Other")
    return label

def get_filename_from_messages(messages) :
    messages_string = " ".join([m['content'] for m in messages])
    start_file_name = messages_string.find("pseudo_")
    if start_file_name == -1 :
        start_file_name = messages_string.find("Pseudo_")
    if start_file_name == -1 :
        start_file_name = messages_string.find("131338")
    end_file_name = messages_string.find(".pdf")
    file_name = messages_string[start_file_name: end_file_name+4]
    return file_name

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=45505)