import logging
from flask import Flask, request
from llm.LLMModel import MistralAI, MistralAIInstruct, NoLLM
from llm.LlamaLoader import Llama_Langchain, Llama_LlamaCpp

from ner.utils import dump, load

from datetime import datetime

BASE_PATH = '/custom-nginx/my-app'

BASE_PROMPT = 'hello can you send me your description in a json ?'
BASE_PROMPT_INSTRUCT = [{'role' : 'user', 'content' : BASE_PROMPT}]

prompts = []

doc_type_from_file_name = load("ketl/mongo/doc_type_from_file_name.pkl")

def get_mistral_instruct_model():
    model = MistralAIInstruct(quantization="Q8_0", llm_loader =Llama_LlamaCpp)
    model.set_grammar("doc_type")
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
    data = request.get_json()  # Assuming JSON payload, adjust as needed
    if 'messages' not in data:
        return 'Missing "prompt" in the POST request data.'
    messages = data['messages']
    label = get_prompt_type_from_messages(messages)
    if label == "Type de document" :
        model.set_grammar('doc_type')
    else :
        model.set_grammar('json')

    answer =  model(messages)
    print(answer)
    logging.debug(f"logging answer { answer}")
    return answer

@app.route(BASE_PATH+'/saving_prompt', methods=['POST'])
def saving_prompt():
    data = request.get_json()  # Assuming JSON payload, adjust as needed
    if 'messages' not in data:
        return 'Missing "prompt" in the POST request data.'
    label = get_prompt_type_from_messages(messages)
    prompts.append({
        "messages" : messages,
        "label" : label
    })
    if label == 'Type de document' : 
        start_file_name = messages_string.find("pseudo_")
        if start_file_name == -1 :
            start_file_name = messages_string.find("Pseudo_")
        end_file_name = messages_string.find(".pdf")
        file_name = messages_string[start_file_name, end_file_name+4]

        return "{" + f"'Type de document' : {doc_type_from_file_name[file_name]}" + '}'
    return {}

@app.route(BASE_PATH+'/save_prompt', methods=['GET'])
def save_prompt():
    dump(prompts, f"ketl/prompts_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv""")
    output = f"{len(prompts)} prompts saved" 
    prompts = []

    return output


def get_prompt_type_from_messages(messages):
    messages_string = " ".join([m['content'] for m in messages])
    label = "Type de document" if "Type de document"  in messages_string else ("Client" if "from the client" in messages_string else "Other")
    return label

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=45505)