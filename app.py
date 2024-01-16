import logging
from flask import Flask, request
from llm.LLMModel import MistralAI, NoLLM

BASE_PATH = '/custom-nginx/my-app'

app = Flask(__name__)

# model = MistralAI(quantization="Q8_0")
# model.add_grammar("json")

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
    answer =  model(data['messages'], with_full_message=False)
    print(answer)
    logging.debug(f"logging answer { answer}")
    return answer


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=45505)
