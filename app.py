from flask import Flask, request
from llm.LLMModel import MistralAI


app = Flask(__name__)
# model = MistralAI()
@app.route('/', methods=['GET'])
def base():
    return 'Welcome'


@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route('/run_prompt', methods=['POST'])
def run_prompt():
    data = request.get_json()  # Assuming JSON payload, adjust as needed
    if 'prompt' in data:
        return f'Hello, {data["prompt"]}!'
    else:
        return 'Missing "prompt" in the POST request data.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=45505)