
from abc import ABC, abstractmethod
import logging
import os
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from llama_cpp import Llama


class LlamaLoader(ABC) :
    def __init__(self, temperature = 0, top_p = 1, stop = ["<end_output>", "\n\n\n", "<end_answer>", '</start_answer>', '</start_output>'], max_tokens = 216) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop

    @abstractmethod
    def get_llm_instance(self, model_path = None) :
        pass 

    @abstractmethod
    def __call__(self, prompt, with_full_message = False):
        pass

class Llama_LlamaCpp(LlamaLoader) : 
    def __init__(self, temperature=0, top_p=1, stop=["<end_output>", "\n\n\n", '}'], max_tokens=216) -> None:
        super().__init__(temperature, top_p, stop, max_tokens)
    def get_llm_instance(self, model_path = None):

        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Make sure the model path is correct for your system!
        llm = Llama(
            model_path= model_path, #"./llama_ft/llama2-7b-llamma-ner-finetune/checkpoint-375/ggml-adapter-model.bin",
            n_ctx = 2048,
            n_batch=512,
            logits_all= True,
            n_gpu_layers=35,
            callback_manager=callback_manager,
            repeat_penalty=1.3,
            verbose = False
        )
        self.model = llm
        return self

    def __call__(self, prompt, with_full_message = False):
        output = self.model(prompt, 
                   temperature = self.temperature, 
                   top_p = self.top_p, 
                   max_tokens = self.max_tokens, 
                   stop = self.stop,
                #    logprobs = 20
                   )
        if with_full_message :
            return output['choices'][0]['text'], output 
        return output
        

class Llama_Langchain(LlamaLoader) :
    def __init__(self, temperature=0, top_p=1, stop=["<end_output>", "\n\n\n", '}'], max_tokens=216) -> None:
        super().__init__(temperature, top_p, stop, max_tokens)

    def get_llm_instance(self, model_path = None):

        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Make sure the model path is correct for your system!
        llm = LlamaCpp(
            model_path= model_path, #"./llama_ft/llama2-7b-llamma-ner-finetune/checkpoint-375/ggml-adapter-model.bin",#
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n_ctx = 2048,
            seed = 42,
            n_batch=512,
            n_threads=12,
            top_p=self.top_p,
            n_gpu_layers=100,
            callback_manager=callback_manager,
            repeat_penalty=1.0,
            verbose = False
        )
        self.model = llm
        return self
    
    def __call__(self, prompt, with_full_message = False):
        output = self.model(prompt, stop = self.stop)
        if with_full_message :
            return output, {
                'choices': [
                    {'text' :  output }
                    ]
                    }
        else :
            return output