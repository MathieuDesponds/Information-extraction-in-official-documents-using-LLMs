
from abc import ABC, abstractmethod
import logging
import os
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch
from ner.llm_ner.llm_finetune import load_model_tokenizer_for_inference

from llama_cpp import Llama


class LlamaLoader(ABC) :
    def __init__(self, temperature = 0, top_p = 1, stop = ["<end_output>", "\n\n\n", "<end_answer>", '</start_answer>', '</start_output>'], max_tokens = 216) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop

    @abstractmethod
    def get_llm_instance(self, model_path, lora_path = None) :
        pass 

    @abstractmethod
    def __call__(self, prompt, with_full_message = False):
        pass

class Llama_LlamaCpp(LlamaLoader) : 
    def __init__(self, temperature=0, top_p=1, stop=["<end_output>", "\n\n\n", '}'], max_tokens=216) -> None:
        super().__init__(temperature, top_p, stop, max_tokens)
    def get_llm_instance(self, model_path, lora_path = None):

        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Make sure the model path is correct for your system!
        llm = Llama(
            model_path= model_path, #"./llama_ft/llama2-7b-llamma-ner-finetune/checkpoint-375/ggml-adapter-model.bin",
            lora_base=model_path,
            lora_path=lora_path,
            n_ctx = 4096,
            n_batch=512,
            logits_all= True,
            n_gpu_layers=100,
            # callback_manager=callback_manager,
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
    def __init__(self, temperature=0, top_p=0.01, stop=["<end_output>", "\n\n\n", '}'], max_tokens=216) -> None:
        super().__init__(temperature, top_p, stop, max_tokens)

    def get_llm_instance(self,  model_path, lora_path = None):

        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Make sure the model path is correct for your system!
        llm = LlamaCpp(
            model_path= model_path, #"./llama_ft/llama2-7b-llamma-ner-finetune/checkpoint-375/ggml-adapter-model.bin",#
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n_ctx = 4096,
            seed = 42,
            n_batch=512,
            n_threads=12,
            top_p=self.top_p,
            n_gpu_layers=100,
            # callback_manager=callback_manager,
            repeat_penalty=1.2,
            verbose = False
        )
        self.model = llm
        return self
    
    def __call__(self, prompt, with_full_message = False):
        output = self.model.invoke(prompt, stop = self.stop)
        if with_full_message :
            return output, {
                'choices': [
                    {'text' :  output }
                    ]
                    }
        else :
            return output
        
class Llama_HF(LlamaLoader) : 
    def __init__(self, temperature=0, top_p=1, stop=["<end_output>", "\n\n\n", '}'], max_tokens=216) -> None:
        super().__init__(temperature, top_p, stop, max_tokens)
    
    def get_llm_instance(self, model_path, lora_path = None):
        llm, tokenizer =load_model_tokenizer_for_inference(ft_path = lora_path)
        llm.save_pretrained(lora_path, safe_serialization = False)
        self.tokenizer = tokenizer
        self.bad_words_ids = self.tokenizer(self.stop, add_special_tokens=False).input_ids 
        self.model = llm
        return self

    def __call__(self, prompt, with_full_message = False):
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        if  'token_type_ids' in model_input :
            del model_input['token_type_ids']
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.model.generate(
                    **model_input, 
                    max_new_tokens=self.max_tokens, 
                    pad_token_id=2, 
                    temperature = self.temperature, 
                    top_p = self.top_p,
                    bad_words_ids=self.bad_words_ids
                )[0], skip_special_tokens=True)
        if with_full_message :
            return output, output 
        return output