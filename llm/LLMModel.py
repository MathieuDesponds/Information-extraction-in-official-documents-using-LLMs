from abc import ABC, abstractmethod
from datetime import datetime
import logging
import time
from typing import Any

from tqdm import tqdm
import pandas as pd
import os
import spacy
import transformers

from ner.Datasets.Conll2003Dataset import get_test_cleaned_split

from ner.llm_ner.ResultInstance import ResultInstance, ResultInstanceWithConfidenceInterval, save_result_instance_with_CI
from ner.llm_ner.verifier import Verifier
from ner.llm_ner.few_shots_techniques import *
from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompt_techniques.pt_discussion import PT_OutputList
from ner.llm_ner.prompt_techniques.pt_gpt_ner import PT_GPT_NER
from ner.llm_ner.prompt_techniques.pt_wrapper import PT_Wrapper
from ner.llm_ner.llm_finetune import load_model_tokenizer_for_training, split_train_test, tokenize_prompt

from ner.utils import run_command


from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from llama_cpp import Llama


class LLMModel(ABC):
    def __init__(self, base_model_id, base_model_name, check_nb_tokens = True, max_tokens = 256, quantization = "Q5_0") -> None:
        self.base_model_id = base_model_id
        self.base_model_name = base_model_name.lower()
        self.name = self.base_model_name

        self.max_tokens = max_tokens
        
        self.model = self.get_model(quantization = quantization)
        self.check_nb_tokens = check_nb_tokens
        if check_nb_tokens :
            self.nlp = spacy.load("en_core_web_sm")  # Load a spaCy language model
            
    
    def get_model(self, quantization = 'Q5_0', gguf_model_path = ""):
        if gguf_model_path :
            model_path = gguf_model_path
        else :
            model_path = f"llm/models/{self.base_model_name}/{self.base_model_name}.{quantization}.gguf"
        self.model = get_llm_llamaCpp(model_path)
        return self.model
    
    def __str__(self) -> str:
        return self.name
    
    def __call__(self, prompt, stop = ["<end_output>", "\n\n\n"] ) -> Any:
        return self.model(prompt, stop = ["<end_output>", "\n\n\n"], max_tokens = self.max_tokens)
        # return prompt
    
    def invoke_mulitple(self, sentences : list[str], pt : PromptTechnique, verifier : Verifier):
        all_entities = []
        for sentence in tqdm(sentences) :
            all_entities.append(self.invoke(sentence, pt, verifier))
        return all_entities
    
    
    def invoke(self, sentence : str, pt : PromptTechnique, verifier : Verifier):
        all_entities = pt.run_prompt(self, sentence, verifier)
        return all_entities
    
    def classical_test(self, fsts : list[FewShotsTechnique]= [FST_NoShots, FST_Sentence, FST_Entity, FST_Random], 
                       pts : list[PromptTechnique] = [PT_GPT_NER, PT_OutputList, PT_Wrapper],
                       nb_few_shots = [5], verifier = False, save = True, nb_run_by_test = 3) :

        
        # data_test.select([0,1])

        if verifier :
            verifier = Verifier(self, data_train)
        else : 
            verifier = None

        results : list[ResultInstanceWithConfidenceInterval] = []

        for n in nb_few_shots :
            fsts_i : list[FewShotsTechnique]= [fst(None, n) for fst in fsts]
            for fst in fsts_i :
                print(f"Testing with {fst}")
                pts_i : list[PromptTechnique] = [pt(fst) for pt in pts]
                for pt in pts_i :
                    print(f"      and {pt}")
                    res_insts = []
                    for run in range(nb_run_by_test) :
                        start_time = time.time()
                        seed = random.randint(0, 1535468)
                        data_train, data_test = get_test_cleaned_split(seed = seed)
                        fst.set_dataset(data_train)
                        predictions = self.invoke_mulitple(data_test['text'], pt, verifier)
                        # Calculate the elapsed time
                        elapsed_time = time.time() - start_time
                        res_insts.append(ResultInstance(
                            model= str(self),
                            nb_few_shots = n,
                            prompt_technique = str(pt),
                            few_shot_tecnique = str(fst),
                            verifier = str(verifier),
                            results = predictions,
                            gold = data_test['spans'],
                            data_test = data_test,
                            data_train = data_train,
                            elapsed_time = elapsed_time,
                            with_precision = pt.with_precision,
                            seed = seed,
                        ))
                        del data_test, data_train
                    results.append(ResultInstanceWithConfidenceInterval(res_insts))
                    if save :
                        save_result_instance_with_CI(results[-1])
                    fst.save_few_shots()
        results_df = pd.DataFrame([result.get_dict() for result in results])
        return results, results_df


    def finetune(self, pt: PromptTechnique, runs = 2000, cleaned = True, precision = None):
        processed_dataset = pt.load_processed_dataset(runs, cleaned)
        nb_samples = len(processed_dataset)
        output_dir = f"./llm/models/{self.base_model_name}/finetuned-{pt.__str__()}-{f'{precision}-' if precision else ''}{nb_samples}"

        test_size = 50
        train_size = nb_samples-test_size
        base_model, tokenizer = load_model_tokenizer_for_training(self.base_model_id)
        tokenized_dataset = processed_dataset.map(lambda row : tokenize_prompt(row, tokenizer))
        tokenized_train_dataset, tokenized_val_dataset = split_train_test(tokenized_dataset, train_size, test_size = test_size)

        trainer = transformers.Trainer(
            model=base_model,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                warmup_steps=1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                num_train_epochs = 1,
                learning_rate=2.5e-4, # Want a small lr for finetuning
                # bf16=True,
                optim="paged_adamw_8bit",
                logging_dir="./logs",        # Directory for storing logs
                save_strategy="steps",       # Save the model checkpoint every logging step
                save_steps=runs/4//5,                # Save checkpoints every 50 steps
                evaluation_strategy="steps", # Evaluate the model every logging step
                eval_steps=runs/4//10,               # Evaluate and save checkpoints every 50 steps
                do_eval=True,                # Perform evaluation at the end of training
                report_to="wandb",           # Comment this out if you don't want to use weights & baises
                run_name=f"finetuned-{pt.__str__}-{nb_samples}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        base_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        trainer.save_model()

    def load_finetuned_model(self, prompt_type, nb_samples = 2000, quantization = "Q5_0"):
        path_to_lora = f"./llm/models/{self.base_model_name}/finetuned-{prompt_type}-{nb_samples}"
        model_out = f"{path_to_lora}/model-{quantization}.gguf"
        if not os.path.exists(model_out):
            model_type = 'llama' #llama, starcoder, falcon, baichuan, or gptneox
            command = f"python3 ../llama.cpp/convert-lora-to-ggml.py {path_to_lora}"
            run_command(command)


            model_base = f"./llm/models/{self.base_model_name}/{self.base_model_name}.{quantization}.gguf"
            lora_scaled = f"{path_to_lora}/ggml-adapter-model.bin"

            # model_out = f"{path_to_lora}/llama-13b-finetuned-2000-v0.gguf"
            # lora_scaled = f"{path_to_lora}/ggml-adapter-model.bin"

            command = f"../llama.cpp/export-lora --model-base {model_base} --model-out {model_out} --lora-scaled {lora_scaled} 1.0"

            run_command(command)

        self.name = f"{self.name}-ft-{prompt_type}-{nb_samples}-{quantization}"
        return self.get_model(gguf_model_path =  model_out)


class Llama13b(LLMModel):
    def __init__(self, base_model_id = "meta-llama/Llama-2-13b-hf", base_model_name = "Llama-2-13b") -> None:
        super().__init__(base_model_id, base_model_name)
    
    @staticmethod
    def name():
        return "Llama-2-13b".lower()


class Llama7b(LLMModel):
    def __init__(self, base_model_id = "meta-llama/Llama-2-7b-hf", base_model_name = "Llama-2-7b") -> None:
        super().__init__(base_model_id, base_model_name)
    
    @staticmethod
    def name():
        return "Llama-2-7b".lower()


class MistralAI(LLMModel):
    def __init__(self, base_model_id = "mistralai/Mistral-7B-v0.1", base_model_name = "Mistral-7B-v0.1") -> None:
        super().__init__(base_model_id, base_model_name)
    
    @staticmethod
    def name():
        return "Mistral-7B-v0.1".lower()

    
class NoLLM(LLMModel):
    def __init__(self, base_model_id = "None", base_model_name = "None") -> None:
        super().__init__(base_model_id, base_model_name)
    
    def __call__(self, prompt, stop = ["<end_output>", "\n\n\n"] ) -> Any:
        return prompt
    
    def get_model(self, gguf_model_path = "", quantization = ""):
        return None
    
    @staticmethod
    def name():
        return "None".lower()




def get_llm_llamaCpp(model_path = None):
    if not model_path:
        model_path = os.getenv("model_path")

    if not model_path:
        logging.error("MODEL_PATH environment variable not set")
        exit(1)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path= model_path, #"./llama_ft/llama2-7b-llamma-ner-finetune/checkpoint-375/ggml-adapter-model.bin",#
        temperature=0,
        max_tokens=150,
        n_ctx = 4096,
        n_batch=512,
        n_threads=12,
        logits_all= True,
        logprobs = 20,
        top_p=1,
        n_gpu_layers=100,
        # callback_manager=callback_manager,
        repeat_penalty=1.0,
        verbose = False
    )
    return llm

def get_llm_Llama(model_path = None):
    if not model_path:
        model_path = os.getenv("model_path")

    if not model_path:
        logging.error("MODEL_PATH environment variable not set")
        exit(1)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm = Llama(
        model_path= model_path, #"./llama_ft/llama2-7b-llamma-ner-finetune/checkpoint-375/ggml-adapter-model.bin",#
        temperature=0,
        max_tokens=150,
        n_ctx = 4096,
        n_batch=512,
        n_threads=12,
        logits_all= True,
        logprobs = 20,
        top_p=1,
        n_gpu_layers=35,
        callback_manager=callback_manager,
        repeat_penalty=1.0,
        verbose = False
    )
    return llm
