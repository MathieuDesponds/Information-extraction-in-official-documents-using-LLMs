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

from ner.Datasets.Conll2003Dataset import get_test_cleaned_split as conll_get_test_cleaned_split
from ner.Datasets.OntoNotes5Dataset import get_test_cleaned_split as ontonote_get_test_cleaned_split

from ner.llm_ner.ResultInstance import ResultInstance, ResultInstanceWithConfidenceInterval, save_result_instance_with_CI
from ner.llm_ner.confidence_checker import ConfidenceChecker
from ner.llm_ner.verifier import Verifier
from ner.llm_ner.few_shots_techniques import *
from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompt_techniques.pt_discussion import PT_OutputList
from ner.llm_ner.prompt_techniques.pt_gpt_ner import PT_GPT_NER
from ner.llm_ner.prompt_techniques.pt_wrapper import PT_Wrapper
from ner.llm_ner.prompt_techniques.pt_multi_pt import PT_Multi_PT, PT_2Time_Tagger
from ner.llm_ner.prompt_techniques.pt_tagger import PT_Tagger
from ner.llm_ner.prompt_techniques.pt_get_entities import PT_GetEntities

from ner.llm_ner.llm_finetune import load_model_tokenizer_for_training, split_train_test, tokenize_prompt

from ner.utils import run_command, latex_escape
from ner.llm_ner.prompts import prompt_template, prompt_template_ontonotes

from llm.LlamaLoader import LlamaLoader, Llama_LlamaCpp, Llama_Langchain

class LLMModel(ABC):
    def __init__(self, base_model_id, base_model_name, check_nb_tokens = True, max_tokens = 256, quantization = "Q5_0", llm_loader : LlamaLoader = None) -> None:
        self.base_model_id = base_model_id
        self.base_model_name = base_model_name.lower()
        self.name = self.base_model_name

        self.max_tokens = max_tokens
        
        if not llm_loader :
            llm_loader = Llama_Langchain()
        self.llm_loader = llm_loader
        self.model : LlamaLoader = self.get_model(quantization = quantization)
        self.check_nb_tokens = check_nb_tokens
        if check_nb_tokens :
            self.nlp = spacy.load("en_core_web_sm")  # Load a spaCy language model
            
    
    def get_model(self, quantization = 'Q5_0', gguf_model_path = ""):
        if gguf_model_path :
            model_path = gguf_model_path
        else :
            model_path = f"llm/models/{self.base_model_name}/{self.base_model_name}.{quantization}.gguf"

        return self.llm_loader.get_llm_instance(model_path)
        
    
    def __str__(self) -> str:
        return self.name

    def __call__(self, prompt, with_full_message) -> Any:
        return self.model(prompt, with_full_message)
        # return prompt
    
    def invoke_mulitple(self, sentences : list[str], pt : PromptTechnique, verifier : Verifier, confidence_checker : ConfidenceChecker, tags = ["PER", "ORG", "LOC", 'MISC']):
        all_entities = []
        for sentence in tqdm(sentences) :
            all_entities.append(self.invoke(sentence, pt, verifier, confidence_checker)[0], tags)
        return all_entities
    
    
    def invoke(self, sentence : str, pt : PromptTechnique, verifier : Verifier, confidence_checker : ConfidenceChecker, tags):
        all_entities, response_all= pt.run_prompt(self, sentence, verifier, confidence_checker, tags)
        return all_entities, response_all
    
    @staticmethod
    def show_prompts(pts : list[PromptTechnique] = [PT_OutputList, PT_Wrapper, PT_Tagger, PT_GetEntities, PT_GPT_NER],
                       nb_few_shots = [5], verifier = False, dataset_loader = ontonote_get_test_cleaned_split,
                       tags = ['CARDINAL', 'ORDINAL', 'WORK_OF_ART', 'PERSON', 'LOC', 'DATE', 'PERCENT', 'PRODUCT', 'MONEY', 'FAC', 'TIME', 'ORG', 'QUANTITY', 'LANGUAGE', 'GPE', 'LAW', 'NORP', 'EVENT']) :
        
        data_train, data_test = dataset_loader()
        fst = FST_Sentence(data_train)
        for i_pt in pts : 
            print()
            print(f"------------{i_pt.name()}-------------------------")
            for plus_plus in [False, True] :
                print(f'---------------------{"prompt++" if plus_plus else "raw"}---------------------')
                print()
                pt = i_pt(fst, with_precision = False, prompt_template = prompt_template_ontonotes, plus_plus = plus_plus)
                print(latex_escape(pt.get_prompts_runnable(data_test[0]['text'], tags)[0][0]))
                print("----------------------------------------------------")

    def classical_test_ontonote5(self, 
                       fsts : list[FewShotsTechnique]= [FST_NoShots, FST_Sentence], 
                       pts : list[PromptTechnique] = [PT_GPT_NER, PT_OutputList, PT_Wrapper],
                       nb_few_shots = [5], 
                       verifier = False, 
                       confidence_checker = False, 
                       save = True, 
                       nb_run_by_test = 3,
                       with_precision = False,
                       prompt_template = prompt_template_ontonotes,
                       plus_plus = False,
                       dataset_loader = ontonote_get_test_cleaned_split,
                       tags = ['CARDINAL', 'ORDINAL', 'WORK_OF_ART', 'PERSON', 'LOC', 'DATE', 'PERCENT', 'PRODUCT', 'MONEY', 'FAC', 'TIME', 'ORG', 'QUANTITY', 'LANGUAGE', 'GPE', 'LAW', 'NORP', 'EVENT']) :
        return self.classical_test(fsts , 
                       pts,
                       nb_few_shots, 
                       verifier, 
                       confidence_checker, 
                       save, 
                       nb_run_by_test ,
                       with_precision ,
                       prompt_template,
                       plus_plus,
                       dataset_loader = dataset_loader,
                       tags = tags)
    

    def classical_test(self, 
                       fsts : list[FewShotsTechnique]= [FST_NoShots, FST_Sentence, FST_Entity, FST_Random], 
                       pts : list[PromptTechnique] = [PT_GPT_NER, PT_OutputList, PT_Wrapper],
                       nb_few_shots = [5], 
                       verifier = False, 
                       confidence_checker = False, 
                       save = True, 
                       nb_run_by_test = 3,
                       with_precision = False,
                       prompt_template = prompt_template,
                       plus_plus = False,
                       dataset_loader = conll_get_test_cleaned_split,
                       tags = ["PER", "ORG", "LOC", 'MISC']) :

        verifier = Verifier(self, data_train) if verifier else None
        confidence_checker = ConfidenceChecker() if confidence_checker else None

        results : list[ResultInstanceWithConfidenceInterval] = []

        for n in nb_few_shots :
            fsts_i : list[FewShotsTechnique]= [fst(None, n) for fst in fsts]
            for fst in fsts_i :
                print(f"Testing with {fst}")
                pts_i : list[PromptTechnique] = [pt(fst, with_precision = with_precision, prompt_template=prompt_template, plus_plus=plus_plus) for pt in pts]
                for pt in pts_i :
                    print(f"      and {pt}")
                    res_insts = []
                    for run in range(nb_run_by_test) :
                        start_time = time.time()
                        seed = random.randint(0, 1535468)
                        data_train, data_test = dataset_loader(seed = seed)
                        fst.set_dataset(data_train)
                        predictions = self.invoke_mulitple(data_test['text'], pt, verifier, confidence_checker, tags)
                        # Calculate the elapsed time
                        elapsed_time = time.time() - start_time
                        res_insts.append(ResultInstance(
                            model= str(self),
                            nb_few_shots = n,
                            noshots = 'noshots' in str(self),
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

    def classical_test_multiprompt(self, pt : PT_Multi_PT,
                       nb_few_shots = [3], verifier = False, confidence_checker = True, save = True, nb_run_by_test = 10, dataset_loader = conll_get_test_cleaned_split) :
        
        verifier = Verifier(self, data_train) if verifier else None
        confidence_checker = ConfidenceChecker() if confidence_checker else None

        results : list[ResultInstanceWithConfidenceInterval] = []

        res_insts = []
        fst : FewShotsTechnique = pt.pts[0].fst
        for _ in range(nb_run_by_test) :
            start_time = time.time()
            seed = random.randint(0, 1535468)
            data_train, data_test = dataset_loader(seed = seed)
            fst.set_dataset(data_train)
            predictions = self.invoke_mulitple(data_test['text'], pt, verifier, confidence_checker)
            # Calculate the elapsed time
            elapsed_time = time.time() - start_time
            res_insts.append(ResultInstance(
                model= str(self),
                nb_few_shots = fst.nb_few_shots,
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
        results_df = pd.DataFrame([result.get_dict() for result in results])
        return results, results_df
    

    def finetune(self, pt: PromptTechnique, runs = 2000, cleaned = True, precision = None):
        processed_dataset = pt.load_processed_dataset(runs, cleaned= cleaned, precision=precision)
        nb_samples = len(processed_dataset)
        output_dir = f"./llm/models/{self.base_model_name}{f'-{precision}' if precision else ''}/finetuned-{pt.__str__()}-{nb_samples}"

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

    def load_finetuned_model(self, prompt_type, nb_samples = 2000, quantization = "Q5_0", precision = None):
        path_to_lora = f"./llm/models/{self.base_model_name}/finetuned-{prompt_type}-{f'{precision}-' if precision else ''}{nb_samples}"
        print(path_to_lora)
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

        self.name = f"{self.name}-ft-{prompt_type}-{nb_samples}-{quantization}{f'-{precision}' if precision else ''}"
        return self.get_model(gguf_model_path =  model_out)


class Llama13b(LLMModel):
    def __init__(self, base_model_id = "meta-llama/Llama-2-13b-hf", base_model_name = "Llama-2-13b", llm_loader = None) -> None:
        super().__init__(base_model_id, base_model_name, llm_loader=llm_loader)
    
    @staticmethod
    def name():
        return "Llama-2-13b".lower()


class Llama7b(LLMModel):
    def __init__(self, base_model_id = "meta-llama/Llama-2-7b-hf", base_model_name = "Llama-2-7b", llm_loader = None) -> None:
        super().__init__(base_model_id, base_model_name, llm_loader=llm_loader)
    
    @staticmethod
    def name():
        return "Llama-2-7b".lower()


class MistralAI(LLMModel):
    def __init__(self, base_model_id = "mistralai/Mistral-7B-v0.1", base_model_name = "Mistral-7B-v0.1", quantization = 'Q5_0', llm_loader = None) -> None:
        super().__init__(base_model_id, base_model_name, quantization=quantization, llm_loader=llm_loader)
    
    @staticmethod
    def name():
        return "Mistral-7B-v0.1".lower()

    
class NoLLM(LLMModel):
    def __init__(self, base_model_id = "None", base_model_name = "None", llm_loader = None) -> None:
        super().__init__(base_model_id, base_model_name, llm_loader=llm_loader)
    
    def __call__(self, prompt, stop = ["<end_output>", "\n\n\n"], with_full_message = False) -> Any:
        if with_full_message :
            return prompt, None
        print(prompt)
        return prompt
    
    def get_model(self, gguf_model_path = "", quantization = ""):
        return None
    
    @staticmethod
    def name():
        return "None".lower()
