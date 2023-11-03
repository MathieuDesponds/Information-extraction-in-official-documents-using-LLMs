from abc import ABC, abstractmethod
import datetime
from typing import Any

from tqdm import tqdm
import pandas as pd
import os
import spacy
import transformers

from ner.Datasets.Conll2003Dataset import get_test_cleaned_split

from ner.llm_ner.ResultInstance import ResultInstance, save_result_instance
from ner.llm_ner.verifier import Verifier
from ner.llm_ner.few_shots_techniques import *
from ner.llm_ner.prompt_techniques import PromptTechnique, PT_GPT_NER, PT_OutputList

from ner.utils import run_command
from ner.llama2_finetune import load_model_tokenizer_for_training, split_train_test, tokenize_prompt


class LLMModel(ABC):
    def __init__(self, base_model_id, base_model_name, check_nb_tokens = True, max_tokens = 256) -> None:
        self.base_model_id = base_model_id
        self.base_model_name = base_model_name 

        self.max_tokens = max_tokens
        
        self.model = self.get_model()
        self.check_nb_tokens = check_nb_tokens
        if check_nb_tokens :
            self.nlp = spacy.load("en_core_web_sm")  # Load a spaCy language model
            
    
    @abstractmethod
    def get_model(self, gguf_model_path = ""):
        return None
    
    def __str__(self) -> str:
        return self.model_name
    
    def __call__(self, prompt, stop = ["<end_output>", "\n\n\n"] ) -> Any:
        # return self.model(prompt, stop = ["<end_output>", "\n\n\n"], max_tokens = self.max_tokens)
        return prompt
    
    def invoke_mulitple(self, sentences : list[str], pt : PromptTechnique, verifier : Verifier):
        all_entities = []
        for sentence in tqdm(sentences) :
            all_entities.append(self.invoke(sentence,pt, verifier))
        return all_entities
    
    
    def invoke(self, sentence : str, pt : PromptTechnique, verifier : Verifier):
        all_entities = []
        prompts = pt.get_prompts_runnable(sentence)
        for prompt,tag in prompts :

            if self.check_nb_tokens :
                doc = self.nlp(prompt)   
                num_tokens = len(doc)
                # print(num_tokens, prompt)
                if num_tokens > 4096 -self.max_tokens :
                    print("prompt is too big") 
                    continue

            response = self(prompt)
            processed_response = pt.process_output(response, tag)
            if verifier : 
                processed_response = verifier.verify(sentence, processed_response)
            all_entities.extend(processed_response)
        return all_entities
    
    def classical_test(self, nb_few_shots = [5], verifier = False, save = False) :
        ## load les datasets ici
        data_train, data_test = get_test_cleaned_split()
        data_test.select([0,1])

        if verifier :
            verifier = Verifier(self, data_train)
        else : 
            verifier = None

        results : list[ResultInstance] = []

        for n in nb_few_shots :
            fsts = [FST_Random(data_train, n), FST_Sentence(data_train, n), FST_Entity(data_train, n)]
            for fst in fsts :
                self.fst = fst
                pts : list[PromptTechnique] = [PT_GPT_NER(fst), PT_OutputList(fst)]
                for pt in pts :
                    predictions = self.invoke_mulitple(data_test['text'], pt, verifier)
                    results.append(ResultInstance(
                        model= self,
                        nb_few_shots = n,
                        prompt_technique = pt,
                        few_shot_tecnique = fst,
                        verifier = verifier,
                        results = predictions,
                        gold = pt.get_gold(data_test),
                        data_test = data_test,
                        data_train = data_train
                    ))
                    if save :
                        save_result_instance(results[-1])
        results_df = pd.DataFrame([result.get_dict() for result in results])
        return results, results_df



    def finetune(self, pt: PromptTechnique, runs = 2000, cleaned = True):
        processed_dataset = pt.load_processed_dataset(runs, cleaned)
        nb_samples = len(processed_dataset)
        output_dir = f"./llm/models/{self.base_model_name}/finetuned-{pt.type}-{nb_samples}"

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
                save_steps=train_size/4//4,                # Save checkpoints every 50 steps
                evaluation_strategy="steps", # Evaluate the model every logging step
                eval_steps=train_size/4//4,               # Evaluate and save checkpoints every 50 steps
                do_eval=True,                # Perform evaluation at the end of training
                report_to="wandb",           # Comment this out if you don't want to use weights & baises
                run_name=f"finetuned-{pt.type}-{nb_samples}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        base_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        trainer.save_model()

    def load_finetuned_model(self, prompt_type, nb_samples, quantization):
        path_to_lora = f"./llm/models/{self.base_model_name}/finetuned-{prompt_type}-{nb_samples}"
        model_out = f"{path_to_lora}/model-{quantization}.gguf"
        if not os.path.exists(model_out):
            model_type = 'llama' #llama, starcoder, falcon, baichuan, or gptneox
            command = f"python3 ./llama.cpp/convert-lora-to-ggml.py {path_to_lora}"
            run_command(command)


            model_base = f"./llm/models/{self.base_model_name}/{self.base_model_name}_{quantization}.gguf"
            lora_scaled = f"{path_to_lora}/ggml-adapter-model.bin"

            # model_out = f"{path_to_lora}/llama-13b-finetuned-2000-v0.gguf"
            # lora_scaled = f"{path_to_lora}/ggml-adapter-model.bin"

            command = f"./llama.cpp/export-lora --model-base {model_base} --model-out {model_out} --lora-scaled {lora_scaled} 1.0"

            run_command(command)
        return self.get_model(gguf_model_path =  model_out)


class Llama13b(LLMModel):
    def __init__(self, base_model_id = "meta-llama/Llama-2-13b-hf", base_model_name = "Llama-2-13b") -> None:
        super().__init__(base_model_id, base_model_name)

    def get_model(self, gguf_model_path = ""):
        pass

class Llama7b(LLMModel):
    def __init__(self, base_model_id = "meta-llama/Llama-2-7b-hf", base_model_name = "Llama-2-7b") -> None:
        super().__init__(base_model_id, base_model_name)

    def get_model(self, gguf_model_path = ""):
        pass

class MistralAI(LLMModel):
    def __init__(self, base_model_id = "mistralai/Mistral-7B-v0.1", base_model_name = "Mistral-7B-v0.1") -> None:
        super().__init__(base_model_id, base_model_name)

    def get_model(self, gguf_model_path = ""):
        pass