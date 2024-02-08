from datetime import datetime
from itertools import chain
import random
import torch
from typing import Any

from datasets import load_dataset, Dataset, concatenate_datasets
import transformers
from transformers import AutoTokenizer
from myMongoClient import load, dump
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel


tokenizer =  AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

def finetune(dataset_path, model_path = "mistralai/Mistral-7B-Instruct-v0.2", precision = "quora|alpaca|ketl", checkpoint = None, testing = False):
    processed_dataset = load(dataset_path)
    tokenized_dataset = processed_dataset.copy()
    for doc_id, values in tokenized_dataset.items():
        tokenized_dataset[doc_id] = [tokenize_prompt(prompt, tokenizer) for prompt in values]
    samples, doc_names, sets = split_train_test(tokenized_dataset)
    print(doc_names)
    training_samples, eval_samples, testing_samples = samples
    quora = quora_dataset(nb_samples= len(training_samples))
    alpaca = alpaca_dataset(nb_samples= len(training_samples))
    training_samples = Dataset.from_list(training_samples)
    training_samples = concatenate_datasets([training_samples, quora, alpaca]).shuffle(seed = 42)

    output_dir = model_path+f"ketl_training/{len(training_samples)}-{dataset_path.split('/')[-1]}-{precision}"

    base_model = load_model_for_training(model_path)

    trainer = transformers.Trainer(
        model=base_model,
        train_dataset=training_samples,
        eval_dataset=eval_samples,
        tokenizer = tokenizer,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir = True,
            warmup_steps=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1, 
            gradient_accumulation_steps=1,
            num_train_epochs = 1,
            learning_rate=2.5e-4, # Want a small lr for finetuning
            
            # bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",        # Directory for storing logs
            save_safetensors = False,
            report_to="wandb",           # Comment this out if you don't want to use weights & baises
            run_name=f"finetuned-ketl-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    base_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    if checkpoint : 
        trainer.train(resume_from_checkpoint = True)
    trainer.train()
    trainer.save_model()

def split_train_test(processed_dataset, ratio = [0.60,0.05,0.35]):
    assert len(ratio)  ==3
    processed_dataset = [{key : value} for key, value in processed_dataset.items()]
    nb_docs = len(processed_dataset)

    lengths = [int(nb * nb_docs) for nb in ratio]
    
    random.seed(42)
    random.shuffle(processed_dataset)
    
    sets = [processed_dataset[:lengths[0]], processed_dataset[lengths[0]:lengths[0]+lengths[1]],processed_dataset[lengths[0]+lengths[1]:] ]
    samples = []
    doc_names = []
    for sset in sets : 
        samples.append(list(chain(*[val for doc in sset for key,val in doc.items()])))
        doc_names.append([key for doc in sset for key,val in doc.items()])

    return samples,doc_names,sets

def tokenize_prompt(prompt, tokenizer, max_length = 4096):
    result = tokenizer(
        prompt['text']
    )
    result["labels"] = result["input_ids"].copy()
    return result


def load_model_for_training(base_model_id : str):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config,
        trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    # Apply the accelerator. You can comment this out to remove the accelerator.
    model = accelerator.prepare_model(model)

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    return model


def quora_dataset(nb_samples = 200):
    quora = load_dataset("toughdata/quora-question-answer-dataset", split='train[:1%]')
    quora = quora.map(lambda row : {'messages' : [{'role': 'user', 'content': row['question']} , {'role' : 'assistant', 'content' : row['answer']}]})
    quora = quora.map(lambda row : {'text' : tokenizer.apply_chat_template(row['messages'], tokenize=False)})
    quora = quora.map(lambda row : tokenize_prompt(row, tokenizer))
    quora = quora.remove_columns(['question', 'answer', 'text', 'messages'])
    return quora.select(range(nb_samples))

import json
def alpaca_dataset(nb_samples = 200) :
    dataset = []
    # Open the JSON file
    with open('data/chatalpaca-10k.json', 'r') as file:
        for line in file :
            line = json.loads(line)
            dataset.append(line['conversations'])
    mapping = {'gpt' : 'assistant', 'human' : 'user'}

    alpaca = [{'messages' : [{'role' : mapping[message['from']] , 'content' : message['value'] }for message in data]} for data in dataset]
    alpaca = Dataset.from_list(alpaca)
    alpaca = alpaca.select(range(nb_samples))
    alpaca = alpaca.map(lambda row : {'text' : tokenizer.apply_chat_template(row['messages'], tokenize=False)})
    alpaca = alpaca.map(lambda row : tokenize_prompt(row, tokenizer))
    alpaca = alpaca.remove_columns(['text', 'messages'])
    return alpaca