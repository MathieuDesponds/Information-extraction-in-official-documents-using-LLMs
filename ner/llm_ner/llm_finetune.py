from datasets import Dataset
from langchain.prompts import PromptTemplate

from ner.Datasets.Conll2003Dataset import Conll2003Dataset
from ner.llm_ner.prompts import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import random

from tqdm import tqdm


############ Prompt ###############

def get_ft_prompt(tag, question, answer = "", inference = False):
  return PromptTemplate(
      input_variables = ["tag", 'precision', 'question', 'answer'],
      template = """### SYSTEM: The task is to label all the {tag} in the given sentence. {precision} When you find one add '@@' at the begining and '##' at the end of the entity.
### QUESTION: {question}
### ANSWER: <start_answer>{answer}""").format(
      tag = mapping_abbr_string_ner[tag],
      precision = precision_ner[tag],
      question = question,
      answer = answer
    ) + ("" if inference else "<end_answer>")

def get_ft_prompt_2( question, answer = "", inference = False):
  return PromptTemplate(
      input_variables = ['precision', 'question', 'answer'],
      template = """### SYSTEM: The task is to label all the entities in the given sentence that are either person, organization, location or miscellaneous. When you find one add '<[entity type]>' at the begining and '</entity type>' at the end of the entity.
### QUESTION: {question}
### ANSWER: <start_answer>{answer}""").format(
      question = question,
      answer = answer
    ) + ("" if inference else "<end_answer>")


########### Model #################

def load_model_tokenizer_for_training(base_model_id : str):
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

  model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
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

  tokenizer = get_tokenizer(base_model_id)
  return model, tokenizer

def load_model_tokenizer_for_inference(ft_path: str, base_model_id = "mistralai/Mistral-7B-v0.1") :
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

  base_model = AutoModelForCausalLM.from_pretrained(
      base_model_id,  # Llama 2 7B, same as before
      quantization_config=bnb_config,  # Same quantization config as before
      device_map="auto",
      trust_remote_code=True
  )

  tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token

  ft_model = PeftModel.from_pretrained(base_model, ft_path)

  ft_model = accelerator.prepare_model(ft_model)

  return ft_model, tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

########### Tokenizer #################

def get_tokenizer(base_model_id :str) -> AutoTokenizer : 
  tokenizer = AutoTokenizer.from_pretrained(
      base_model_id,
      padding_side="left",
      add_eos_token=True,
      add_bos_token=True,
  )
  tokenizer.pad_token = tokenizer.eos_token
  return tokenizer

def tokenize_prompt(prompt, tokenizer, max_length = 256):
    result = tokenizer(
        prompt['text'],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


############ Training ################

def split_train_test(tokenized_dataset, train_size, test_size):
   
  rd_idx = random.choices(range(len(tokenized_dataset)), k = train_size+test_size+1)
  splitted = tokenized_dataset.select(rd_idx).train_test_split(test_size = test_size)
  tokenized_train_dataset, tokenized_val_dataset = splitted['train'], splitted['test']
  
  print(f"The model will be trained on {len(tokenized_train_dataset)} train samples and on {len(tokenized_val_dataset)} evaluation samples")

  return tokenized_train_dataset, tokenized_val_dataset


############ Evaluation ################

def evaluate(dataset : Conll2003Dataset, ft_model, tokenizer,tags = ["PER", "LOC", "ORG", "MISC"]) :
  output = []
  ft_model.eval()
  for datapoint in dataset : 
    for tag in tags :
      prompt = get_ft_prompt(tag, datapoint['text'], inference = True)
      model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
      del model_input['token_type_ids']
      with torch.no_grad():
          output.append(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=50, pad_token_id=2, stop = ["<end_answer>"])[0], skip_special_tokens=True))






  