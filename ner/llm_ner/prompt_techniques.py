from abc import ABC, abstractmethod
import ast
import pickle
import random

from langchain import FewShotPromptTemplate, PromptTemplate
from tqdm import tqdm
from ner.Datasets.Conll2003Dataset import load_conll_dataset
from ner.Datasets.MyDataset import MyDataset
from datasets import Dataset, concatenate_datasets

from ner.llm_ner.few_shots_techniques import FST_Sentence, FewShotsTechnique

from ner.llm_ner.prompts import *

import re

import spacy

nlp = spacy.load("en_core_web_sm")  # Load a spaCy language model


class PromptTechnique(ABC):
    def __init__(self,fst : FewShotsTechnique, type :str):
        self.fst = fst
        self.type = type

    def __str__(self) -> str:
        return self.type

    @abstractmethod
    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        pass
    
    @abstractmethod
    def get_prompts_runnable(self, sentence):
        pass

    @abstractmethod
    def get_gold(self, dataset : MyDataset) -> list[str]:
        pass

    @abstractmethod
    def process_output(self, response : str, tag : str):
        pass

    def get_few_shots(self, sentence : str, tag : str, nearest_neighbors : list)-> str:
        nearest_neighbors = self.process_nearest_neighbors(nearest_neighbors, tag)
        if nearest_neighbors :
            return """### ASSISTANT : Yes I can do that. Can you provide me examples ?  
### USER : Yes of course, there are some examples : \n""" + few_shot_prompt(nearest_neighbors).format()
        else : 
            return ""
        
    
    def process_dataset_for_finetuning(self, dataset : MyDataset = None, runs = 2000, save = True, test_size = 400):
        if not dataset :
            dataset = load_conll_dataset(split = 'train', cleaned = True)
        old_fst = self.fst
        all_datas = []
        for i in range(runs//test_size):
            seed = random.randint(0,2156867)
            data_tr_tr, data_tr_te = dataset.train_test_split(test_size = test_size, seed=seed)
            data_tr_tr.select(range(1600))
            self.fst = FST_Sentence(data_tr_tr, -1)
            processed_data = self.process_dataset_for_finetuning_helper(data_tr_te)
            all_datas.append(processed_data)
        
        merged_datasets = concatenate_datasets(all_datas)
        with open(f"./ner/saves/datasets/conll2003_for-ft_{'cleaned_' if dataset.cleaned else ''}_{self.type}_{runs}.pkl", 'wb')as f:
            pickle.dump(merged_datasets,f)

        self.fst = old_fst
        return merged_datasets


    def process_dataset_for_finetuning_helper(self, dataset_test : MyDataset):
        output = []
        gold = self.get_gold(dataset_test)
        for i, sample in tqdm(enumerate(dataset_test)) :
            self.fst.nb_few_shots = random.randint(1,4)
            for prompt, tag in self.get_prompts_runnable(sample['text']):
                output.append({'text' : f"{prompt}{gold[i]} <end_output>"})

        processed_dataset = Dataset.from_list(output)  
        return processed_dataset
    
    def load_processed_dataset(self, runs, cleaned = True):
        with open(f"./ner/saves/datasets/conll2003_for-ft_{'cleaned_' if cleaned else ''}_{self.type}_{runs}.pkl", 'rb')as f:
            return pickle.load(f)

class PT_GPT_NER(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique):
        super().__init__(fst, type = '@@##')

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        nearest_neighbors = [{
                "text" : row['text'],
                "output_text" : row['llama_text'][tag]} for row in nearest_neighbors]
        return nearest_neighbors


    def get_prompts_runnable(self, sentence, tags = ["PER", "ORG", "LOC", 'MISC']):
        nearest_neighbors = self.fst.get_nearest_neighbors(sentence)
        prompts = []
        for tag in tags :
            few_shots = self.get_few_shots(sentence, tag, nearest_neighbors)
            prompt = prompt_template[self.type].format(tag=mapping_abbr_string_ner[tag], 
                                        precision = precision_ner[tag],
                                        few_shots = few_shots,
                                        sentence = sentence)
            prompts.append(prompt)
        return list(zip(prompts,tags))
    
    def process_output(self, response : str, tag : str):
        pattern = r'@@\s*(.*?)##'
        named_entities = re.findall(pattern, response, re.DOTALL)
        return [(ne, tag) for ne in named_entities]
        
    def get_gold(self, dataset : MyDataset) -> list[str]:
        return dataset['llama_text']


class PT_OutputList(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique):
        super().__init__(fst, type = 'discussion')

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        nearest_neighbors = [{
                "text" : row['text'],
                "output_text" : row['spans']} for row in nearest_neighbors]
        return nearest_neighbors
    
    def get_prompts_runnable(self, sentence):
        nearest_neighbors = self.fst.get_nearest_neighbors(sentence)
        prompt =  prompt_template[self.type].format(sentence = sentence,
                                            few_shots = self.get_few_shots(sentence, [], nearest_neighbors))
        return [(prompt, "None")]
    
    def process_output(self, response : str, tag : str):
        start_index = response.find('[[')  # Find the opening curly brace
        end_index = response.rfind(']]')    # Find the closing curly brace
        
        if start_index != -1 and end_index != -1:
            response = response[start_index:end_index+2]
        else:
            response ="[]"
    
        try:
            named_entities = ast.literal_eval(response)
        except Exception as e:
            named_entities = []
        
        named_entities = list(set([(ne, tag ) for ne,tag in named_entities if tag in ["PER", "ORG", "LOC", 'MISC']]))
        return named_entities
    
    def get_gold(self, dataset : MyDataset) -> list[str]:
        return dataset['spans']

class PT_Wrapper(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique):
        super().__init__(fst, type = '<>')

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        nearest_neighbors = [{
                "text" : row['text'],
                "output_text" : row['llama_text_2']} for row in nearest_neighbors]
        return nearest_neighbors
    
    def get_prompts_runnable(self, sentence):
        nearest_neighbors = self.fst.get_nearest_neighbors(sentence)
        prompt =  prompt_template[self.type].format(sentence = sentence,
                                            few_shots = self.get_few_shots(sentence, [], nearest_neighbors))
        return [(prompt, "None")]
    
    def process_output(self, response : str, tag : str):
        pass
        # ToDO : process output

    
    def get_gold(self, dataset : MyDataset) -> list[str]:
        return dataset['llama_text_2']
        