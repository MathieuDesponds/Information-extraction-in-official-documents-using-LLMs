from abc import ABC, abstractmethod
import pickle
import random

from tqdm import tqdm
from ner.Datasets.Conll2003Dataset import Conll2003Dataset
from ner.Datasets.MyDataset import MyDataset
from datasets import Dataset, concatenate_datasets

from ner.llm_ner.few_shots_techniques import FST_Sentence, FewShotsTechnique
from ner.llm_ner.prompts import *
from ner.utils import load, dump

class PromptTechnique(ABC):
    def __init__(self,fst : FewShotsTechnique, with_precision = True, prompt_template : dict[PromptTemplate] = prompt_template_ontonotes, plus_plus = False ):
        self.fst = fst
        self.with_precision = with_precision
        self.prompt_template = prompt_template(plus_plus)

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @abstractmethod
    def __str__(self) ->str :
        pass

    @abstractmethod
    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        pass
    
    @abstractmethod
    def get_prompts_runnable(self, sentence, tags):
        pass

    @abstractmethod
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        pass

    @abstractmethod
    def process_output(self, response : str, tag : str, tags):
        pass

    def run_prompt(self, llm : "LLMModel", 
                   sentence : str, 
                   verifier : "Verifier", 
                   confidence_checker : "ConfidenceChecker" = None, 
                   prefix : str = "",
                   tags = ["PER", "ORG", "LOC", 'MISC']) :
        all_entities, all_responses = [], []
        prompts = self.get_prompts_runnable(sentence, tags)
        for prompt,tag in prompts :
            if llm.check_nb_tokens :
                doc = llm.nlp(prompt)   
                num_tokens = len(doc)
                # print(num_tokens, prompt)
                if num_tokens > 4096 - llm.max_tokens :
                    print("prompt is too big") 
                    continue

            reponse_text, response_all = llm(prompt, with_full_message =True)
            processed_response = self.process_output(prefix + reponse_text, tag, tags = tags)
            if verifier : 
                processed_response = verifier.verify(sentence, processed_response, llm)
            if confidence_checker :
                processed_response = confidence_checker.check(sentence, processed_response, llm)
            all_entities.extend(processed_response)
            all_responses.append(response_all)
        return all_entities, all_responses[0]

    def get_precision(self):
        if self.with_precision :
            return """### ASSISTANT : Can you give me clarification on the different type of entities ? 
### USER : Yes. """+'\n'.join([val for key, val in precision_ner.items()])+'\n'
        else :
            return ""

    def get_few_shots(self, sentence : str, tag : str, nearest_neighbors : list)-> str:
        nearest_neighbors = self.process_nearest_neighbors(nearest_neighbors, tag)
        if nearest_neighbors :
            return """### ASSISTANT : Can you provide me examples ?  
### USER : There are examples : \n""" + few_shot_prompt(nearest_neighbors).format()+'\n'
        else : 
            return ""
        
    
    def process_dataset_for_finetuning(self, precision = "", 
                                       dataset : MyDataset = None, 
                                       fst : FewShotsTechnique = FST_Sentence,
                                       runs = 2000, 
                                       save = True, 
                                       test_size = 200, 
                                       nb_few_shots = [1,2,3,4]):
        if runs < test_size :
            test_size = runs
        if not dataset :
            dataset = MyDataset.my_load_dataset(dataset=Conll2003Dataset, split = 'train', cleaned= True)
        old_fst = self.fst
        all_datas = []
        for i in range(runs//test_size):
            seed = random.randint(0,2156867)
            data_tr_tr, data_tr_te = dataset.train_test_split(test_size = test_size, seed=seed)
            data_tr_tr.select(range(800))
            self.fst = fst(data_tr_tr, -1)
            processed_data = self.process_dataset_for_finetuning_helper(data_tr_te, nb_few_shots)
            all_datas.append(processed_data)
        
        merged_datasets = concatenate_datasets(all_datas)
        
        if save :
            path = f"./ner/saves/datasets/{dataset.name()}_for-ft_{'cleaned_' if dataset.name() == 'conll2003' and dataset.cleaned else ''}{self.__str__()}_{f'{precision}_' if precision else ''}{runs}.pkl"
            dump(merged_datasets,path)

        self.fst = old_fst
        return merged_datasets


    def process_dataset_for_finetuning_helper(self, dataset_test : MyDataset, nb_few_shots):
        output = []
        for i, sample in tqdm(enumerate(dataset_test)) :
            self.fst.nb_few_shots = random.choice(nb_few_shots)
            prompt, tag = random.choice(self.get_prompts_runnable(sample['text'], tags = dataset_test.get_tags()))
            gold = self.get_gold(dataset_test, tag)
            output.append({'text' : f"{prompt}{gold[i]} <end_output>"})

        processed_dataset = Dataset.from_list(output)  
        return processed_dataset
    
    def load_processed_dataset(self, runs, cleaned = True, precision = None, dataset = "ontonote5"):
        path = f"./ner/saves/datasets/{dataset}_for-ft_{'cleaned_' if cleaned else ''}{self.__str__()}_{f'{precision}_' if precision else ''}{runs}.pkl"
        return load(path)