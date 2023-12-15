from abc import abstractmethod
from ner.Datasets.MyDataset import MyDataset
from ner.llm_ner.confidence_checker import ConfidenceChecker
from ner.llm_ner.prompt_techniques.pt_get_entities import PT_GetEntities
from ner.llm_ner.prompt_techniques.pt_tagger import PT_Tagger
from datasets import Dataset
from ner.llm_ner.few_shots_techniques import *

from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompts import *

import re

class PT_Multi_PT(PromptTechnique):
    def __init__(self, pts : list[PromptTechnique], with_precision = False):
        """
        pts : a list of PromptTechnique where the input output names of the prompts should be 
                (sentence, output), (input, output), ... , (input, output)
        """ 
        super().__init__(None,  with_precision = with_precision)
        self.pts = pts

    @staticmethod
    def name():
        return 'multi_prompt'
    
    def __str__(self):
        return 'multi_prompt-'+'-'.join([pt.__str__() for pt in self.pts])
    
    @abstractmethod
    def run_prompt(self, llm : "LLMModel", 
                   sentence : str, 
                   verifier : "Verifier" = None, 
                   confidence_checker : ConfidenceChecker= None,
                   tags = ["PER", "ORG", "LOC", 'MISC']) :
        pass

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        raise TypeError("This function in multiprompt should not be used")


    def get_prompts_runnable(self, sentence, tags = ["PER", "ORG", "LOC", 'MISC']):
        raise TypeError("This function in multiprompt should not be used")
    
    def process_output(self, response : str, tag : str, tags = None):
        raise TypeError("This function in multiprompt should not be used")
        
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        raise TypeError("This function in multiprompt should not be used")

    def process_dataset_for_finetuning(self, precision = "", 
                                       dataset : MyDataset = None, 
                                       fst : FewShotsTechnique = FST_Sentence,
                                       runs = 2000, 
                                       save = True, 
                                       test_size = 200, 
                                       nb_few_shots = [1,2,3,4]):
        for pt in self.pts :
            pt.process_dataset_for_finetuning(precision, dataset, fst, runs, save, test_size, nb_few_shots)
    
    def load_processed_dataset(self, runs, cleaned = True, precision = None, dataset = "ontonote5"):
        datasets = [pt.load_processed_dataset(runs, cleaned, precision, dataset) for pt in self.pts]

        multi = []
        for i in range(len(datasets[0])):
            for dataset in datasets :
                multi.append(dataset[i])
        
        return Dataset.from_list(multi)
            

class PT_2Time_Tagger(PT_Multi_PT) :
    def __init__(self, fst : FewShotsTechnique , with_precision = False, plus_plus = False, prompt_template = prompt_template_ontonotes):
        """
        pts : a list of PromptTechnique where the input output names of the prompts should be 
                (sentence, output), (input, output), ... , (input, output)
        """         
        super().__init__(pts = [PT_GetEntities(fst, plus_plus= plus_plus, prompt_template = prompt_template, with_precision=with_precision), PT_Tagger(fst,plus_plus= plus_plus, with_precision=with_precision, prompt_template = prompt_template)],
                          with_precision = with_precision)
        
    def run_prompt(self, llm : "LLMModel", 
                   sentence : str, 
                   verifier : "Verifier" = None, 
                   confidence_checker : ConfidenceChecker= None,
                   tags = ["PER", "ORG", "LOC", 'MISC']) :
        output, response_all = self.pts[0].run_prompt(llm, sentence, verifier, None)
        # print(f"Output after the first prompt : {output}")
        for pt in self.pts[1:]:
            output, response_all = pt.run_prompt(llm, [output,sentence], verifier, confidence_checker)
        return output, response_all