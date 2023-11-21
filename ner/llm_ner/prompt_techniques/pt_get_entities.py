import ast
from ner.Datasets.MyDataset import MyDataset
from ner.llm_ner.confidence_checker import ConfidenceChecker
from ner.llm_ner.few_shots_techniques import FewShotsTechnique

from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompts import *

import re

class PT_GetEntities(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique, with_precision = True, prompt_template : dict[PromptTemplate] = prompt_template, plus_plus = False ):
        super().__init__(fst, with_precision, prompt_template, plus_plus)

    @staticmethod
    def name():
        return 'get-entities'

    def __str__(self):
        return 'get-entities'

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        nearest_neighbors = [{
                "text" : row['text'],
                "output_text" : f"{[ne for ne, tag in row['spans']]}"} for row in nearest_neighbors]
        return nearest_neighbors


    def get_prompts_runnable(self, sentence, tags = None):
        nearest_neighbors = self.fst.get_nearest_neighbors(sentence)
        prompt =  self.prompt_template[self.__str__()].format(sentence = sentence,
                                            few_shots = self.get_few_shots(sentence, [], nearest_neighbors),
                                            precisions = self.get_precision())
        return [(prompt, "None")]
    
    
    
    def run_prompt(self, llm : "LLMModel", 
                   sentence : str, 
                   verifier : "Verifier" = None, 
                   confidence_checker : ConfidenceChecker= None,
                   tags = ["PER", "ORG", "LOC", 'MISC']) :
        return super(PT_GetEntities, self).run_prompt(llm, sentence, verifier, confidence_checker, prefix = '[', tags = tags)
    
    
    def process_output(self, response : str, tag : str, tags = None):
        start_index = response.find('[')  # Find the opening curly brace
        end_index = response.rfind(']')    # Find the closing curly brace
        
        if start_index != -1 and end_index != -1:
            response = response[start_index:end_index+1]
        else:
            response ="[]"
    
        try:
            named_entities = ast.literal_eval(response)
        except Exception as e:
            named_entities = []

        return named_entities
        
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        return [[ne for ne, tag in row['spans']] for row in dataset]