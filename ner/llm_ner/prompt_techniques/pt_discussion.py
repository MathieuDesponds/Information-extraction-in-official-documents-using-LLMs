import ast
from ner.Datasets.MyDataset import MyDataset
from ner.llm_ner.few_shots_techniques import FewShotsTechnique

from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompts import *


class PT_OutputList(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique, with_precision = True):
        super().__init__(fst, with_precision = with_precision)
    
    @staticmethod
    def name():
        return 'discussion'
    
    def __str__(self):
        return 'discussion'

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        nearest_neighbors = [{
                "text" : row['text'],
                "output_text" : row['spans']} for row in nearest_neighbors]
        return nearest_neighbors
    
    def get_prompts_runnable(self, sentence):
        nearest_neighbors = self.fst.get_nearest_neighbors(sentence)
        prompt =  prompt_template[self.__str__()].format(sentence = sentence,
                                            few_shots = self.get_few_shots(sentence, [], nearest_neighbors),
                                            precisions = self.get_precision())
        return [(prompt, "None")]
    
    def process_output(self, response : str, tag : str = None):
        start_index = response.find('[[')  # Find the opening curly brace
        end_index = response.rfind(']]')    # Find the closing curly brace
        
        if start_index != -1 and end_index != -1:
            response = response[start_index:end_index+2]
        else:
            print("-----------------------------------------------")
            print(f"response does not contain [[]]. Returned {response}")
            print("-----------------------------------------------")
            response ="[]"
    
        try:
            named_entities = ast.literal_eval(response)
        except Exception as e:
            named_entities = []
        
        named_entities = list(set([(ne, tag ) for ne,tag in named_entities if tag in ["PER", "ORG", "LOC", 'MISC']]))
        return named_entities
    
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        return dataset['spans']