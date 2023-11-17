from ner.Datasets.MyDataset import MyDataset
from ner.llm_ner.few_shots_techniques import FewShotsTechnique

from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompts import *

import re

class PT_Wrapper(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique, with_precision = True, prompt_template : dict[PromptTemplate] = prompt_template, plus_plus = False ):
        super().__init__(fst, with_precision, prompt_template, plus_plus)
    
    @staticmethod
    def name():
        return 'wrapper'
    
    def __str__(self):
        return 'wrapper'

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        nearest_neighbors = [{
                "text" : row['text'],
                "output_text" : row['llama_text_2']} for row in nearest_neighbors]
        return nearest_neighbors
    
    def get_prompts_runnable(self, sentence, tags = None):
        nearest_neighbors = self.fst.get_nearest_neighbors(sentence)
        prompt =  self.prompt_template[self.__str__()].format(sentence = sentence,
                                            few_shots = self.get_few_shots(sentence, [], nearest_neighbors),
                                            precisions = self.get_precision())
        return [(prompt, "None")]
    
    def process_output(self, response : str, tag : str = None):
        pattern = r'<([^>]+)>([^<]+)</\1>'

        # Find all matches of the pattern in the text
        matches = re.findall(pattern, response)

        # Create a list of tuples with the extracted named entities and tags
        named_entities = [(entity, mapping_string_abbr[tag]) for tag, entity in matches if tag in mapping_string_abbr]
        return named_entities

    
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        return dataset['llama_text_2']
        