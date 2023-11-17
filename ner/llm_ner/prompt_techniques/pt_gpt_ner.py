from ner.Datasets.MyDataset import MyDataset
from ner.llm_ner.few_shots_techniques import FewShotsTechnique

from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.Datasets.OntoNotes5Dataset import ONTONOTE5_TAGS_PRECISION
from ner.llm_ner.prompts import *

import re

class PT_GPT_NER(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique, with_precision = True, prompt_template : dict[PromptTemplate] = prompt_template, plus_plus = False ):
        super().__init__(fst, with_precision, prompt_template, plus_plus)

    @staticmethod
    def name():
        return '@@##'

    def __str__(self):
        return '@@##'

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
            prompt = self.prompt_template[self.__str__()].format(tag=mapping_abbr_string_ner[tag], 
                                        precision = precision_ner[tag]if len(tags) == 4 else ONTONOTE5_TAGS_PRECISION[tag],
                                        few_shots = few_shots,
                                        sentence = sentence)
            prompts.append(prompt)
        return list(zip(prompts,tags))
    
    def process_output(self, response : str, tag : str):
        pattern = r'@@\s*(.*?)##'
        named_entities = re.findall(pattern, response, re.DOTALL)
        return [(ne, tag) for ne in named_entities]
        
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        return [row['llama_text'][tag] for row in dataset]