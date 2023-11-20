import ast
from ner.Datasets.MyDataset import MyDataset
from ner.Datasets.OntoNotes5Dataset import OntoNote5Dataset, ONTONOTE5_TAGS
from ner.llm_ner.confidence_checker import ConfidenceChecker
from ner.llm_ner.few_shots_techniques import FewShotsTechnique

from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompts import *


class PT_Filing(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique, with_precision = True, prompt_template : dict[PromptTemplate] = prompt_template, plus_plus = False, tags = ONTONOTE5_TAGS ):
        super().__init__(fst, with_precision, prompt_template, plus_plus)
        self.tags = tags
    
    @staticmethod
    def name():
        return 'filing'
    
    def __str__(self):
        return 'filing'

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        output = lambda spans : '{{' + ', '.join([
            f"{tag} : {[', '.join([ne for ne,t in spans if t == tag])] if [ne for ne,t in spans if t == tag] else []}"
            for tag in self.tags[::-1]
        ]) + '}}'
        nearest_neighbors = [{
                "text" : row['text'],
                "output_text" : output(row['spans'])} for row in nearest_neighbors]
        print(type(nearest_neighbors[0]['output_text']))
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
                   tags = ONTONOTE5_TAGS) :
        return super(PT_Filing, self).run_prompt(llm, sentence, verifier, confidence_checker, prefix = '[',tags =  tags)
    
    def process_output(self, response : str, tag : str = None, tags = ONTONOTE5_TAGS):
        start_index = response.find('{')  # Find the opening curly brace
        end_index = response.rfind('}')    # Find the closing curly brace
        
        if start_index != -1 and end_index != -1:
            response = response[start_index:end_index+2]
        else:
            print("-----------------------------------------------")
            print(f"response does not contain {{}}. Returned {response}")
            print("-----------------------------------------------")
            response ="{{}}"
    
        try:
            named_entities = ast.literal_eval(response)
        except Exception as e:
            named_entities = {}
        
        out = []
        for tag, nes in named_entities.items():
            if tag in tags :
                out.extend([(ne,tag) for ne in nes])

        named_entities = list(set(out))
        return named_entities
    
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        output = lambda spans : {
            f"{tag} : [{', '.join([ne for ne,t in spans if t == tag])}]\n"
            for tag in self.tags
        }
        return [output(row['spans']) for row in dataset] 