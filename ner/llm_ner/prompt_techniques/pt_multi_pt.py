from ner.Datasets.MyDataset import MyDataset
from ner.llm_ner.few_shots_techniques import FewShotsTechnique

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
        return 'multi_prompt-'+'-'.join([pt.__str__ for pt in self.pts])
    

    def run_prompt(self, llm : "LLMModel", sentence : str, verifier : "Verifier") :
        
        output = self.pts[0].run_prompt(llm, sentence, None)
        # To Remove 
        output = f"['Italy', 'Marcello Cuttitta']"
        for pt in self.pts[1:]:
            output = pt.run_prompt(llm, f"{output} in '{sentence}'", None)
        return output

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        raise TypeError("This function in multiprompt should not be used")


    def get_prompts_runnable(self, sentence, tags = ["PER", "ORG", "LOC", 'MISC']):
        raise TypeError("This function in multiprompt should not be used")
    
    def process_output(self, response : str, tag : str):
        raise TypeError("This function in multiprompt should not be used")
        
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        raise TypeError("This function in multiprompt should not be used")
