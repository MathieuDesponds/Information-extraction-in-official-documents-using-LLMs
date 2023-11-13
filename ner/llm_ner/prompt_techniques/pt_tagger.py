import ast
from ner.Datasets.MyDataset import MyDataset
from ner.llm_ner.few_shots_techniques import FewShotsTechnique

from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompts import *

import re

LETTER_TO_TAG_MAPPING = {"P" : "PER", "O": "ORG", "L" : "LOC", "M" : "MISC"}

class PT_Tagger(PromptTechnique):
    def __init__(self, fst : FewShotsTechnique, with_precision = True):
        super().__init__(fst, with_precision = with_precision)

    @staticmethod
    def name():
        return 'tagger'

    def __str__(self):
        return 'tagger'

    def process_nearest_neighbors(self, nearest_neighbors :list, tag):
        nearest_neighbors_out = []
        for row in nearest_neighbors :
            nes = [ne for ne, tag in row['spans']]
            tags = [tag for ne, tag in row['spans']]
            output_json = '{{\n' + '\n   '.join([
                f"'{ne}' : '{tags[i][0]}',"  #do not remove the comma !!!! It is used in the evaluation of confidence
                for i, ne in enumerate(nes)
            ])+'}}'
            nearest_neighbors_out.append({
                "text" : f"{nes} in '{row['text']}'",
                "output_text" : output_json})
            # print(output_json)
        return nearest_neighbors_out
    
    # ToDo 
    def run_prompt(self, llm : "LLMModel", sentence : str, verifier : "Verifier") :
        all_entities = []
        prompts = self.get_prompts_runnable(sentence)
        for prompt,tag in prompts :
            # print(prompt)
            if llm.check_nb_tokens :
                doc = llm.nlp(prompt)   
                num_tokens = len(doc)
                # print(num_tokens, prompt)
                if num_tokens > 4096 - llm.max_tokens :
                    print("prompt is too big") 
                    continue
            reponse_text, response_all = llm(prompt, with_full_message = True)
            response = '{'+ reponse_text
            # print(f"Response of llm in tagger : {response}")
            processed_response = self.process_output(response, tag)
            if verifier : 
                processed_response = verifier.verify(sentence, processed_response)
            all_entities.extend(processed_response)
        # return all_entities, response_all
        return processed_response, response_all

    def get_prompts_runnable(self, sentence):
        # sentence is in fact "{previous_output} in '{sentence}'"
        entities_sentence = sentence
        real_sentence = sentence.split("'")[-2]
        nearest_neighbors = self.fst.get_nearest_neighbors(real_sentence)
        prompt =  prompt_template[self.__str__()].format(entities_sentence = entities_sentence,
                                            few_shots = self.get_few_shots(real_sentence, [], nearest_neighbors),
                                            precisions = self.get_precision())
        return [(prompt, "None")]
    
    def process_output(self, response : str, tag : str):
        start_index = response.find('{')  # Find the opening curly brace
        end_index = response.rfind('}')    # Find the closing curly brace
        
        if start_index != -1 and end_index != -1:
            response = response[start_index:end_index+1]
        else:
            response ="{}"
    
        print(f"response of tagger : {response}")
        try:
            named_entities = ast.literal_eval(response)
        except Exception as e:
            named_entities = {}

        return [(ne,LETTER_TO_TAG_MAPPING[tag]) for ne, tag in named_entities.items() if tag in LETTER_TO_TAG_MAPPING]
        
    def get_gold(self, dataset : MyDataset, tag : str) -> list[str]:
        return [
                '{\n' + '\n   '.join([
                    f"{ne} : ONEOF['P', 'O', 'L', 'M', 'N']" 
                    for ne, tag in row['spans']
                ])+'\n}'
            for row in dataset]