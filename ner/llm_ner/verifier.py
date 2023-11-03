from ner.Datasets.MyDataset import MyDataset
from ner.llm_ner.few_shots_techniques import FST_Entity, FewShotsTechnique
from ner.llm_ner.prompts import *
from ner.utils import get_embbeding
import random

class Verifier():
    def __init__(self, model : 'LLMModel', data_train : MyDataset) -> None:
        self.model = model
        self.fst = FST_Entity(data_train, nb_few_shots = 5)
        
    def verify(self, sentence, processed_response):
        verified = []
        for ne in processed_response :
            named_entity, tag = ne[0], ne[1]
            nearest_neighbors = self.fst.get_nearest_neighbors(sentence)
            nearest_neighbors = self.process_examples(nearest_neighbors, tag)
            few_shots = verifier_few_shot_prompt(nearest_neighbors).format(sentence = sentence, 
                                                        tag = mapping_abbr_string_verifier[tag], 
                                                        named_entity = named_entity)
            prompt = verifier_prompt_template.format(tag=mapping_abbr_string_verifier[tag], 
                                                     precision = precision_ner[tag],
                                                     few_shots = few_shots)
            print(prompt)
            response = self.model(prompt, stop = ["<end_answer>", '</start_answer>', '<end_output>', '</start_output>'])

            if "yes" in response.lower() :
                verified.append(ne)
            elif 'no' in response.lower() :
                continue
            else :
                print(f"Neither 'No'  nor 'Yes' in the response of the verification for {ne}")
        return verified
    

    def __str__(self) -> str:
        return "verifier-{fst}"

    def process_examples(self, examples, tag):
        output = []
        for row in examples :
            if row['spans'] :
                rand_ent = random.choice(row['spans'])
                output.append({
                    "sentence" : row['text'], 
                    "named_entity" : rand_ent[0], 
                    "tag" : mapping_abbr_string_verifier[tag], 
                    "answer" : 'yes' if tag == rand_ent[1] else 'no'
                })
        return output