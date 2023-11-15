import ast
from ner.llm_ner.prompts import *

class ConfidenceChecker() :
    def check(self, sentence, spans, model):
        confidences = []
        prompt = confidence_prompt_template.format(precision = self.get_precision(),
                                                     entities_sentence = f"""In the sentence "{sentence}" the extracted entities were "{spans}" """)
        
        model_response = model(prompt, stop = ["<end_answer>", '</start_answer>', '<end_output>', '</start_output>'])

        processed_model_response = self.process_output('{'+model_response, spans)
        return processed_model_response
    
    def get_precision(self):
        return """### ASSISTANT : Can you give me clarification on the different type of entities ? 
### USER : Yes. """+'\n'.join([val for key, val in precision_ner.items()])+'\n'


    def process_output(self, response : str, spans : list):
        start_index = response.find('{')  # Find the opening curly brace
        end_index = response.rfind('}')    # Find the closing curly brace
        
        if start_index != -1 and end_index != -1:
            response = response[start_index:end_index+1]
        else:
            response ="{}"

        try:
            confidences = ast.literal_eval(response)
        except Exception as e:
            confidences = {}

        ne_with_conf = [(ne[0], ne[1], confidences[ne[0]]) if ne[0] in confidences else 'None' for ne in spans]

        return ne_with_conf