from llm.LLMModel import *
from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompt_techniques.pt_discussion import PT_OutputList
from ner.llm_ner.prompt_techniques.pt_gpt_ner import PT_GPT_NER
from ner.llm_ner.prompt_techniques.pt_wrapper import PT_Wrapper
from ner.llm_ner.few_shots_techniques import *


mapping_tag_tokens = {
    'PER' : [6512],
    'ORG' : [1017,28777],
    'LOC' : [9668],
    'MISC': [28755, 1851, 28743]
}

mapping_letter_tokens = {
    'P': [464, 28753, 647],
    'O': [464, 28762, 647],
    'L': [464, 28758, 647],
    'M': [464, 28755, 647],
    'N': [464, 28759, 647]
}

values_to_find = lambda mapping : [l[0] for k, l in mapping.items()]

def index_of_values_to_find(tokens):
    tokens = list(tokens)
    index = []
    values = []
    for i in range(len(tokens)-2) :
        for key, val in mapping_letter_tokens.items() :  
            if tokens[i:i+3] == val :
                values.append(key)
                index.append(i)
    return index, values



def _old_get_logits_for_tags(sentence : str, llm: LLMModel, pt : PromptTechnique):
    prompt = pt.get_prompts_runnable(sentence)[0][0]
    output = llm(prompt, stop = ["<end_output>"])['choices'][0]['text']
    generated_tokens = list(llm.model.eval_tokens)[-output['usage']['completion_tokens']:]
    generated_logits = list(llm.model.eval_logits)[-output['usage']['completion_tokens']:]
    idx_tags = [i for i, val in enumerate(generated_tokens) if val in values_to_find(mapping_tag_tokens)]
    logits_for_tags = [{tag: generated_logits[idx][mapping_tag_tokens[tag][0]] for tag in mapping_tag_tokens} for idx in idx_tags]
    return logits_for_tags, output

def get_logits_for_tags(sentence : str, model: LLMModel, pt : PromptTechnique):
    entities, output = model.invoke(sentence, pt, None)
    generated_tokens = list(model.model.eval_tokens)[-output['usage']['completion_tokens']:]
    generated_logits = list(model.model.eval_logits)[-output['usage']['completion_tokens']:]
    
    index, values = index_of_values_to_find(generated_tokens)
    logits_for_tags = [{'gold_tag' : gold_tag, 'tags_logits': {tag: generated_logits[idx][mapping_letter_tokens[tag][1]] for tag in mapping_letter_tokens.keys()}} for idx, gold_tag in zip(index,values)]
    return logits_for_tags, output, index, values