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
values_to_find = [l[0] for k, l in mapping_tag_tokens.items()]

def get_logits_for_tags(sentence : str, llm: LLMModel, pt : PromptTechnique):
    prompt = pt.get_prompts_runnable(sentence)[0][0]
    output = llm(prompt, stop = ["<end_output>"])#['choices'][0]['text']
    generated_tokens = list(llm.model.eval_tokens)[-output['usage']['completion_tokens']:]
    generated_logits = list(llm.model.eval_logits)[-output['usage']['completion_tokens']:]
    idx_tags = [i for i, val in enumerate(generated_tokens) if val in values_to_find]
    logits_for_tags = [{tag: generated_logits[idx][mapping_tag_tokens[tag][0]] for tag in mapping_tag_tokens} for idx in idx_tags]
    return logits_for_tags, output