
# for tag in [" 'PER'", " 'ORG'", " 'LOC'", " 'MISC'"]:
#     print(llm.tokenize(tag.encode()))


mapping_tag_tokens = {
    'PER' : [6512],
    'ORG' : [1017,28777],
    'LOC' : [9668],
    'MISC': [28755, 1851, 28743]
}
values_to_find = [l[0] for k, l in mapping_tag_tokens.items()]

def get_logits_for_tags(sentence, llm, llama_ner):
    prompt = llama_ner.get_prompts(sentence=sentence, tags = [], n = 0)
    output = llm(prompt, stop = ["<end_output>"])['choices'][0]['text']
    idx_tags = [i for i, val in enumerate(llm.eval_tokens) if val in values_to_find]
    logits = llm.eval_logits
    logits_for_tags = [{tag: logits[idx-1][mapping_tag_tokens[tag][0]] for tag in mapping_tag_tokens} for idx in idx_tags]
    return logits_for_tags, output