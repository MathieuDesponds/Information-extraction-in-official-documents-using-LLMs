from matplotlib import pyplot as plt
from llm.LLMModel import *
from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique
from ner.llm_ner.prompt_techniques.pt_multi_pt import PT_2Time_Tagger
from ner.llm_ner.prompt_techniques.pt_tagger import LETTER_TO_TAG_MAPPING
from ner.llm_ner.few_shots_techniques import *
from ner.Datasets.Conll2003Dataset import load_conll_dataset
from ner.utils import dump, load
from tqdm import tqdm
import numpy as np


mapping_tag_tokens = {
    'PER' : [6512],
    'ORG' : [1017,28777],
    'LOC' : [9668],
    'MISC': [28755, 1851, 28743]
}

mapping_letter_tokens = {
    'P': [ 464, 28753],
    'O': [ 464, 28762],
    'L': [ 464, 28758],
    'M': [ 464, 28755],
    'N': [ 464, 28759]
}

values_to_find = lambda mapping : [l[0] for k, l in mapping.items()]

def index_of_values_to_find(tokens):
    tokens = list(tokens)
    index = []
    values = []
    window_width = len(mapping_letter_tokens['P'])
    for i in range(len(tokens)-(window_width-1)) :
        for key, val in mapping_letter_tokens.items() :  
            if tokens[i:i+window_width] == val :
                values.append(key)
                index.append(i)
    return index, values



def _old_get_logits_for_tags(sentence : str, llm: LLMModel, pt : PromptTechnique):
    prompt = pt.get_prompts_runnable(sentence)[0][0]
    output = llm(prompt)
    generated_tokens = list(llm.model.eval_tokens)[-output['usage']['completion_tokens']:]
    generated_logits = list(llm.model.eval_logits)[-output['usage']['completion_tokens']:]
    idx_tags = [i+1 for i, val in enumerate(generated_tokens) if val in values_to_find(mapping_tag_tokens)]
    logits_for_tags = [{tag: generated_logits[idx][mapping_tag_tokens[tag][0]] for tag in mapping_tag_tokens} for idx in idx_tags]
    return logits_for_tags, output

def get_logits_for_tags(data_point, model: LLMModel, pt : PromptTechnique):
    entities, output = model.invoke(data_point['text'], pt, None)
    gold_tags = dict(data_point['spans'])
    generated_tokens = list(model.model.model.eval_tokens)[-output['usage']['completion_tokens']:]
    generated_logits = list(model.model.model.eval_logits)[-output['usage']['completion_tokens']:]
    index, values = index_of_values_to_find(generated_tokens)
    logits_for_tags = []
    print(gold_tags)
    print([entity[0] for entity in entities])
    for entity, idx, tag_found in zip(entities, index, values) :
            logits_for_tags.append({
                'entity' : entity,
                'gold tag' : gold_tags[entity[0]] if entity[0] in gold_tags else 'None',
                'outputted_tag' : LETTER_TO_TAG_MAPPING[tag_found],
                'tags_logits': {LETTER_TO_TAG_MAPPING[tag] : generated_logits[idx][mapping_letter_tokens[tag][1]] for tag in mapping_letter_tokens.keys()},
                'confidence' : {tag : confidence for tag, confidence in zip(logits_for_tags['tags_logits'].keys(), softmax(logits_for_tags['tags_logits'].values()))}
                })
    return {
            'data_point_idx' : data_point['id'],
            'spans' : data_point['spans'],
            'logits_for_tags' : logits_for_tags
         }, output, index, values

def generate_data_for_confidence():
    data_train = load_conll_dataset(split = 'train', cleaned = True)
    data_test =  load_conll_dataset(split = 'test',  cleaned = True)
    data_test.select(range(100))
    model = MistralAI(llm_loader = Llama_LlamaCpp())
    fst = FST_Sentence(data_train, 3)
    multi_pt = PT_2Time_Tagger(fst)
    all_data = []
    for data_point in tqdm(data_test) :
        logits_for_tags, output, index, values = get_logits_for_tags(data_point, model, pt = multi_pt)
        all_data.append(logits_for_tags)
    
    dump(all_data, './ner/saves/confidence/data_for_confidence.pkl')

def load_generated_data_for_confidence():
    return load('./ner/saves/confidence/data_for_confidence.pkl')

def softmax(logits):
    exp_logits = np.exp(logits)  # Subtracting the maximum value for numerical stability
    return exp_logits / np.sum(exp_logits)  # Assuming logits is a 2D array

confidence_functions = {
    "softmax_direct" : lambda logits : softmax(logits),
    "softmax_min" : lambda logits : softmax(logits - np.min(logits)),
    "softmax_max" : lambda logits : softmax(logits - np.max(logits)),
    "proba_direct" : lambda logits : [log/np.sum(logits) for log in logits],
    "proba_centered" : lambda logits : [log-np.min(logits)/np.sum(logits-np.min(logits)) for log in logits],
    "transparent" : lambda logits: logits
}
def add_confidence_to_results(logits_for_tags, confidence_fct = lambda l : softmax(l)):
    logits_for_tags['confidence'] = {tag : confidence for tag, confidence in zip(logits_for_tags['tags_logits'].keys(), confidence_fct(list(logits_for_tags['tags_logits'].values())))}
    

    return logits_for_tags

def show_confidence(all_data = None):
    if not all_data :
        all_data = load_generated_data_for_confidence()
    
    for key, func in confidence_functions.items() :
        for dp in all_data:
            for ent in dp['logits_for_tags'] :
                ent = add_confidence_to_results(ent, func)
        points = [] # (right/false, confidence)
        for data_point in all_data:
            for entity_point in data_point['logits_for_tags'] :
                if entity_point['gold tag'] != 'None':
                    points.append(
                        (entity_point['gold tag'] == entity_point['outputted_tag'],
                        entity_point['confidence'][entity_point['outputted_tag']])
                    )
        true_values =  [pair[1] for pair in points if pair[0]]
        false_values = [pair[1] for pair in points if not pair[0]]

        # Create a figure and two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

        # Plotting histograms for True and False values
        ax1.hist(true_values, color='blue', edgecolor='black')
        ax2.hist(false_values, color='red', edgecolor='black')

        # Set titles and labels for each subplot
        ax1.set_title(f'Histogram of True Values with {key}')
        ax1.set_ylabel('Frequency')
        ax2.set_title(f'Histogram of False Values with {key}')
        ax2.set_ylabel('Frequency')

        # Get the maximum count between the two histograms
        max_count = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        print(max_count)
        # Set the same y-axis limits for both subplots
        ax1.set_ylim(0, max_count)
        ax2.set_ylim(0, max_count)

        plt.tight_layout()
        plt.show()