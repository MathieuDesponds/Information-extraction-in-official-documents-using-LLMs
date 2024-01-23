from myMongoClient import *
import pandas as pd

ALL_PROMPTS_FILE = "data/saved_prompts/prompts_2024-01-23_15:30:03.csv"

def get_all_prompts():
    all_prompts = pd.DataFrame(load(ALL_PROMPTS_FILE))
    all_prompts['fields_name_key'] = all_prompts['fields_name_key'].map(lambda fields_name : [field.replace("Person who the document is in reference to, not the bank nor the mandatary","client") for field in fields_name])
    
    map_doc_id_doc_name = load("map_doc_id_doc_name.pkl")
    map_doc_name_doc_id = {value:key for key, value in map_doc_id_doc_name.items()}

    all_prompts['doc_id'] = all_prompts['file_name'].apply(lambda name : map_doc_name_doc_id[name] if name in map_doc_name_doc_id else None)
    all_prompts = all_prompts[all_prompts['doc_id'] != None]
    return all_prompts

def print_all_missing_mapping():
    all_prompts_label_name = []
    all_prompts = get_all_prompts()
    all_prompts['fields_name_key'].apply(all_prompts_label_name.extend)
    all_prompts_label_name = set(all_prompts_label_name)

    label_name_to_class = {} 
    for key, value in load("classifaction_label_to_label_name.pkl").items() :
        if isinstance(value, list) :
            for val in value :
                label_name_to_class[val] = key
        else : 
            label_name_to_class[value] = key

    for label_name in all_prompts_label_name :
        if label_name not in label_name_to_class and '-' not in label_name:
            print(label_name)