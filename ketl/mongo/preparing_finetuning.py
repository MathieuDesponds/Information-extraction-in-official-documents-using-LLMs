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

    all_prompts['label'] = all_prompts.apply(lambda row : 
                                             row['fields_name_key'][0].split('-')[0] 
                                             if row['fields_name_key'] and "-" in row['fields_name_key'][0] 
                                             else row['label'], axis = 1)
    all_prompts['fields_name'] = all_prompts['fields_name_key']

    label_name_to_class = load("label_name_to_classifaction_label.pkl")
    all_prompts['fields_key'] =  all_prompts['fields_name'].apply(lambda names : [label_name_to_class[name] if name in label_name_to_class else name for name in names]) 
    all_prompts = all_prompts.drop("fields_name_key", axis =1)
    mongo_client = MyMongoClientLocal()
    all_prompts['fields_key_value'] = all_prompts.apply(lambda row : get_values_from_keys(row), axis = 1)
    # all_prompts['output'] = all_prompts.apply(lambda row : 
    #             '{\n' + ',\n'.join([f"{row['fields_key'][i]} : {row['fields_value'][i]}" for i in range(len(row['fields_key']))]) +'\n}', axis = 1)
    return all_prompts

mongo_client = MyMongoClientLocal()
def get_values_from_keys(row):
    gold_labels = mongo_client.get_labels_from_doc_hash(row['doc_id'])[0]
    if not gold_labels :
        return []
    possible_keys = [key for key in row['fields_key'] if key in gold_labels]
    return {key : gold_labels[key] for key in possible_keys}



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

# def from_label_name_to_label_key():
