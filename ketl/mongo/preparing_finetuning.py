from myMongoClient import *
import pandas as pd

from datetime import datetime
from transformers import AutoTokenizer

ALL_PROMPTS_FILE = "data/saved_prompts/prompts_2024-01-23_15:30:03.csv"

def get_all_prompts():
    all_prompts = pd.DataFrame(load(ALL_PROMPTS_FILE))
    all_prompts['fields_name_key'] = all_prompts['fields_name_key'].map(lambda fields_name : [field.replace("Person who the document is in reference to, not the bank nor the mandatary","client") for field in fields_name])
    
    map_doc_id_doc_name = load("data/map_doc_id_doc_name.pkl")
    map_doc_name_doc_id = {value:key for key, value in map_doc_id_doc_name.items()}

    all_prompts['doc_id'] = all_prompts['file_name'].apply(lambda name : map_doc_name_doc_id[name] if name in map_doc_name_doc_id else None)
    all_prompts = all_prompts[all_prompts['doc_id'] != None]

    all_prompts['label'] = all_prompts.apply(lambda row : 
                                             row['fields_name_key'][0].split('-')[0] 
                                             if row['fields_name_key'] and "-" in row['fields_name_key'][0] 
                                             else row['label'], axis = 1)
    all_prompts['fields_name'] = all_prompts['fields_name_key']

    label_name_to_class = load("data/label_name_to_classifaction_label.pkl")
    all_prompts['fields_key'] =  all_prompts['fields_name'].apply(lambda names : [label_name_to_class[name] if name in label_name_to_class else name for name in names]) 
    all_prompts = all_prompts.drop("fields_name_key", axis =1)
    all_prompts['fields_key_value'] = all_prompts.apply(lambda row : get_values_from_keys(row), axis = 1)
    all_prompts['text'] = all_prompts.apply(lambda row : get_text_for_ft(row), axis = 1)
    return all_prompts

def get_dataset_for_ft(with_save = False):
    all_prompts = get_all_prompts()
    all_prompts_text = all_prompts.groupby('doc_id')['text'].agg(list).to_dict()
    for doc, values in all_prompts_text.items():
        all_prompts_text[doc] = [{'text' : prompt } for prompt in values]
        
    if with_save :
        dump(all_prompts_text, f"data/labels_results/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_mistral-instruct_ft-data.pkl")
    return all_prompts_text
    
mongo_client = MyMongoClientLocal()
tokenizer =  AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
def get_text_for_ft(row):
    row['messages'].append({
        'role' : "assistant",
        'content' : str(row['fields_key_value'])
    })
    return tokenizer.apply_chat_template(row['messages'], tokenize=False)


def get_values_from_keys(row):
    gold_labels = mongo_client.get_labels_from_doc_hash(row['doc_id'], only_gold=True)
    if not gold_labels :
        return []
    possible_keys = [(key,name) for key,name in zip(row['fields_key'],row['fields_name']) if key in gold_labels]
    return {(name if not '-' in name else '-'.join(name.split('-')[1:]).capitalize())  : gold_labels[key] for key,name in possible_keys}



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
user_llm = {
   "2vIzdtbIlRfELCEIj+Wsxw==" : '7wKQEgxKSZEspILT7n7UqA==',
   "4qvfu2lB0Qe9e6j0Kq+Q0Q==" : "3+h1BtgoAL0PhXWyK1+40g==",
   "HbCTvF8+qSdmSZx2554y1A==" : "VO580j3/ILBAvlhfKY7hFQ==",
   "0Dvjg/odKW5noqUL8wrOKw==" : "mHK8QQTTOTp0sx+85PF1qw==",
   "HwklrgHthb7XGEXVDgu6xg==" : "yEpfWSZXTPQZ84K+aj1drQ==",
   "HbCTvF8+qSdmSZx2554y1A==" : "VO580j3/ILBAvlhfKY7hFQ==",
   "hcqBfAlCslQ0a1QFu1RcJA==" : "63KSW0tjZ6CrwLEQoCfuUg==",
   "fTXgJLNmmgWMv9xzQjnRpQ==" : "29yradmnIewK1RIRbRSTrg==",
}
def map_doc_id_doc_name() :
    all_docs_labels = pd.read_csv("data/labels_results/2024-01-12_16:42:23_df_gold_labels.csv")
    map_doc_id_doc_name = {}
    all_docs = mongo_client.get_all_documents()
    for doc_id in list(all_docs_labels['doc_id'].unique()):
        if doc_id in user_llm :
            doc_id = user_llm[doc_id] 
        map_doc_id_doc_name[doc_id] = [doc['files'][0]['fileName'] for doc in all_docs if doc['_id'] == doc_id and doc['files']][0]
    dump(map_doc_id_doc_name, "map_doc_id_doc_name.pkl")

def update_mapping_class_label_label_name(label_class, label_name):
    file = load("classifaction_label_to_label_name.pkl")
    file[label_class] = label_name
    dump(file, "classifaction_label_to_label_name.pkl") 


def map_doc_type_from_file_name():
    label_versions = pd.read_csv("data/labels_results/2024-01-18_16:15:33_df_gold_labels.csv")
    output = label_versions[label_versions['label_name'] == 'document type'][['doc_id', 'label_value']]
    doc_hash_value = {rows['doc_id'] : rows['label_value'] for idx , rows in output.iterrows()}
    file_name_doc_hash = {inst['files'][0]['fileName'] : inst['_id'] for inst in mongo_client.get_all_documents() if inst['files'] and inst['files'][0]['fileName'].startswith(('Pseudo_',"pseudo_"))}
    file_nam_value = {key : doc_hash_value[value] for key, value in file_name_doc_hash.items() if value in doc_hash_value}
    # file_nam_value, len(file_nam_value)

    dump(file_nam_value, "doc_type_from_file_name.pkl")