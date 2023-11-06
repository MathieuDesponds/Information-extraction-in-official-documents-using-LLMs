import pickle
from datasets import load_dataset
from ner.Datasets.MyDataset import MyDataset
from ner.Datasets.utils import *

class Conll2003Dataset(MyDataset):
    def __init__(self, dataset = None, split = "test", cleaned = False):
        self.split = split
        self.cleaned = cleaned
        if not dataset : 
            dataset = load_dataset("conll2003", split = split)
            dataset = dataset.remove_columns(["chunk_tags", "pos_tags"])
            dataset = dataset.map(lambda ex : {'text' : " ".join(ex['tokens'])})
            if cleaned : 
                dataset = clean_dataset(dataset)
            dataset = dataset.map(Conll2003Dataset.get_spans)
            dataset = dataset.map(MyDataset.add_llama_ner_tags)
            dataset = dataset.map(MyDataset.add_llama_ner_tags_2)
            dataset = dataset.map(MyDataset.add_sentence_embedding)
            dataset = dataset.map(MyDataset.add_entity_embeddings, with_indices=True)
            self.dataset = dataset
            self.all_entity_embeddings = self.get_all_embeddings()

        else :
            self.dataset = dataset.map(self.adjust_entity_embeddings_idx, with_indices=True)
            self.all_entity_embeddings = self.get_all_embeddings()
            self.dataset = dataset

    @staticmethod
    def get_spans(data_point):
        tag_mapping = {
            0: 'O',
            1: 'B-PER',
            2: 'I-PER',
            3: 'B-ORG',
            4: 'I-ORG',
            5: 'B-LOC',
            6: 'I-LOC',
            7: 'B-MISC',
            8: 'I-MISC'
        }

        named_entities = []  # To store the extracted named entities and tags
        current_entity = None  # To keep track of the current named entity being processed
        current_tag = None  # To keep track of the current NER tag being processed

        for token, ner_tag in zip(data_point['tokens'], data_point['ner_tags']):
            ner_tag = tag_mapping[ner_tag]
            if ner_tag == 'O':
                if current_entity is not None:
                    named_entities.append((current_entity, current_tag))
                    current_entity = None
                    current_tag = None
            else:
                tag_prefix, entity_type = ner_tag.split('-')
                if tag_prefix == 'B':
                    if current_entity is not None:
                        named_entities.append((current_entity, current_tag))
                    current_entity = token
                    current_tag = entity_type
                elif tag_prefix == 'I':
                    if current_entity is not None:
                        current_entity += ' ' + token
                    else:
                        current_entity = token
                    current_tag = entity_type

        # Check if there is a named entity at the end of the sequence
        if current_entity is not None:
            named_entities.append((current_entity, current_tag))

        data_point['spans'] = named_entities
        return data_point

    def train_test_split(self, test_size=0.3, seed = 42):
        splitted_dataset = self.dataset.train_test_split(test_size=test_size, seed =seed)
        return Conll2003Dataset(dataset = splitted_dataset['train']), Conll2003Dataset(dataset = splitted_dataset['test'])

def load_conll_dataset(split = 'test', cleaned = False):
    if cleaned :
        if split =="test":
            with open('./ner/saves/datasets/conll2003_test_cleaned_1588.pkl', 'rb') as f:
                dataset_test =  pickle.load(f)
                return dataset_test
        elif split == 'train':
            with open('./ner/saves/datasets/conll2003_train_cleaned_7577.pkl', 'rb') as f:
                dataset_train =  pickle.load(f)
                return dataset_train
    elif split =="test":
        with open('./ner/saves/datasets/conll2003_test_2762.pkl', 'rb') as f:
            dataset_train =  pickle.load(f)
        with open('./ner/saves/datasets/conll2003_test_50.pkl', 'rb') as f:
            dataset_test =  pickle.load(f)
    else :
        with open('./ner/saves/datasets/conll2003_train_3616.pkl', 'rb') as f:
            dataset_train =  pickle.load(f)
        with open('./ner/saves/datasets/conll2003_train_2000.pkl', 'rb') as f:
            dataset_test =  pickle.load(f)
    return dataset_train, dataset_test

def save_conll_dataset(dataset : Conll2003Dataset):
    with open(f"./ner/saves/datasets/conll2003_{dataset.split}_{'cleaned_' if dataset.cleaned else ''}{len(dataset)}.pkl", 'wb')as f:
        pickle.dump(dataset,f)

def get_test_cleaned_split():
    dataset_test = load_conll_dataset(split = 'test', cleaned = True)
    return dataset_test.train_test_split(test_size = 50)