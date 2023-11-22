import pickle
import random
from datasets import load_dataset
from ner.Datasets.MyDataset import MyDataset
from ner.Datasets.utils import *
CONLL2003_TAG_MAPPING = {
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
CONLL2003_TAGS =  ['PER', 'ORG', 'LOC', 'MISC']


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
            dataset = dataset.map(lambda row : MyDataset.get_spans(row, CONLL2003_TAG_MAPPING))
            dataset = dataset.map(lambda row : MyDataset.add_llama_ner_tags(row, self.get_tags(), CONLL2003_TAG_MAPPING))
            dataset = dataset.map(lambda row : MyDataset.add_llama_ner_tags_2(row, self.get_tags()))
            dataset = dataset.map(MyDataset.add_sentence_embedding)
            dataset = dataset.map(MyDataset.add_entity_embeddings, with_indices=True)
            self.dataset = dataset
            self.all_entity_embeddings = self.get_all_embeddings()

        else :
            self.dataset = dataset.map(self.adjust_entity_embeddings_idx, with_indices=True)
            self.all_entity_embeddings = self.get_all_embeddings()
            self.dataset = dataset


    def get_tags(self):
        return ['PER', 'ORG', 'LOC', 'MISC']

    def get_tags_mapping(self):
        return CONLL2003_TAG_MAPPING
    
    @staticmethod
    def name():
        return 'conll2003'
    
    def train_test_split(self, test_size=0.3, seed = 42):
        splitted_dataset = self.dataset.train_test_split(test_size=test_size, seed =seed)
        return Conll2003Dataset(dataset = splitted_dataset['train']), Conll2003Dataset(dataset = splitted_dataset['test'])


def get_test_cleaned_split(seed = None):
    dataset_test : Conll2003Dataset = MyDataset.my_load_dataset(Conll2003Dataset, split = 'test', cleaned = True, length = 1588)
    if not seed :
        seed = random.randint(0, 1535468)
    return dataset_test.train_test_split(test_size = 50, seed = seed)