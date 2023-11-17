import pickle
import random
from datasets import load_dataset
from ner.Datasets.MyDataset import MyDataset
from ner.Datasets.utils import *
from ner.utils import load, dump

class OntoNote5Dataset(MyDataset):
    def __init__(self, dataset = None, split = "test"):
        self.split = split
        if not dataset : 
            dataset = load_dataset("tner/ontonotes5", split = split)
            dataset = dataset.rename_column("tags", "ner_tags")
            dataset.select(range(50))
            dataset = dataset.map(lambda ex : {'text' : " ".join(ex['tokens'])})

            dataset = dataset.map(lambda row : MyDataset.get_spans(row, ONTONOTE5_MAPPING_TAG))
            # dataset = dataset.map(MyDataset.add_llama_ner_tags)
            # dataset = dataset.map(MyDataset.add_llama_ner_tags_2)
            # dataset = dataset.map(MyDataset.add_sentence_embedding)
            # dataset = dataset.map(MyDataset.add_entity_embeddings, with_indices=True)
            self.dataset = dataset
            # self.all_entity_embeddings = self.get_all_embeddings()

        else :
            self.dataset = dataset.map(self.adjust_entity_embeddings_idx, with_indices=True)
            self.all_entity_embeddings = self.get_all_embeddings()
            self.dataset = dataset

    @staticmethod
    def name():
        return 'ontonote5'
    

    def train_test_split(self, test_size=0.3, seed = 42):
        splitted_dataset = self.dataset.train_test_split(test_size=test_size, seed =seed)
        return OntoNote5Dataset(dataset = splitted_dataset['train']), OntoNote5Dataset(dataset = splitted_dataset['test'])


def get_test_cleaned_split(seed = None):
    dataset_test : OntoNote5Dataset = MyDataset.my_load_dataset(OntoNote5Dataset, split = 'test')
    if not seed :
        seed = random.randint(0, 1535468)
    return dataset_test.train_test_split(test_size = 50, seed = seed)

ONTONOTE5_MAPPING_TAG = {0: 'O',
 1: 'B-CARDINAL',
 2: 'B-DATE',
 3: 'I-DATE',
 4: 'B-PERSON',
 5: 'I-PERSON',
 6: 'B-NORP',
 7: 'B-GPE',
 8: 'I-GPE',
 9: 'B-LAW',
 10: 'I-LAW',
 11: 'B-ORG',
 12: 'I-ORG',
 13: 'B-PERCENT',
 14: 'I-PERCENT',
 15: 'B-ORDINAL',
 16: 'B-MONEY',
 17: 'I-MONEY',
 18: 'B-WORK_OF_ART',
 19: 'I-WORK_OF_ART',
 20: 'B-FAC',
 21: 'B-TIME',
 22: 'I-CARDINAL',
 23: 'B-LOC',
 24: 'B-QUANTITY',
 25: 'I-QUANTITY',
 26: 'I-NORP',
 27: 'I-LOC',
 28: 'B-PRODUCT',
 29: 'I-TIME',
 30: 'B-EVENT',
 31: 'I-EVENT',
 32: 'I-FAC',
 33: 'B-LANGUAGE',
 34: 'I-PRODUCT',
 35: 'I-ORDINAL',
 36: 'I-LANGUAGE'}
ONTONOTE5_MAPPING_TAG_REVERSE = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12, 
    "B-PERCENT": 13,
    "I-PERCENT": 14, 
    "B-ORDINAL": 15, 
    "B-MONEY": 16, 
    "I-MONEY": 17, 
    "B-WORK_OF_ART": 18, 
    "I-WORK_OF_ART": 19, 
    "B-FAC": 20, 
    "B-TIME": 21, 
    "I-CARDINAL": 22, 
    "B-LOC": 23, 
    "B-QUANTITY": 24, 
    "I-QUANTITY": 25, 
    "I-NORP": 26, 
    "I-LOC": 27, 
    "B-PRODUCT": 28, 
    "I-TIME": 29, 
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36
}