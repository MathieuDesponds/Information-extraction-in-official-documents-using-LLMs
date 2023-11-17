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
class WikiNeuralDataset(MyDataset):
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
    def name():
        return 'wikineural'
    
    def train_test_split(self, test_size=0.3, seed = 42):
        splitted_dataset = self.dataset.train_test_split(test_size=test_size, seed =seed)
        return WikiNeuralDataset(dataset = splitted_dataset['train']), WikiNeuralDataset(dataset = splitted_dataset['test'])


def get_test_cleaned_split(seed = None):
    dataset_test : MyDataset = MyDataset.my_load_dataset(WikiNeuralDataset, split = 'test', cleaned = True)
    if not seed :
        seed = random.randint(0, 1535468)
    return dataset_test.train_test_split(test_size = 50, seed = seed)