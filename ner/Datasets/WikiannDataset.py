from datasets import load_dataset
from ner.Datasets.MyDataset import MyDataset


class WikiannDataset(MyDataset):
    def __init__(self, dataset = None ):
        if not dataset : 
            dataset = load_dataset("wikiann", 'fr', split = 'test').select([1,2])
            dataset = dataset.remove_columns(["langs"])
            dataset = dataset.map(lambda ex : {'text' : " ".join(ex['tokens'])})
            dataset = dataset.map(lambda ex : {'spans' : [(span[5:], span[:3]) for span in ex['spans']]})
            dataset = dataset.map(self.add_llama_ner_tags)
            dataset = dataset.map(self.add_sentence_embedding)
            dataset = dataset.map(self.add_entity_embeddings, with_indices=True)
            self.dataset = dataset
            self.all_entity_embeddings = self.get_all_embeddings()
        else :
            self.dataset = dataset.map(self.add_entity_embeddings, with_indices=True)
            self.all_entity_embeddings = self.get_all_embeddings()
    
    def train_test_split(self, test_size=0.3, seed = 42):
        splitted_dataset = self.dataset.train_test_split(test_size=test_size, seed =seed)
        return WikiannDataset(dataset = splitted_dataset['train']), WikiannDataset(dataset = splitted_dataset['test'])