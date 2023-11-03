from abc import ABC, abstractmethod      

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

from ner.utils import get_embbeding, sentence_transformer
from ner.Datasets.MyDataset import MyDataset


class FewShotsTechnique(ABC) :
    def __init__(self, training_dataset : MyDataset, nb_few_shots = 5):
        self.training_dataset = training_dataset
        self.nb_few_shots = nb_few_shots
        
    @abstractmethod
    def get_nearest_neighbors(self, sentence : str)-> list[str]:
        pass

    @abstractmethod
    def __str__(self):
        pass

    
class FST_NoShots(FewShotsTechnique):        
    def get_nearest_neighbors(self, sentence : str)-> list[str]:
        return []
    
    def __str__(self):
        return "no-shots"
    
class FST_Random(FewShotsTechnique):    
    def get_nearest_neighbors(self, sentence : str)-> list[str]:
        random_rows = [self.training_dataset[i] 
                       for i in random.sample(range(len(self.training_dataset)), self.nb_few_shots)]
        return random_rows
    
    def __str__(self):
        return "random"
    
class FST_Sentence(FewShotsTechnique):    
    def get_nearest_neighbors(self, sentence : str)-> list[str]:
        sentence_embedding = sentence_transformer.encode(sentence)
        similarities = cosine_similarity([sentence_embedding], self.training_dataset['sentence_embedding'])
        top_k_indices = np.argsort(similarities[0])[-self.nb_few_shots:][::-1]
        nearest_neighbors = [self.training_dataset[int(i)] for i in list(top_k_indices)]
        return nearest_neighbors
    
    def __str__(self):
        return "sentence"
    
class FST_Entity(FewShotsTechnique):
    def __str__(self):
        return "entity"
    
    def get_nearest_neighbors(self, sentence : str)-> list[str]:
        return self.get_similar_sentence_by_entities(
            get_embbeding(sentence), 
            self.training_dataset.all_entity_embeddings)
    
    def get_similar_sentence_by_entities(self, tokens_embedding, other_embeddings):
        nk_best = []
        for token_embed in tokens_embedding :
            similarities = cosine_similarity([token_embed['embedding'].numpy()], [emb['embedding'] for emb in other_embeddings])
            k_best_indices = np.argsort(similarities[0])[-self.nb_few_shots:][::-1]
            k_best = [{
                'embedding' : other_embeddings[index]['embedding'], 
                'token' : other_embeddings[index]['token'], 
                'idx' : other_embeddings[index]['idx'],
                'score' : similarities[0][index]} for index in k_best_indices]
            nk_best.extend(k_best)
        bests = [best['idx'] for best in sorted(nk_best, key=lambda x: x['score'], reverse=True)]
        bests = list(dict.fromkeys(bests))[:self.nb_few_shots]
        nearest_neighbors = [self.training_dataset.dataset[best] for best in bests]
        return nearest_neighbors