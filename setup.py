from ner.Datasets.OntoNotes5Dataset import OntoNote5Dataset
from ner.Datasets.MyDataset import save_dataset

dataset_test = OntoNote5Dataset(split = 'test')
save_dataset(dataset_test)
dataset_train = OntoNote5Dataset(split = 'train') 
save_dataset(dataset_train)