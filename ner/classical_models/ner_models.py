import spacy
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pickle
from collections import defaultdict
from tqdm import tqdm
from ner.process_results import show_cm_multi, get_metrics_all
from ner.utils import load, dump

language = 'en'

# refined_mapping = {"PERSON" : "PER","PER" : "PER", "GPE" : "LOC", "ORG": "ORG", "EVENT":'MISC', "WORK_OF_ART":'MISC',"FAC" : 'MISC', "LANGUAGE":'MISC'}
# refined_mapping = defaultdict(lambda : "None", refined_mapping)

# spacy_mapping = {"PERSON" : "PER", "GPE" : "LOC", "LOC" : "LOC", "ORG": "ORG", "EVENT":'MISC', "NORP" : 'MISC', "LANGUAGE":'MISC'}
# spacy_mapping = defaultdict(lambda : "None", spacy_mapping)

nlp_spacy = spacy.load("en_core_web_sm" if language == 'en' else "fr_core_news_sm")

# Function to extract entities from a given text using SpaCy
def extract_entities_spacy(text):
    doc = nlp_spacy(text)
    # entities = [(ent.text, spacy_mapping[ent.label_], ent.label_) for ent in doc.ents]
    entities = [(ent.text, ent.label_, ent.label_) for ent in doc.ents]
    return entities


# Load Flair's NER model for French
flair.device = "cpu"  # Change to "gpu" if you have a GPU
tagger_flair_ontonote = SequenceTagger.load("flair/ner-english-ontonotes")
tagger_flair_ontonote_large = SequenceTagger.load("flair/ner-english-ontonotes-large")

# Load Flair's NER model for multi
# tagger_flair_multi = SequenceTagger.load("flair/ner-multi-fast")

# Function to extract entities from a given text using Flair
def extract_entities_flair(text, tagger):
    sentence = Sentence(text)
    tagger.predict(sentence)
    entities = [(entity.text, entity.get_labels()[0].value) for entity in sentence.get_spans('ner')]
    return entities


# tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
# model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
# nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
# def extract_entities_babelspace(text):
#     ner_results = nlp(text)
#     entities = [(entity['word'], entity['entity_group']) for entity in ner_results]
#     return entities

from refined.inference.processor import Refined
refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikipedia")

def extract_entities_refined(text):
    ner_results = refined.process_text(text)
    # entities = [(entity.text, refined_mapping[entity.coarse_mention_type],entity.coarse_mention_type) 
    entities = [(entity.text, entity.coarse_mention_type,entity.coarse_mention_type) 
                for entity in ner_results]
    return entities


def get_results(dataset, with_save = False, path_to_save= lambda dataset : f'./ner/saves/results/{dataset.name()}/test_full.pkl'):
    path_to_save = path_to_save(dataset)
    # Evaluate SpaCy and Flair models on the test dataset
    true_labels = []
    spacy_pred_labels = []
    flair_pred_labels = []
    flair_multi_pred_labels, babelspace_pred_labels, refined_pred_labels = [], [], []
    flair_ontonote_base, flair_ontonote_large = [], [] 

    for i in tqdm(range(len(dataset))):
        true_labels.append(dataset[i]['spans'])

        spacy_entities = extract_entities_spacy(dataset[i]['text'])
        spacy_pred_labels.append(spacy_entities)

        # flair_entities = extract_entities_flair(dataset[i]['text'], tagger_flair)
        # flair_pred_labels.append(flair_entities)

        # flair_multi_entities = extract_entities_flair(dataset[i]['text'], tagger_flair_mutli)
        # flair_multi_pred_labels.append(flair_multi_entities)

        flair_ontonote_base_entities = extract_entities_flair(dataset[i]['text'], tagger_flair_ontonote)
        flair_ontonote_base.append(flair_ontonote_base_entities)

        # flair_ontonote_large_entities = extract_entities_flair(dataset[i]['text'], tagger_flair_ontonote_large)
        # flair_ontonote_large.append(flair_ontonote_large_entities)

        # babelsapce_entities = extract_entities_babelspace(dataset[i]['text'])
        # babelspace_pred_labels.append(babelsapce_entities)

        refined_entities = extract_entities_refined(dataset[i]['text'])
        refined_pred_labels.append(refined_entities)


    results = {
        "true_labels" : true_labels,
        "spacy" : spacy_pred_labels,
        # "flair" : flair_pred_labels,
        # "babelspace" : babelspace_pred_labels,
        "refined" : refined_pred_labels,
        "flair_ontonote_base" : flair_ontonote_base,
        "flair_ontonote_large" : flair_ontonote_large
    }

    if with_save :
        dump(results, path_to_save)

    f1 = get_f1s(results, dataset.get_tags())
    
    return results, f1


def get_f1s(results, tags):
    print(tags)
    cm = {}
    f1 = {}
    precision = {}
    recall = {}
    y_true, y_pred, all_nes= {},{}, {}
    for model in results : 
        if model != "true_labels" :
            cm[model],f1[model], precision[model], recall[model], y_true[model], y_pred[model], all_nes[model]= get_metrics_all(results[model], results['true_labels'], tags)
            show_cm_multi(cm[model],f1[model], precision[model], recall[model], model, tags = tags)
    return f1