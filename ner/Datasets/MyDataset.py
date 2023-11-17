from ner.utils import sentence_transformer, get_embbeding

mapping_abbr_word =  {'PER' : 'person', 'ORG' : 'organisation', 'LOC' : 'location', 'MISC' : 'miscellaneous'}

class MyDataset():
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def get_spans(data_point, tag_mapping):

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
    
    @staticmethod
    def add_llama_ner_tags(data_point, tags = ['PER', 'ORG', 'LOC', 'MISC']):
        mapping_BMES = {'PER' : 1, 'ORG' : 3, 'LOC' : 5, "MISC": 7}
        llamas_tokens = {}
        current_entity = []
        for tag in tags :
            llamas_tokens[tag] = []
            current_entity = []
            for token, ent_tag in zip(data_point['tokens'], data_point['ner_tags']):
                if ent_tag == mapping_BMES[tag] :  # Beginning of a named entity
                    if current_entity:
                        current_entity[0] = "@@"+current_entity[0]
                        current_entity[-1] = current_entity[-1] + "##"
                        llamas_tokens[tag].extend(current_entity)
                    current_entity = [token]
                elif ent_tag == mapping_BMES[tag]+1 and current_entity:  # Inside a named entity
                    current_entity.append(token)
                elif current_entity: #we finish to retrieve the entity
                    current_entity[0] = "@@"+current_entity[0]
                    current_entity[-1] = current_entity[-1] + "##"
                    llamas_tokens[tag].extend(current_entity)
                    current_entity = []
                    llamas_tokens[tag].append(token)
                else :
                    llamas_tokens[tag].append(token)
                
            # Check if there's an entity at the end of the sequence
            if current_entity:
                current_entity[0] = "@@"+current_entity[0]
                current_entity[-1] = current_entity[-1] + "##"
                llamas_tokens[tag].extend(current_entity)

        data_point['llama_tokenized'] = llamas_tokens
        data_point['llama_text'] = {key: ' '.join(value) for key, value in llamas_tokens.items()}
        return data_point
    
    @staticmethod
    def add_llama_ner_tags_2(data_point, tags = ['PER', 'ORG', 'LOC', 'MISC']):
        text : str = data_point['text']
        for ne, tag in {ne : tag for ne, tag in  data_point['spans']}.items():
            text = text.replace(ne, f"<{mapping_abbr_word[tag]}>{ne}</{mapping_abbr_word[tag]}>")
        data_point['llama_text_2'] = text
        return data_point
    
    @staticmethod
    def add_sentence_embedding(data_point):
        data_point['sentence_embedding'] = sentence_transformer.encode(data_point['text'])
        return data_point
    
    @staticmethod
    def adjust_entity_embeddings_idx(data_point, idx):
        for value in data_point['entity_embeddings'] :
            value['idx'] = idx
        return data_point
    
    @staticmethod
    def add_entity_embeddings(data_point, idx):
        data_point['entity_embeddings'] = get_embbeding(data_point['text'], idx)
        return data_point
    
    def get_all_embeddings(self):
        all_entity_embeddings = []
        for embeddings in self.dataset['entity_embeddings'] :
            all_entity_embeddings.extend(embeddings)
        return all_entity_embeddings
    
    def select(self, iterable):
        self.dataset = self.dataset.select(iterable)