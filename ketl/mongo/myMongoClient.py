import pickle
from pymongo import MongoClient
import pandas as pd
import hashlib

# Create a hashlib object for the SHA-256 hash
hash_object = hashlib.sha256()

ONEDRIVE_STORAGE_HASH = "b!AexAOmJBX0GFrFCDPly37vK7ahDKKlBHgbMmpv0CDCXt3nRYonxISZkyQ12XYZPz"
BANQUE_STORAGE_HASH = "b!sYa62uHHtUqTlMT05hXNNDimI6dJj1xDlCQOeoCbtQ2y7z_UU_D9QKTSaBBIB995"
GOLD_DOCUMENT_FOLDER = [ONEDRIVE_STORAGE_HASH, "Banque documents", "Filed"]
PREDICTION_DOCUMENT_FOLDER = [BANQUE_STORAGE_HASH, "Clients"]

class MyMongoClient :
    def __init__(self,
                 db,
        MONGO_HOST = 'localhost',
        MONGO_PORT = 43413,
        MONGO_USER = 'root',
        MONGO_PASSWORD = "7pk_ht*JvMjUpXFF!qxE" # TODO: Remove this password from the code
        ):
        self.db = db
        self.mongo_client = MongoClient(f'mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}')
        self.all_docs = None

    def get_all_field_of_documents(self):
        cursor = self.mongo_client[self.db]['documents']
        documents = cursor.find({})
        fields = set()
        for i, doc in enumerate(documents):
            for field in doc['information']:
                fields.add(field)
        return fields
    
    def get_str_to_match(self, fields):
        """
        return a dictionary with 
        value : the value
        field : the field name 
        doc_hash : the hash of the docuement it is in 
        """
        cursor = self.mongo_client[self.db]['documents']
        documents = cursor.find({})

        str_to_match = []
        for i, doc in enumerate(documents):
            for field in fields:
                if field in doc['information'] :#'sign' in str(doc) or 'fournisseur' in str(doc) or 'destinataire' in str(doc):
                    str_to_match.append(
                        {"value": doc['information'][field], 
                        'field' : field, 
                        'doc_hash': doc['_id']} 
                    )
        return str_to_match
    
    def get_values_to_match(self, fields):
        # We load all the stings that were not entities
        all_values = self.get_str_to_match(self.db, fields)

        df = pd.DataFrame(all_values)
        df['doc_field'] = df.apply(lambda row : (row['field'], row['doc_hash']), axis = 1)
        df_grouped = pd.DataFrame(df.groupby('value').doc_field.agg(set)).reset_index()
        out = []
        for idx, row in df_grouped.iterrows():
            out.append({
                'value' : row['value'],
                'docs_hashs' : row['doc_field']
                })
        return out

    def get_all_entities(self):
        cursor = self.mongo_client[self.db]['entities']
        entities = cursor.find({})
        return entities
    
    def get_all_documents(self) :
        if not self.all_docs :
            cursor = self.mongo_client[self.db]['documents']
            self.all_docs = list(cursor.find({}))
        return self.all_docs
    
    def get_docs_labels(self, path_storage):
        all_docs = self.get_all_documents()
        filtered_data = []
        file_names = []
        for item in all_docs:
            for file in item.get("files", [{}]) :
                if file.get("fullPath")[:len(path_storage)] == path_storage:
                    filtered_data.append(item)
                    file_names.append(file['fileName'])
        # filtered_data = [item for item in all_docs 
        #                  if item.get("files", [{}])[0].get("fullPath")[:len(path_storage)] == path_storage]
        # print([doc['files'][0]['fileName'] for doc in filtered_data[0:3]])
        labels = {doc['_id'] : {
            label['name'] : label['value'] for label in doc['labels']}
                     for doc in filtered_data}
        return labels, file_names
    
    def get_labels_versions(self, path_storage):
        all_docs = self.get_all_documents()
        filtered_data = []
        file_names = []
        labels = []
        for item in all_docs:
            for file in item.get("files", [{}]) :
                if file.get("fullPath")[:len(path_storage)] == path_storage:
                    filtered_data.append(item)
                    file_names.append(file['fileName'])
        # filtered_data = [item for item in all_docs 
        #                  if item.get("files", [{}])[0].get("fullPath")[:len(path_storage)] == path_storage]
        # print([doc['files'][0]['fileName'] for doc in filtered_data[0:3]])
        for doc in filtered_data :
            for label in doc['labels']:
                for version in label['versions'] :
                    labels.append(
                        (doc['_id'], label['name'], version['value'], version['confidence'], version['modelName'] if version['modelName'] else 'user', version['createdOn'])
                    )
        return pd.DataFrame(labels, columns=['doc_id', 'label_name', 'label_value', 'confidence', 'model', 'created_on'])
    
    def get_results(self):
        gold_docs_labels, gold_names = load_gold_labels()
        pred_docs_labels, pred_names = self.get_docs_labels(PREDICTION_DOCUMENT_FOLDER)
        missing, wrong, right, total = 0,0,0,0
        acc_by_doc = {}
        # print(gold_names)
        # print(pred_names)
        # print(len(gold_docs_labels),len(pred_docs_labels))
        # print(set(gold_docs_labels.keys()) & set(pred_docs_labels.keys()))

        for i, doc_labels in enumerate(gold_docs_labels) :
            if doc_labels not in pred_docs_labels:
                print(f"{gold_names[i]} is not in prediction label")
                continue
            doc_right, doc_total = 0,len(gold_docs_labels[doc_labels])
            for key in gold_docs_labels[doc_labels] :
                if key in ['language', 'filename date'] or '-' in key:
                    doc_total -= 1
                elif key in pred_docs_labels[doc_labels]:
                    if pred_docs_labels[doc_labels][key] == gold_docs_labels[doc_labels][key] :
                        doc_right +=1
                    else :
                        wrong +=1 
                        print(f"{gold_names[i]} Wrong field {key}. Found {pred_docs_labels[doc_labels][key]} instead of {gold_docs_labels[doc_labels][key]}")
                    
                else :
                    print(f"{gold_names[i]}Missing field : {key}")
                    missing += 1
            right += doc_right
            total += doc_total
            acc_by_doc[doc_labels] = doc_right / doc_total
        if not total :
            return "No document in common"
        general_accuracy = right / total
        others = {
            "missing" : missing,
            "wrong" : wrong,
            "right" : right,
            "total" : total
        }
        return general_accuracy, acc_by_doc, others 
    
    def get_labels_from_doc_hash(self, doc_hash):
        gold_docs_labels, gold_names = load_gold_labels()
        pred_docs_labels, pred_names = self.get_docs_labels(PREDICTION_DOCUMENT_FOLDER)
        return gold_docs_labels[doc_hash], pred_docs_labels[doc_hash]

    def get_hash_of_doc(self, doc_id):
        hash_object = hashlib.sha256()
        input_string = list(self.mongo_client[self.db]['contents'].find({'_id' : doc_id}))[0]['content']
        # Update the hash object with the bytes of your string
        hash_object.update(input_string.encode('utf-8'))

        # Get the hexadecimal representation of the hash
        return hash_object.hexdigest()
    
def load_gold_labels():
    with open("./data/2023-11-01-gold_labels_filenames_clean", 'rb') as f :
        res = pickle.load(f)
    return res['labels'], res['file_names']