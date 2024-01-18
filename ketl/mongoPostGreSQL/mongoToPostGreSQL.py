import pandas as pd
def get_document_representation(document):
    return document['_id']


def get_labels(all_documents) :
    labels= []
    for doc in all_documents :
        for label in doc['labels']:
            for version in label['versions'] :
                labels.append(
                    (doc['_id'], label['name'], version['value'], version['confidence'], version['modelName'] if version['modelName'] else 'user', version['createdOn'])
                )
    return pd.DataFrame(labels, columns=['doc_id', 'label_name', 'label_value', 'confidence', 'model', 'created_on'])
