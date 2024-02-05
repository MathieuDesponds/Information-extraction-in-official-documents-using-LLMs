from datetime import datetime
import pandas as pd
from myMongoClient import *

NONE_USER = '6536892d127f4f001df8215e'

GOLD_LABEL_FILE = "data/labels_results/2024-01-12_16:42:23_df_gold_labels.csv"

def filter_rows(group):
    model_name = group['model'].unique().item()
    if 'entity' in model_name:
        possibility = group[group['confidence']>=0.5]
        if not possibility.empty:
            return group.loc[[possibility['confidence'].idxmax()]]
        else:
            return possibility
    else:
        return group


def format_date(row):
    if 'date' in row['label_name'].lower():
        original_date = pd.to_datetime(row['label_value'])
        return original_date.strftime('%Y-%m-%d')
    else:
        return row['label_value']
# Function to check if 'label_value' for 'model'='user' is in any row where 'model'!='user'
def check_label_value(group):
    user_row = group[group['model'] == 'user']
    label_name = group['label_name'].unique().item()
    if not user_row.empty:
        user_label_value = user_row['label_value'].iloc[0]
        if label_name.lower() == 'client' and user_label_value == NONE_USER :
            return 1 if len(group[group['model'] != 'user']) == 0 else 0
        
        # Si la date validée par l'utilisateur en en 01,24 alors on valide si le LLM n'a rien proposé
        if label_name.lower() == 'relevant date': 
            dt = datetime.strptime(user_label_value, '%Y-%m-%d')
            # Check if the datetime is in January 2024
            if (dt.month == 1) and (dt.year == 2024) :
                return 1 if len(group[group['model'] != 'user']) == 0 else 0
            else :
                pass
        
        # Check if 'label_value' for 'model'='user' is in any row where 'model'!='user'
        return int(any(user_row['label_value'].isin(group.loc[group['model'] != 'user', 'label_value'])))
    else:
        return 0
    
def get_LLM_performance(mongo_client : MyMongoClient, doc_folder, with_gold, with_label_version = False, label_versions = None):
    if not with_label_version :
        label_versions = mongo_client.get_labels_versions(doc_folder)

        # take only into consideration the label of the useer and of the llm
        label_versions = label_versions[(label_versions['model'] == 'user') | (label_versions['model'].str.contains('llm - openai azure'))]

        #Do not take into consideration the label with '-' and language 
        # label_versions = label_versions[(~label_versions['label_name'].str.contains('-')) & ~label_versions['label_name'].isin(['language', 'description document'])]

        # Filter the matching rows where we keep only the maximum
        label_versions = label_versions.groupby(['doc_id', 'label_name', 'model']).apply(filter_rows).reset_index(drop=True)

        # Apply the function to the DataFrame
        label_versions['label_value'] = label_versions.apply(format_date, axis=1)

    if not with_gold :
        gold = pd.read_csv(GOLD_LABEL_FILE)
        doc_not_both1 = set(label_versions['doc_id'].tolist()).difference(set(gold['doc_id'].tolist()))
        doc_not_both2 = set(gold['doc_id'].tolist()).difference(set(label_versions['doc_id'].tolist()))
        
        print(f"len label_versions {len(label_versions['doc_id'].tolist())}\ndoc not both {len(doc_not_both1)} {len(doc_not_both2)} : {doc_not_both1} { doc_not_both2}")
        label_versions = pd.concat([label_versions, gold], ignore_index=True)

    label_versions = label_versions[(~label_versions['label_name'].str.contains('-')) & ~label_versions['label_name'].isin(['language', 'description document'])]
        # return label_versions[label_versions['doc_id'].isin(doc_not_both1.union(doc_not_both2))]

    # Applying the function to each group
    result_df = label_versions.groupby(['doc_id', 'label_name']).apply(check_label_value).reset_index(name='output')

    # score_by_fields = result_df['output'].sum()/len(result_df)
    # score_by_documents = result_df.groupby('doc_id').agg({'output' : 'mean'})['output'].mean()
    score_by_fields, score_by_documents = get_score_for_asked_fields(result_df, no_compare_doc = doc_not_both1.union(doc_not_both2))
    return score_by_fields,score_by_documents, result_df, label_versions

def get_results_by_label_name(score_df) :
    result_df = score_df.groupby('label_name')['output'].agg(['mean', 'count']).reset_index()
    result_df = result_df.rename(columns={'mean': 'mean_output', 'count': 'count_values'})

    # Sort the DataFrame by the count of values in descending order
    result_df = result_df.sort_values(by='count_values', ascending=False)
    return result_df

def get_results_by_doc_type(label_versions) :
    # return label_versions[label_versions['label_name'] == 'document type']
    doc_type_analysis = label_versions[label_versions['label_name'] == 'document type'].pivot(index=['doc_id'], columns='model', values='label_value').reset_index()
    doc_type_analysis['output'] = doc_type_analysis.apply(lambda row : row['llm - openai azure'] == row['user'], axis  = 1)
    doc_type_analysis.groupby('user').agg({'output' : ['mean', 'count']}).reset_index().sort_values([('output', 'count')], ascending = False)
    return doc_type_analysis

def get_doc_hash_wrong_value(score_df, label_name):
    return score_df[(score_df['label_name'] == label_name) & (score_df['output'] == 0)]['doc_id'].tolist()

def get_score_for_asked_fields(score_df, no_compare_doc):
    doc_wrong_type = get_doc_hash_wrong_value(score_df, 'document type') + list(no_compare_doc)
    score_df2 = score_df[~(score_df['doc_id'].isin(doc_wrong_type) & (~score_df['label_name'].isin(['client', 'document type'])))]
    score_by_fields = score_df2['output'].sum()/len(score_df2)
    score_by_documents = score_df2.groupby('doc_id').agg({'output' : 'mean'})['output'].mean()
    return score_by_fields,score_by_documents

