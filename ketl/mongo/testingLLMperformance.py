import pandas as pd
from myMongoClient import *

def filter_rows(group):
    model_name = group['model'].unique().item()
    if 'entity' in model_name:
        return group.loc[[group[group['confidence']>=0.5]['confidence'].idxmax()]]
    else:
        return group#.loc[group['confidence'].idxmax()]

def format_date(row):
    if 'date' in row['label_name'].lower():
        original_date = pd.to_datetime(row['label_value'])
        return original_date.strftime('%Y-%m-%d')
    else:
        return row['label_value']
# Function to check if 'label_value' for 'model'='user' is in any row where 'model'!='user'
def check_label_value(group):
    user_row = group[group['model'] == 'user']
    
    if not user_row.empty:
        user_label_value = user_row['label_value'].iloc[0]
        
        # Check if 'label_value' for 'model'='user' is in any row where 'model'!='user'
        return int(any(user_row['label_value'].isin(group.loc[group['model'] != 'user', 'label_value'])))
    else:
        return 0
    
def get_LLM_performance(mongo_client : MyMongoClient, PREDICTION_DOCUMENT_FOLDER):
    label_versions = mongo_client.get_labels_versions(PREDICTION_DOCUMENT_FOLDER).head(60)
    # take only into consideration the label of the useer and of the llm
    label_versions = label_versions[(label_versions['model'] == 'user') | (label_versions['model'].str.contains('llm - openai azure'))]

    #Do not take into consideration the label with '-' and language 
    label_versions = label_versions[~label_versions['label_name'].str.contains('-')]



    # Filter the matching rows where we keep only the maximum
    label_versions = label_versions.groupby(['doc_id', 'label_name', 'model']).apply(filter_rows).reset_index(drop=True)

    # Apply the function to the DataFrame
    label_versions['label_value'] = label_versions.apply(format_date, axis=1)



    # Applying the function to each group
    result_df = label_versions.groupby(['doc_id', 'label_name']).apply(check_label_value).reset_index(name='output')

    score = result_df['output'].sum()/len(result_df)
    return score, result_df, label_versions