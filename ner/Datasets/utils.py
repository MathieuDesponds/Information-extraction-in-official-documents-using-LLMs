import pickle 

import pandas as pd
import nltk
import re 
from datasets import Dataset


# Download the NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define a function to check if a sentence contains verbs
def contains_verbs(text):
    words = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(words)
    
    for word, pos in pos_tags:
        if pos.startswith('VB'):  # Words tagged as verbs
            return True
    return False

# Define a function to check if there are more numeric words than non-numeric words
def has_more_or_equal_numeric_words(text):
    words = nltk.word_tokenize(text)
    numeric_words = [word for word in words if re.match(r'^\d+$', word)]  # Numeric words
    only_letter_words = [word for word in words if re.match(r"^[a-zA-Z]+$", word)]  # Non-numeric words
    
    # print(words, numeric_words, only_letter_words)
    return len(numeric_words) >= len(only_letter_words)


def show_duplicates(df):
    result = df.groupby('text').agg(
        count=('spans', 'count'),
        values_list=('spans', list)
    ).reset_index()
    return result.sort_values('count')[-15:]

def clean_dataset(dataset : Dataset, save_name = None):
    filtered_dataset= dataset.filter(lambda row : contains_verbs(row['text']) and not has_more_or_equal_numeric_words(row['text']))
    # Filter out entries that contain verbs
    filtered_df = pd.DataFrame(filtered_dataset)
    # not_filtered_df = tmp[~tmp['contains_verbs'] | tmp['more_numeric_numbers']]
    filtered_df = filtered_df.drop_duplicates('text')

    ds = Dataset.from_pandas(filtered_df).remove_columns('__index_level_0__')

    return ds