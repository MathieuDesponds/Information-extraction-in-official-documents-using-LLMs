from myMongoClient import *
import pandas as pd

ALL_PROMPTS_FILE = "data/saved_prompts/prompts_2024-01-23_14:22:20.csv"

def prepare_finetuning():
    pd.DataFrame(load(ALL_PROMPTS_FILE))