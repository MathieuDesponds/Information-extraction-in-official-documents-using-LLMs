import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from ner.llm_ner.ResultInstance import load_all_results
from ner.utils import get_student_conf_interval


def get_results(with_ft, few_shots = [0,3], dataset = "ontonote5"):
    if dataset == "ontonote5":
        df_results = load_all_results(root_directory = f"ner/saves/results/{dataset}/")
        if with_ft:
            df_to_show = df_results[df_results['model'].str.contains('ft') & df_results['model'].str.contains('2000')]
        else : 
            df_to_show = df_results[~df_results['model'].str.contains('ft')]
        df_res = df_to_show[df_to_show['precision'] == '300']
        df_res = df_res[df_res['nb_few_shots'].isin(few_shots)]
    elif dataset == "conll2003_cleaned" :
        df_results = load_all_results(root_directory = f"ner/saves/results/{dataset}/")
        df_results['nb_samples'] = df_results['nb_test_run'] * df_results['len_data_test']
        df_res = df_results[df_results['nb_samples'] == 300]
        if with_ft:
            df_res = df_res[df_res['model'].str.contains('ft')]
        else : 
            df_res = df_res[~df_res['model'].str.contains('ft')]
        
        df_res = df_res[df_res['nb_few_shots'].isin(few_shots)]
    return df_res[['model', 'f1_mean', 'f1_conf_inter', 'prompt_technique',
            'few_shot_tecnique', 'nb_few_shots', 'precision', 'plus_plus']]


def show_results(with_ft = False, datasets = ["ontonote5", "conll2003_cleaned"]):
    
   
    # Set up the plot
    fig, axs = plt.subplots(1,len(datasets),figsize = (15,8))

    # Set up jitter for x-axis positions
    jitter = 0.1
    for i, dataset in enumerate(datasets) :
        df = get_results(with_ft, dataset = dataset)
        df['tech_name'] = df.apply(lambda row :f"With {row['nb_few_shots']} few_shots {'and ++' if row['plus_plus']  else ''}", axis = 1)

        # Convert the f1_conf_inter column to a tuple of floats
        df['f1_conf_inter'] = df['f1_conf_inter'].apply(lambda x: ast.literal_eval(x))
        # Loop through unique tech_names
        for h, tech_name in enumerate(df['tech_name'].unique()):
            # Filter the DataFrame for the current tech_name
            tech_df = df[df['tech_name'] == tech_name].sort_values('prompt_technique')
            
            # Add jitter to x-axis positions
            x_positions = np.arange(len(tech_df['prompt_technique'])) + h * jitter
            # Plot the f1_mean values with jitter
            axs[i].scatter(x_positions, tech_df['f1_mean'], label=tech_name)
            
            # Plot the confidence intervals using error bars
            for j, (_, row) in enumerate(tech_df.iterrows()):
                low, high = row['f1_conf_inter']
                axs[i].errorbar(x_positions[j], row['f1_mean'], yerr=[[row['f1_mean'] - low], [high - row['f1_mean']]], color='gray', capsize=5, capthick=2, fmt='none')

        # Customize the plot
        axs[i].set_xticks(np.arange(len(df['prompt_technique'].unique())) + 0.5 * (len(df['tech_name'].unique()) - 1) * jitter)
        axs[i].set_xticklabels(df.sort_values('prompt_technique')['prompt_technique'].unique(), rotation = 15)
        axs[i].set_xlabel('Prompt Technique', fontsize=12)
        axs[i].set_ylabel('F1 Mean', fontsize=12)
        axs[i].set_title(f'{dataset.capitalize()} : F1 Mean with Confidence Intervals', fontsize=14)
        axs[i].legend(fontsize=10)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        axs[i].legend()
    plt.savefig('ner/saves/results/ontonote5/graph_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_diff_plus_plus(with_ft = False, datasets = ["ontonote5"]):
    output = ""
    tables = {}
    for dataset in datasets:
        df_to_show = get_results(with_ft, dataset =dataset)

        # Pivot the DataFrame to have 'True' and 'False' types as columns
        pivot_df = df_to_show.pivot(index=['prompt_technique',
            'few_shot_tecnique', 'nb_few_shots', 'precision'], columns='plus_plus', values='f1_mean').reset_index()

        # Subtract 'False' from 'True'
        pivot_df['result'] = pivot_df[True] - pivot_df[False]
        mean, conf = get_student_conf_interval(list(pivot_df['result']))
        # Display the result
        output += f"For {dataset.capitalize()} :\n    The mean difference between results that had a detailled explenation and thos that did not is \n     {mean} with confidence 95% student interval {conf}"
        tables[dataset] = pivot_df[['prompt_technique','few_shot_tecnique', 'nb_few_shots', 'precision', 'result', True, False]]
    return output, tables


def show_diff_few_shots(with_ft = False, datasets = ["ontonote5", "conll2003_cleaned"]):
    output = ""
    tables = {}
    for dataset in datasets:
        df_to_show = get_results(with_ft, dataset =dataset)

        # Pivot the DataFrame to have 'True' and 'False' types as columns
        pivot_df = df_to_show.pivot(index=['prompt_technique', 'plus_plus', 'precision'], columns='nb_few_shots', values='f1_mean').reset_index()

        # Subtract 'False' from 'True'
        pivot_df['result'] = pivot_df[3] - pivot_df[0]
        mean, conf = get_student_conf_interval(list(pivot_df['result']))
        # Display the result
        output += f"For {dataset.capitalize()} :\n    The mean difference between results that uses few_shots and those that do not and raw is \n     {mean} with confidence 95% student interval {conf}\n"
        tables[dataset] = pivot_df[['prompt_technique','plus_plus', 'precision', 'result', 3, 0]]
    return output, tables

def show_diff_ft(with_few_shots = False, datasets = ["ontonote5", "conll2003_cleaned"]):
    output = ""
    tables = {}
    for dataset in datasets:
        df_results = load_all_results(root_directory = f"ner/saves/results/{dataset}/")
        if with_few_shots:
            df_to_show = df_results[df_results['nb_few_shots'] == 3]
        else : 
            df_to_show = df_results[df_results['nb_few_shots'] == 0]
        if dataset == 'ontonote5' :
            df_to_show = df_to_show[df_to_show['precision'] == '300']
        elif dataset == 'conll2003_dataset' :
            df_to_show = df_to_show[df_results['nb_test_run'] * df_results['len_data_test'] == '300']
        df_to_show['with_ft'] = df_to_show['model'].str.contains('ft') & df_to_show['model'].str.contains('2000')
        df_to_show = df_to_show[['f1_mean', 'f1_conf_inter', 'prompt_technique', 'nb_few_shots', 'precision', 'plus_plus', 'with_ft']]
        # Pivot the DataFrame to have 'True' and 'False' types as columns
        pivot_df = df_to_show.pivot(index=['prompt_technique',  'plus_plus','nb_few_shots'], columns='with_ft', values='f1_mean').reset_index()

        # Subtract 'False' from 'True'
        pivot_df['result'] = pivot_df[True] - pivot_df[False]
        mean, conf = get_student_conf_interval(list(pivot_df['result']))
        # Display the result
        output+= f"For {dataset.capitalize()} :\n    The mean difference between results that used a finetuned model and those that do not is \n     {mean} with confidence 95% student interval {conf}"
        tables[dataset]= pivot_df[['prompt_technique','plus_plus', 'nb_few_shots', 'result', True, False]]
    return output, tables

def show_results_few_shots(datasets = ["ontonote5", "conll2003_cleaned"]):
    fig, axs = plt.subplots(1, len(datasets), figsize=(15,10))
    for idx, dataset in enumerate(datasets):
        df_results = load_all_results(root_directory = f"ner/saves/results/{dataset}/")
        df_results = df_results[df_results['model'].str.contains('2000') & df_results['model'].str.contains('ft') | ~df_results['model'].str.contains('ft')]
        df_results['ft'] = df_results['model'].str.contains('ft') & df_results['model'].str.contains('2000')
        df_to_show = df_results[['model', 'f1_mean', 'f1_conf_inter', 'prompt_technique',
            'few_shot_tecnique', 'nb_few_shots', 'precision', 'plus_plus', 'ft']]
        df_to_show = df_to_show[df_to_show['plus_plus'] == True]
        if dataset == 'ontonote5' :
            df_to_show = df_to_show[df_to_show['precision'] == '300']
        elif dataset == 'conll2003_dataset' :
            df_to_show = df_to_show[df_results['nb_test_run'] * df_results['len_data_test'] == '300']
        df_to_show = df_to_show[df_to_show['prompt_technique'].isin(['wrapper', 'discussion'])]

        df_to_show['x_names']= df_to_show.apply(lambda row :f"{row['prompt_technique']} | {'With' if row['ft'] else 'Without'} finetuning", axis = 1)
        df_to_show['tech_name'] = df_to_show.apply(lambda row :f"With {' ' if row['nb_few_shots'] <10 else ''}{row['nb_few_shots']} few shots", axis = 1)

        df = df_to_show

        # Convert the f1_conf_inter column to a tuple of floats
        df['f1_conf_inter'] = df['f1_conf_inter'].apply(lambda x: ast.literal_eval(x))

        # Set up the plot

        # Set up jitter for x-axis positions
        jitter = 0.1

        # Loop through unique tech_names
        for i, tech_name in enumerate(df['tech_name'].sort_values().unique()):
            # Filter the DataFrame for the current tech_name
            tech_df = df[df['tech_name'] == tech_name].sort_values('x_names')
            
            # Add jitter to x-axis positions
            x_positions = np.arange(len(tech_df['x_names'])) + i * jitter
            # Plot the f1_mean values with jitter
            axs[idx].scatter(x_positions, tech_df['f1_mean'], label=tech_name)
            
            # Plot the confidence intervals using error bars
            for j, (_, row) in enumerate(tech_df.iterrows()):
                low, high = row['f1_conf_inter']
                axs[idx].errorbar(x_positions[j], row['f1_mean'], yerr=[[row['f1_mean'] - low], [high - row['f1_mean']]], color='gray', capsize=5, capthick=2, fmt='none')

        # Customize the plot
        axs[idx].set_xticks(np.arange(len(df['x_names'].unique())) + 0.5 * (len(df['tech_name'].unique()) - 1) * jitter)
        axs[idx].set_xticklabels(df.sort_values('x_names')['x_names'].unique(), rotation = 20)
        axs[idx].set_xlabel('Parameters', fontsize=12)
        axs[idx].set_ylabel('F1 Mean', fontsize=12)
        axs[idx].set_title(f'{dataset.capitalize()} F1 Mean with Confidence Intervals', fontsize=14)
        axs[idx].legend(fontsize=10)
        axs[idx].grid(axis='y', linestyle='--', alpha=0.7)
        axs[idx].legend()
    plt.savefig('ner/saves/results/ontonote5/impact_few_shots.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_diff_ft_few_shots(datasets = ["ontonote5", "conll2003_cleaned"]):
    output = ""
    tables = {}
    for idx, dataset in enumerate(datasets):
        df_results = load_all_results(root_directory = f"ner/saves/results/{dataset}/")
        df_results = df_results[df_results['model'].str.contains('2000') & df_results['model'].str.contains('ft') | ~df_results['model'].str.contains('ft')]
        df_results['ft'] = df_results['model'].str.contains('ft') & df_results['model'].str.contains('2000')
        df_to_show = df_results[['model', 'f1_mean', 'f1_conf_inter', 'prompt_technique',
            'few_shot_tecnique', 'nb_few_shots', 'precision', 'plus_plus', 'ft']]
        df_to_show = df_to_show[df_to_show['plus_plus'] == True]
        if dataset == 'ontonote5' :
            df_to_show = df_to_show[df_to_show['precision'] == '300']
        elif dataset == 'conll2003_dataset' :
            df_to_show = df_to_show[df_results['nb_test_run'] * df_results['len_data_test'] == '300']

        # Pivot the DataFrame to have 'True' and 'False' types as columns
        pivot_df = df_to_show.pivot(index=['prompt_technique',  'nb_few_shots'], columns='ft', values='f1_mean').reset_index()

        # Subtract 'False' from 'True'
        pivot_df['result'] = pivot_df[True] - pivot_df[False]
        mean, conf = get_student_conf_interval(list(pivot_df['result'].dropna()))
        # Display the result
        output += f"For {dataset.capitalize()} :\n    The mean score increase with few_shots between finetuning and without finetuning is \n     {mean} with confidence 95% student interval {conf}"
        tables['dataset'] = pivot_df[['prompt_technique', 'nb_few_shots', 'result', True, False]]
    return output, tables