import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from ner.llm_ner.ResultInstance import load_all_results
from ner.utils import get_student_conf_interval

def show_results_ontonote(with_ft = False):
    df_results = load_all_results(root_directory = "ner/saves/results/ontonote5/mistral-7b-v0.1/")
    if with_ft:
        df_to_show = df_results[df_results['model'].str.contains('ft')]
    else : 
        df_to_show = df_results[~df_results['model'].str.contains('ft')]
    
    df_to_show = df_to_show[['f1_mean', 'f1_conf_inter', 'prompt_technique',
        'few_shot_tecnique', 'nb_few_shots', 'precision', 'plus_plus']]

    df_res = df_to_show[df_to_show['precision'] == '300']
    df_res['tech_name'] = df_res.apply(lambda row :f"With {row['nb_few_shots']} few_shots {'and ++' if row['plus_plus']  else ''}", axis = 1)
    df_res

    df = df_res

    # Convert the f1_conf_inter column to a tuple of floats
    df['f1_conf_inter'] = df['f1_conf_inter'].apply(lambda x: ast.literal_eval(x))

    # Set up the plot
    fig, ax = plt.subplots()

    # Set up jitter for x-axis positions
    jitter = 0.1

    # Loop through unique tech_names
    for i, tech_name in enumerate(df['tech_name'].unique()):
        # Filter the DataFrame for the current tech_name
        tech_df = df[df['tech_name'] == tech_name].sort_values('prompt_technique')
        
        # Add jitter to x-axis positions
        x_positions = np.arange(len(tech_df['prompt_technique'])) + i * jitter
        # Plot the f1_mean values with jitter
        ax.scatter(x_positions, tech_df['f1_mean'], label=tech_name)
        
        # Plot the confidence intervals using error bars
        for j, (_, row) in enumerate(tech_df.iterrows()):
            low, high = row['f1_conf_inter']
            ax.errorbar(x_positions[j], row['f1_mean'], yerr=[[row['f1_mean'] - low], [high - row['f1_mean']]], color='gray', capsize=5, capthick=2, fmt='none')

    # Customize the plot
    ax.set_xticks(np.arange(len(df['prompt_technique'].unique())) + 0.5 * (len(df['tech_name'].unique()) - 1) * jitter)
    ax.set_xticklabels(df.sort_values('prompt_technique')['prompt_technique'].unique(), rotation = 45)
    ax.set_xlabel('Prompt Technique', fontsize=12)
    ax.set_ylabel('F1 Mean', fontsize=12)
    ax.set_title('F1 Mean with Confidence Intervals', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.savefig('ner/saves/results/ontonote5/graph_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_diff_plus_plus(with_ft = False):
    df_results = load_all_results(root_directory = "ner/saves/results/ontonote5/mistral-7b-v0.1/")
    if with_ft:
        df_to_show = df_results[df_results['model'].str.contains('ft')]
    else : 
        df_to_show = df_results[~df_results['model'].str.contains('ft')]
    df_to_show = df_to_show[df_to_show['precision'] == '300']
    df_to_show = df_to_show[['f1_mean', 'f1_conf_inter', 'prompt_technique',
        'few_shot_tecnique', 'nb_few_shots', 'precision', 'plus_plus']]
    # Pivot the DataFrame to have 'True' and 'False' types as columns
    pivot_df = df_to_show.pivot(index=['prompt_technique',
        'few_shot_tecnique', 'nb_few_shots', 'precision'], columns='plus_plus', values='f1_mean').reset_index()

    # Subtract 'False' from 'True'
    pivot_df['result'] = pivot_df[True] - pivot_df[False]
    mean, conf = get_student_conf_interval(list(pivot_df['result']))
    # Display the result
    return f"The mean difference between plus plus and raw is {mean} with confidence 95% student interval {conf}",pivot_df[['prompt_technique',
        'few_shot_tecnique', 'nb_few_shots', 'precision', 'result', True, False]]

def show_diff_few_shots(with_ft = False):
    df_results = load_all_results(root_directory = "ner/saves/results/ontonote5/mistral-7b-v0.1/")
    if with_ft:
        df_to_show = df_results[df_results['model'].str.contains('ft')]
    else : 
        df_to_show = df_results[~df_results['model'].str.contains('ft')]
    df_to_show = df_to_show[df_to_show['precision'] == '300']
    df_to_show = df_to_show[['f1_mean', 'f1_conf_inter', 'prompt_technique', 'nb_few_shots', 'precision', 'plus_plus']]
    # Pivot the DataFrame to have 'True' and 'False' types as columns
    pivot_df = df_to_show.pivot(index=['prompt_technique', 'plus_plus', 'precision'], columns='nb_few_shots', values='f1_mean').reset_index()

    # Subtract 'False' from 'True'
    pivot_df['result'] = pivot_df[3] - pivot_df[0]
    mean, conf = get_student_conf_interval(list(pivot_df['result']))
    # Display the result
    return f"The mean difference between plus plus and raw is {mean} with confidence 95% student interval {conf}",pivot_df[['prompt_technique',
    'plus_plus', 'precision', 'result', 3, 0]]

def show_diff_ft(with_few_shots = False):
    df_results = load_all_results(root_directory = "ner/saves/results/ontonote5/mistral-7b-v0.1/")
    if with_few_shots:
        df_to_show = df_results[df_results['nb_few_shots'] == 3]
    else : 
        df_to_show = df_results[df_results['nb_few_shots'] == 0]
    df_to_show = df_to_show[df_to_show['precision'] == '300']
    df_to_show['with_ft'] = df_to_show['model'].str.contains('ft')
    df_to_show = df_to_show[['f1_mean', 'f1_conf_inter', 'prompt_technique', 'nb_few_shots', 'precision', 'plus_plus']]
    # Pivot the DataFrame to have 'True' and 'False' types as columns
    pivot_df = df_to_show.pivot(index=['prompt_technique',  'plus_plus', 'precision'], columns='with_ft', values='f1_mean').reset_index()

    # Subtract 'False' from 'True'
    pivot_df['result'] = pivot_df[True] - pivot_df[False]
    mean, conf = get_student_conf_interval(list(pivot_df['result']))
    # Display the result
    return f"The mean difference between plus plus and raw is {mean} with confidence 95% student interval {conf}",pivot_df[['prompt_technique',
    'plus_plus', 'precision', 'result', True, False]]



