import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, f1_score, precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn as sns

tag_type = ['LOC', 'PER', 'ORG', 'MISC']

     
def show_cm(cm, f1s, model):
    fig, axs= plt.subplots(1,len(tag_type)+1,sharey=True, figsize=(25,5))
    for i, tag in enumerate(["general"] + tag_type):
    # labels, title and ticks
        sns.heatmap(cm[tag], annot=True, fmt='g', ax=axs[i])  #annot=True to annotate cells, ftm='g' to disable scientific notation
        axs[i].set_xlabel('Predicted labels')
        axs[i].set_ylabel('True labels')
        axs[i].set_title(f'tag {tag}')
        axs[i].xaxis.set_ticklabels(['Not classified as ne', 'Classified as ne'])
        axs[i].yaxis.set_ticklabels(['Not classified as ne', 'Classified as ne'])
    fig.suptitle(f"Confusion matrix for tags of model {model}\nF1-score by tags : {f1s}", fontsize=16)
    fig.subplots_adjust(top=0.80)
    plt.show()

def show_cm_multi(cm,f1, precision, recall, model, nb_few_shots = None, nb_training_samples = 0, tags = ['LOC', 'PER', 'ORG', 'MISC']):
    classes = tags +['None']
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust the font size if needed
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)

    # Add labels and title
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Model {model}\nF1: {f1:.2f} prec: {precision:.2f}, rec: {recall:.2f}")
    plt.show()


def show_cm_multi_2(cm,f1, precision, recall, tags = ['LOC', 'PER', 'ORG', 'MISC']):
    classes = tags +['None']
    
    # Create a figure with three subplots in the same row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, mod in enumerate(cm):
        sns.set(font_scale=1.2)  # Adjust the font size if needed
        sns.heatmap(cm[mod], annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[i])
        
        # Add labels and title for each subplot
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        # axes[i].set_title(f"Model {model} with {f'{nb_few_shots} few-shots and' if nb_few_shots else ''} {nb_training_samples} training samples\nF1-score : {f1[mod]:.2f}, precision : {precision[mod]:.2f}, recall : {recall[mod]:.2f}")
        axes[i].set_title(f"Model {mod}\nF1: {f1[mod]:.2f} prec: {precision[mod]:.2f}, rec: {recall[mod]:.2f}")

    plt.show()
    
def show_cm_multi_extended(cm,f1, precision, recall, model):
    classes = refined_labels
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1.2)  # Adjust the font size if needed
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)

    # Add labels and title
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for model {model}\n F1-score : {f1:.2f}, precision : {precision:.2f}, recall : {recall:.2f}, ")
    plt.show()
    
def get_metrics(results, gold):
    cm_tag, f1_scores = {}, {}
    gold_nes_by_tag, all_nes_by_tag,results_nes_by_tag = {},{},{}
    y_true, y_pred = [], []
    for i in range(len(results)):
        gold_nes = [ne[0] for ne in gold[i]]
        results_nes = [ne[0] for ne in results[i] if ne[1] != 'None']
        all_nes = list(results_nes)
        all_nes.extend([g for g in gold_nes if g not in results_nes])
        y_true.extend([1 if n in gold_nes else 0 for n in all_nes ])
        y_pred.extend([1 if n in results_nes else 0 for n in all_nes ])
    cm_tag['general'] = confusion_matrix(y_true, y_pred, labels = [0,1])
    f1_scores['general'] = f1_score(y_true, y_pred)

    for tag in tag_type:
        y_true, y_pred = [], []
        for i in range(len(results)):
            results_nes_by_tag[tag] = [ne[0] for ne in  results[i] if ne[1] == tag ]
            gold_nes_by_tag[tag] = [ne[0] for ne in  gold[i] if ne[1] == tag ]
            all_nes_by_tag[tag] = list(results_nes_by_tag[tag])
            all_nes_by_tag[tag].extend([g for g in gold_nes_by_tag[tag] if g not in results_nes_by_tag[tag]])
            y_true.extend([1 if n in gold_nes_by_tag[tag] else 0 for n in all_nes_by_tag[tag]])
            y_pred.extend([1 if n in results_nes_by_tag[tag] else 0 for n in all_nes_by_tag[tag]])
        cm_tag[tag] = confusion_matrix(y_true, y_pred, labels = [0,1])
        f1_scores[tag] = f1_score(y_true, y_pred)
    return cm_tag,f1_scores


def get_metrics_all(results, gold, tags = ['LOC', 'PER', 'ORG', 'MISC'], average = 'weighted', with_y_conf = False):
    y_true, y_pred, y_conf, all_nes = [], [], [], []
    for i in range(len(results)):
        gold_nes = {ne[0] :ne[1] for ne in gold[i]}
        res_sanitized = [n for n in results[i] if n[1] != 'None']
        results_nes = {ne[0] : ne[1] for ne in [n for n in res_sanitized if n[1] != 'None']}
        nes = list(res_sanitized)
        nes.extend([g for g in gold[i] if g[0] not in results_nes.keys()])
        y_true.extend([gold_nes[n[0]] if n[0] in gold_nes.keys() else 'None' for n in nes])
        y_pred.extend([results_nes[n[0]] if n[0] in results_nes.keys() else 'None' for n in nes])
        if with_y_conf :
            y_conf.extend([results_nes_conf[n[0]] if n[0] in results_nes.keys() else 'None' for n in nes])
            results_nes_conf= {ne[0] : ne[2] for ne in [n for n in res_sanitized if n[1] != 'None']}
        all_nes.extend(nes)
    y_true = [y if y else 'None' for y in y_true]
    y_pred = [y if y else 'None' for y in y_pred]
    y_pred = ['ORG' if y == 'organisation' else y for y in y_pred]
    cm = None#confusion_matrix(y_true, y_pred, labels = tags + ['None'])
    precision, recall, f1, _= precision_recall_fscore_support(y_true, y_pred, average = average, zero_division=0)
    if with_y_conf :
        return cm,f1, precision, recall, y_true, y_pred, [ne[0] for ne in all_nes],y_conf
    return cm,f1, precision, recall, y_true, y_pred, [ne[0] for ne in all_nes]


def get_metrics_all_extended(results, gold):
    y_true, y_pred, all_nes = [], [], []
    for i in range(len(results)):
        gold_nes = {ne[0] :ne[1] for ne in gold[i]}
        results_nes = {ne[0] : ne[2] for ne in results[i]}
        nes = list(results[i])
        nes.extend([g for g in gold[i] if g[0] not in results_nes.keys()])
        y_true.extend([gold_nes[n[0]] if n[0] in gold_nes.keys() else 'None' for n in nes])
        y_pred.extend([results_nes[n[0]] if n[0] in results_nes.keys() else 'None' for n in nes])
        all_nes.extend(nes)
    y_pred = [y if y != None else 'None' for y in y_pred]
    cm = confusion_matrix(y_true, y_pred, labels = refined_labels)
    precision, recall, f1, _= precision_recall_fscore_support(y_true, y_pred, average = 'micro')
    return cm,f1, precision, recall, y_true, y_pred, [ne[0] for ne in all_nes]

refined_labels = ['None', 'LOC', 'GPE', 'PER', 'ORG', 'MISC', 'EVENT', 'WORK_OF_ART', 'FAC',
       'LANGUAGE']
spacy_labels = ['LOC', 'GPE', 'PER','PERSON', 'ORG', 'MISC',  'DATE', 'EVENT', 'CARDINAL', 'ORDINAL', 'TIME',
       'NORP', 'MONEY', 'QUANTITY', 'PERCENT',
       'WORK_OF_ART', 'LAW', 'PRODUCT', 'FAC', 'LANGUAGE']