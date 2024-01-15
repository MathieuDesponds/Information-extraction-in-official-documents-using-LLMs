import os
import pandas as pd
from ner.llm_ner.few_shots_techniques import FewShotsTechnique
from ner.llm_ner.prompt_techniques.pt_abstract import PromptTechnique

from ner.Datasets.Conll2003Dataset import CONLL2003_TAGS
from ner.Datasets.OntoNotes5Dataset import ONTONOTE5_TAGS

from ner.process_results import get_metrics_all, show_cm_multi
from ner.utils import get_student_conf_interval, load, dump

class ResultInstance():
    def __init__(self, model, nb_few_shots, prompt_technique, few_shot_tecnique, verifier, results, gold, data_train, data_test, elapsed_time= -1, with_precision = False, seed = -1, noshots = False, plus_plus = False, tags = ['LOC', 'PER', 'ORG', 'MISC']) -> None:
        self.model = model
        self.nb_few_shots = nb_few_shots
        self.prompt_technique = prompt_technique
        self.few_shot_tecnique = few_shot_tecnique
        self.verifier = verifier
        self.noshots = noshots
        self.results = results
        self.gold = gold
        self.len_data_train = len(data_train)
        self.len_data_test = len(data_test)
        self.elapsed_time = elapsed_time
        self.with_precision = with_precision
        self.seed = seed
        self.plus_plus = plus_plus
        self.tags = tags

        self.cm, self.f1, self.precision, self.recall = None, None, None,None
        
    def get_scores(self, with_y = False):
        if not self.f1 :
            self.cm, self.f1, self.precision, self.recall, y_true, y_pred, nes = get_metrics_all(self.results, self.gold, self.tags)
        if with_y :
            return self.cm, self.f1, self.precision, self.recall, y_true, y_pred
        return self.cm, self.f1, self.precision, self.recall
        
    
    def show_cm(self) :
        show_cm_multi(*self.get_scores(), self.model, tags = self.tags)

    def get_dict(self) -> dict:
        self.get_scores()
        return {
            'model' : self.model,
            'noshots' : self.noshots,
'prompt_technique' : str(self.prompt_technique),
'few_shot_tecnique' : str(self.few_shot_tecnique),
'nb_few_shots' : self.nb_few_shots,
'verifier' : self.verifier,
'with_precision' : self.with_precision,
'len_data_train' : self.len_data_train,
'len_data_test' : self.len_data_test,
'f1' : self.f1,
'precision' : self.precision,
'recall' : self.recall,
'elapsed_time' : self.elapsed_time if hasattr(self, 'elapsed_time') else -1 ,
'plus_plus' : self.plus_plus if hasattr(self, 'plus_plus') else False,
'seed' : self.seed if hasattr(self, 'seed') else -1  
       }

    def analyse_results(self):
        for i in range(len(self.results)):
            print(i)
            print(self.gold[i])
            print(self.results[i])
            print("--------------------------------------------")


class ResultInstanceWithConfidenceInterval():
    def __init__(self, res_insts : list[ResultInstance]):
        self.res_insts = res_insts

    def get_scores(self):
        f1s, precisions, recalls = [], [], [] 
        for inst in self.res_insts :
            _, f1, precision, recall = inst.get_scores()
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            self.f1_mean, self.f1_conf_inter = get_student_conf_interval(f1s)
            self.precision_mean, self.precision_conf_inter = get_student_conf_interval(precisions)
            self.recall_mean, self.recall_conf_inter = get_student_conf_interval(recalls)
        return {"f1_mean" : self.f1_mean, "f1_conf_inter" : self.f1_conf_inter, 
                "precision_mean" : self.precision_mean, "precision_conf_inter" : self.precision_conf_inter, 
                "recall_mean" : self.recall_mean, "recall_conf_inter" : self.recall_conf_inter}
    
    def get_dict(self) -> dict:
            self.get_scores()
            return {
            "f1_mean" : self.f1_mean, "f1_conf_inter" : self.f1_conf_inter, 
            'model' : self.res_insts[0].model,
            'noshots' : self.res_insts[0].noshots,
            'prompt_technique' : self.res_insts[0].prompt_technique,
            'few_shot_tecnique' : self.res_insts[0].few_shot_tecnique,
            'nb_few_shots' : self.res_insts[0].nb_few_shots,
            'precision' :  self.res_insts[0].with_precision if hasattr(self.res_insts[0], 'with_precision') else "no-precision",
            'plus_plus' : self.res_insts[0].plus_plus if hasattr(self.res_insts[0], 'plus_plus') else False,
            'verifier' : self.res_insts[0].verifier,
            'len_data_train' : self.res_insts[0].len_data_train,
            'len_data_test' : self.res_insts[0].len_data_test,
            'nb_test_run' : len(self.res_insts),
            'confidence_interval' : 0.95,
            'distribution_used' : 'Student',
            "precision_mean" : self.precision_mean, "precision_conf_inter" : self.precision_conf_inter, 
            "recall_mean" : self.recall_mean, "recall_conf_inter" : self.recall_conf_inter
        }

def save_result_instance(res_inst : ResultInstance):
    file_path = f"./ner/saves/results/conll2003_cleaned/{res_inst.model}/{res_inst.prompt_technique}/{res_inst.few_shot_tecnique}_{res_inst.nb_few_shots}_{res_inst.verifier}_{res_inst.len_data_train}_{res_inst.len_data_test}.pkl"
    dump(res_inst, file_path)

def save_result_instance_with_CI(res_inst : ResultInstanceWithConfidenceInterval, dataset :str = "conll2003_cleaned"):
    file_path = f"./ner/saves/results/{dataset}/{res_inst.res_insts[0].model}/{res_inst.res_insts[0].prompt_technique}/{res_inst.res_insts[0].nb_few_shots}_{res_inst.res_insts[0].plus_plus}_{res_inst.res_insts[0].len_data_test*len(res_inst.res_insts)}.pkl"    
    dump(res_inst, file_path)

def load_all_results(root_directory = "ner/saves/results/conll2003_cleaned"):

    # Initialize a list to store the loaded data
    results = []
    raw_results= []

    # Walk through the directory and its subdirectories
    for foldername, subfolders, filenames in os.walk(root_directory):
        for filename in filenames:
            # Construct the full path of the file
            file_path = os.path.join(foldername, filename)

            # Check if the file has a .pkl extension (assuming you are looking for pickle files)
            if file_path.endswith(".pkl"):
                res_inst = load(file_path)
                if isinstance(res_inst ,ResultInstanceWithConfidenceInterval) :
                    precision = file_path.split('_')[-1][:-4]
                    noshots = 'noshots' in foldername
                    for r in res_inst.res_insts:
                        r.prompt_technique = r.prompt_technique if r.prompt_technique != '<>' else 'wrapper'
                        r.with_precision = precision
                        r.noshots = noshots
                        r.tags = ONTONOTE5_TAGS if "ontonote" in root_directory else CONLL2003_TAGS
                    res_inst.noshots = noshots
                    results.append(res_inst.get_dict())
                    raw_results.append(res_inst)
                    # save_result_instance_with_CI(res_inst)
    return pd.DataFrame(results).sort_values('f1_mean', ascending = False), raw_results

def load_result(model : str, pt : str, fst : str, nb_few_shots = 5, 
                verifier = None, len_data_train = 1538, len_data_test = 50, with_precision = 'no-precision', nb_runs = None):
    file_path = f"./ner/saves/results/ontonote5/{model}/{pt}/{fst}_{nb_few_shots}_{verifier}_{len_data_train}_{len_data_test}_{f'{nb_runs}_' if nb_runs else ''}{with_precision}.pkl"
    return load_all_results(file_path)