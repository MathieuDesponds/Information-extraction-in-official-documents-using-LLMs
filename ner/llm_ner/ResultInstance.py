import pickle

from ner.process_results import get_metrics_all, show_cm_multi
from ner.utils import get_student_conf_interval, load, dump

class ResultInstance():
    def __init__(self, model, nb_few_shots, prompt_technique, few_shot_tecnique, verifier, results, gold, data_train, data_test) -> None:
        self.model = model
        self.nb_few_shots = nb_few_shots
        self.prompt_technique = prompt_technique
        self.few_shot_tecnique = few_shot_tecnique
        self.verifier = verifier
        self.results = results
        self.gold = gold
        self.len_data_train = len(data_train)
        self.len_data_test = len(data_test)

        self.cm, self.f1, self.precision, self.recall = None, None, None,None
        
    def get_scores(self):
        if not self.cm :
            self.cm, self.f1, self.precision, self.recall, y_true, y_pred, nes = get_metrics_all(self.results, self.gold)
        return self.cm, self.f1, self.precision, self.recall
    
    def show_cm(self) :
        show_cm_multi(self.model, **self.get_scores())

    def get_dict(self) -> dict:
        self.get_scores()
        return {
            'model' : self.model,
'prompt_technique' : self.prompt_technique,
'few_shot_tecnique' : self.few_shot_tecnique,
'nb_few_shots' : self.nb_few_shots,
'verifier' : self.verifier,
'len_data_train' : self.len_data_train,
'len_data_test' : self.len_data_test,
'f1' : self.f1,
'precision' : self.precision,
'recall' : self.recall
       }


class ResultInstanceWithConfidenceInterval():
    def __init__(self, res_insts : list[ResultInstance]):
        self.res_insts = res_insts

    def get_scores(self):
        f1s, precisions, recalls = [], [], [] 
        for inst in self.res_insts :
            if not inst.cm :
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
            "precision_mean" : self.precision_mean, "precision_conf_inter" : self.precision_conf_inter, 
            "recall_mean" : self.recall_mean, "recall_conf_inter" : self.recall_conf_inter,
            'model' : self.res_insts[0].model,
            'prompt_technique' : self.res_insts[0].prompt_technique,
            'few_shot_tecnique' : self.res_insts[0].few_shot_tecnique,
            'nb_few_shots' : self.res_insts[0].nb_few_shots,
            'verifier' : self.res_insts[0].verifier,
            'len_data_train' : self.res_insts[0].len_data_train,
            'len_data_test' : self.res_insts[0].len_data_test,
            'nb_test_run' : len(self.res_insts),
            'confidence_interval' : 0.95,
            'distribution_used' : 'Student'
        }

def save_result_instance(res_inst : ResultInstance):
    file_path = f"./ner/saves/results/conll2003_cleaned/{res_inst.model}/{res_inst.prompt_technique}/{res_inst.few_shot_tecnique}_{res_inst.nb_few_shots}_{res_inst.verifier}_{res_inst.len_data_train}_{res_inst.len_data_test}.pkl"
    dump(res_inst, file_path)

def save_result_instance_with_CI(res_inst : ResultInstanceWithConfidenceInterval):
    file_path = f"./ner/saves/results/conll2003_cleaned/{res_inst.res_insts[0].model}/{res_inst.res_insts[0].prompt_technique}/{res_inst.res_insts[0].few_shot_tecnique}_{res_inst.res_insts[0].nb_few_shots}_{res_inst.res_insts[0].verifier}_{res_inst.res_insts[0].len_data_train}_{res_inst.res_insts[0].len_data_test}.pkl"
    dump(res_inst, file_path)