import pickle

from ner.process_results import get_metrics_all, show_cm_multi

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

def save_result_instance(res_inst : ResultInstance):
    with open(f"./ner/saves/results/colnn2003_cleaned/{res_inst.model}/{res_inst.prompt_technique}/{res_inst.few_shot_tecnique}_{res_inst.verifier}_{res_inst.len_data_train}_{res_inst.len_data_test}.pkl", 'wb') as f :
        pickle.dump(res_inst, f) 