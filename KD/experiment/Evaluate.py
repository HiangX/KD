from sklearn.model_selection import KFold
import pandas as pd
from ..datasets import Cancer_fold
from torch.utils.data import DataLoader
from ..utils import *
from ..models import *
import torch
import numpy as np
__all__ = [
    'Evaluate',
           ]

class Evaluate:
    """Evaluate is a  class to evaluate the trained model on various cancer.
    Evaluate supports two modes for evaluation: 1)cross validation(cv=True). 2)inference once(cv=False).
    To run the experiment with only one class,
    we have to impose some restrictions to make
    sure the robustness of the code:
    1. The path of your model file should recorded in 'args.resume'
    2. By default, cross validation is performed if the cv parameter is not specified.


    Parameters
    ----------
    cancer_type : str
        The type of the cancer. ('ACC')

    csv_folder : str
        Path to the csv files. ('./data')

    folder : str
        Path to the gene csv files.

    seed : int
        Random initialization parameters.
    """
    def __init__(self, cancer_type, folder, resume, csv_folder, round=1, seed=42):
        self.resume = resume
        self.csv_folder = csv_folder
        self.cancer_type = cancer_type
        self.folder = folder
        self.round = round
        self.seed = seed

    def eval(self, cv=True):
        setup_seed(self.seed)
        if cv:
            self.eval_cv()
        else:
            self.eval_inference()

    def eval_cv(self):
        cancer_type = self.cancer_type
        print(f'Cancer Type: {cancer_type}')
        model = DAPLModel()
        print('Load Model!')
        model.load_state_dict(torch.load(self.resume))
        df = pd.read_csv(f'{self.csv_folder}/gene/{cancer_type}.csv')
        kf = KFold(n_splits=5, random_state=43, shuffle=True).split(np.arange(df.shape[0]))
        for round in range(self.round):
            print(f'ROUND:{round}')
            _, val_idx = next(kf)
            test = df.loc[val_idx, :].reset_index(drop=True)
            test_dataset = Cancer_fold(test, self.folder, cancer_type)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
            x_test, ytime, ystatus = norm_data(test_loader)
            CI, _ = do_final_eval(model, x_test, ytime, ystatus)
            print(f'Cancer: {cancer_type} Round: {round} Cindex: {CI}')

    def eval_inference(self):

        cancer_type = self.cancer_type
        print(f'Cancer Type: {cancer_type}')
        model = DAPLModel()
        print('Load Model!')
        model.load_state_dict(torch.load(self.resume))
        df = pd.read_csv(f'{self.csv_folder}/gene/{cancer_type}.csv')
        test = df
        test_dataset = Cancer_fold(test, self.folder, cancer_type)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        x_test, ytime, ystatus = norm_data(test_loader)
        CI, _ = do_final_eval(model, x_test, ytime, ystatus)
        print(f'Cancer: {cancer_type} Cindex: {CI}')
