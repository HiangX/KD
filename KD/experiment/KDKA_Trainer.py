from sklearn.model_selection import KFold
from random import sample
from ..utils import *
from ..models import *
from ..datasets import *
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
__all__ = [
    'KDKA_Trainer'
           ]

class KDKA_Trainer:
    """Trainer is a  class to train the model on various cancer.
    Trainer supports two modes for trainning: 1)all(mode='all'). 2)few-shot.
    To run the experiment with only one class,
    we have to impose some restrictions to make
    sure the robustness of the code:
    1. The path of your model file should recorded in 'args.resume'
    2. By default, use all the data for training.


    Parameters
    ----------
    cancer_type : str
        The type of the cancer. ('ACC')

    folder : str
        Path to the gene csv files.

    csv_folder : str
        Path to the csv files. ('./data')

    round : int
        Number of rounds.

    dim : int
        Dimensions of input data.

    batch_size : int

    output_folder : str
        Save address of output data.

    mode : str
        Training mode, few samples or full data.

    lr : float
        Learning rate.

    seed : int
        Random initialization parameters.
    """
    def __init__(self, cancer_type, folder, resume, csv_folder, round, dim, batch_size, output_folder, lr, seed, mode='all'):
        self.resume = resume
        self.csv_folder = csv_folder
        self.cancer_type = cancer_type
        self.folder = folder
        self.round = round
        self.seed = seed
        self.dim = dim
        self.batch_size = batch_size
        self.output_folder = output_folder
        self.mode = mode
        self.lr = lr

        setup_seed(seed)
    def train(self):
        cancer_type = self.cancer_type
        print(f'Cancer Type: {cancer_type}')
        weight_model = WeightLayer(self.dim)
        location = pd.read_csv(f'{self.csv_folder}/location/{cancer_type}_gene5.csv')['pos'].values
        reason_model = WeightLayer(len(location))
        train_dataset = Cancer(f'{self.csv_folder}/gene/{cancer_type}_Clean.csv', self.folder, cancer_type)
        test_dataset = Cancer(f'{self.csv_folder}/gene/{cancer_type}.csv', self.folder, cancer_type)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        save_distill = os.path.join(self.output_folder, 'distill')
        print('Start distillation!')
        x_test, ytime, ystatus = norm_data(test_loader)
        x_val, ytimeval, ystatusval = norm_data(val_loader)
        bic = 0
        for epoch in range(30):
            do_final_learning(weight_model, train_loader, lr_inner=self.lr, reg_scale=0.1)
            if epoch % 1 == 0:
                CI, score_test = do_final_eval(weight_model, x_test, ytime, ystatus)
                CI_val, _ = do_final_eval(weight_model, x_val, ytimeval, ystatusval)
                if CI > bic:
                    bic = CI
                    torch.save(weight_model.state_dict(), f'{save_distill}/{cancer_type}.pth')
        weight_model.load_state_dict(torch.load(f'{save_distill}/{cancer_type}.pth'))
        print('Start Reasoning!')
        x_test, ytime, ystatus = norm_data(test_loader, location)
        x_val, ytimeval, ystatusval = norm_data(val_loader, location)
        bic = 0
        save_reason = os.path.join(self.output_folder, 'reason')
        for epoch in range(120):
            init_weight(reason_model, location, loader=train_loader, lr_inner=1e-3, reg_scale=0.1)
            if epoch % 1 == 0:
                CI, score_test = do_final_eval(reason_model, x_test, ytime, ystatus)
                CI_val, _ = do_final_eval(reason_model, x_val, ytimeval, ystatusval)
                if CI > bic:
                    bic = CI
                    torch.save(reason_model.state_dict(), f'{save_reason}/{cancer_type}.pth')
        reason_model.load_state_dict(torch.load(f'{save_reason}/{cancer_type}.pth'))

        print('Start Final Training!')
        df = pd.read_csv(f'{self.csv_folder}/gene/{cancer_type}.csv')
        kf = KFold(n_splits=5, random_state=43, shuffle=True).split(np.arange(df.shape[0]))
        trn_idx, val_idx = next(kf)
        test = df.loc[val_idx, :].reset_index(drop=True)
        test_dataset = Cancer_fold(test, self.folder, cancer_type)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        x_test, ytime, ystatus = norm_data(test_loader)
        IC25 = 0.
        cidx = []
        for round in range(25):
            model = DAPLModel()
            print(f'ROUND:{round}')
            if self.mode == 'all':
                train_idx = trn_idx.tolist()
            else:
                train_idx = sample(trn_idx.tolist(), k=20)
            train_se = df.loc[train_idx, :].reset_index(drop=True)
            train_dataset = Cancer_fold(train_se, self.folder, cancer_type)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            bic = 0
            save_final = os.path.join(self.output_folder, 'final')
            for epoch in range(30):
                do_pair_learning_mix(reason_model, model, weight_model, loader=train_loader, lr_inner=1e-4,
                                     reg_scale=0.1, location=location)
                if epoch % 1 == 0:
                    CI, score_test = do_final_eval(model, x_test, ytime, ystatus)
                    if CI > bic:
                        bic = CI
                        if bic > IC25:
                            IC25 = bic
                            torch.save(model.state_dict(), f'{save_final}/{cancer_type}.pth')
            cidx.append(bic)
            print(f'Cancer: {cancer_type} Round: {round} Cindex: {bic}')
        av = sum(cidx)/len(cidx)
        print(f'Cancer: {cancer_type} Ave_Cindex: {av}')
