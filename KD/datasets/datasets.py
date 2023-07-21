from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class Cancer(Dataset):
    '''
    cancer dataset
    '''
    def __init__(self, sur, gene_path, type):
        df_sur = pd.read_csv(sur)
        self.id = df_sur['sample_id'].values
        self.time = df_sur['os_time'].values
        self.status = df_sur['os_status'].values
        self.gene = pd.read_csv(f'{gene_path}/{type}_data.csv')
    def __len__(self):
        return len(self.id)
    def __getitem__(self, index):
        id = self.id[index]
        time = int(self.time[index])
        status = float(self.status[index])
        feature = np.asarray(self.gene[id])
        return feature, status, time

class Cancer_fold(Dataset):
    '''
    cancer dataset
    '''
    def __init__(self, df_sur, gene_path, type):
        # df_sur = pd.read_csv(sur)
        self.id = df_sur['sample_id'].values
        self.time = df_sur['os_time'].values
        self.status = df_sur['os_status'].values
        self.gene = pd.read_csv(f'{gene_path}/{type}_data.csv')
    def __len__(self):
        return len(self.id)
    def __getitem__(self, index):
        id = self.id[index]
        time = int(self.time[index])
        status = float(self.status[index])
        feature = np.asarray(self.gene[id])
        return feature, status, time

