from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import pickle
class Cancer(Dataset):
    '''
    cancer dataset
    '''
    def __init__(self, sur, pkl_path):
        df_sur = pd.read_csv(sur)
        self.id = df_sur['sample_id'].values
        self.time = df_sur['os_time'].values
        self.status = df_sur['os_status'].values
        self.pkl_path = pkl_path
    def __len__(self):
        return len(self.id)
    def __getitem__(self, index):
        id = self.id[index]
        time = int(self.time[index])
        status = float(self.status[index])
        file_path = os.path.join(self.pkl_path, id+'.pkl')
        with open(file_path, 'rb') as f:
            feature = pickle.load(f)
        return feature, status, time
class Cancer_fold(Dataset):
    '''
    cancer dataset
    '''
    def __init__(self, df_sur, pkl_path):
        # df_sur = pd.read_csv(sur)
        self.id = df_sur['sample_id'].values
        self.time = df_sur['os_time'].values
        self.status = df_sur['os_status'].values
        self.pkl_path = pkl_path
    def __len__(self):
        return len(self.id)
    def __getitem__(self, index):
        id = self.id[index]
        time = int(self.time[index])
        status = float(self.status[index])
        file_path = os.path.join(self.pkl_path, id+'.pkl')
        with open(file_path, 'rb') as f:
            feature = pickle.load(f)
        return feature, status, time
