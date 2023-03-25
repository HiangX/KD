import os.path
from random import sample
from sklearn.model_selection import KFold
import pandas as pd
from datasets import Cancer,Cancer_fold
from torch.utils.data import DataLoader
from utils import *
from models import *
from config import config

def main(args):
    setup_seed(args.seed)
    print(args)
    cancer_type = args.cancer_type
    print(f'Cancer Type: {cancer_type}')
    weight_model = WeightLayer(args.dim)
    location = pd.read_csv(f'{args.csv_folder}/location/{cancer_type}_gene5.csv')['pos'].values
    reason_model = WeightLayer(len(location))
    save_distill = os.path.join(args.resume, 'distill')
    weight_model.load_state_dict(torch.load(f'{save_distill}/{cancer_type}.pth'))
    save_reason = os.path.join(args.resume, 'reason')
    reason_model.load_state_dict(torch.load(f'{save_reason}/{cancer_type}.pth'))
    print('Start Final Training!')
    df = pd.read_csv(f'{args.csv_folder}/gene/{cancer_type}.csv')
    kf = KFold(n_splits=5, random_state=43, shuffle=True).split(np.arange(df.shape[0]))
    trn_idx, val_idx = next(kf)
    test = df.loc[val_idx, :].reset_index(drop=True)
    test_dataset = Cancer_fold(test, args.folder)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    x_test, ytime, ystatus = norm_data(test_loader)
    model = DAPLModel()
    save_final = os.path.join(args.resume, 'final')
    model.load_state_dict(save_final)
    CI, score_test = do_final_eval(model, x_test, ytime, ystatus)
    print(f"Cancer: {cancer_type} Cindex: {CI}")

if __name__=='__main__':
    args = config()
    main(args)
