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
    train_dataset = Cancer(f'{args.csv_folder}/gene/{cancer_type}_Clean.csv', args.folder)
    test_dataset = Cancer(f'{args.csv_folder}/gene/{cancer_type}.csv', args.folder)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    save_distill = os.path.join(args.output_folder, 'distill')
    print('Start distillation!')
    x_test, ytime, ystatus = norm_data(test_loader)
    x_val, ytimeval, ystatusval = norm_data(val_loader)
    bic = 0
    for epoch in range(30):
        do_final_learning(weight_model, train_loader, lr_inner=args.lr, reg_scale=args.reg_scale)
        if epoch % 1 == 0:
            CI, score_test = do_final_eval(weight_model, x_test, ytime, ystatus)
            CI_val, _ = do_final_eval(weight_model, x_val, ytimeval, ystatusval)
            if CI > bic:
                bic = CI
                torch.save(weight_model.state_dict(), f'{save_distill}/{cancer_type}.pth')
    weight_model.load_state_dict(torch.load(f'{save_distill}/{cancer_type}.pth'))
    print('Start Reasoning!')
    x_test,ytime,ystatus = norm_data(test_loader, location)
    x_val, ytimeval, ystatusval = norm_data(val_loader, location)
    bic = 0
    save_reason = os.path.join(args.output_folder, 'reason')
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
    df = pd.read_csv(f'{args.csv_folder}/gene/{cancer_type}.csv')
    kf = KFold(n_splits=5, random_state=43, shuffle=True).split(np.arange(df.shape[0]))
    trn_idx, val_idx = next(kf)
    test = df.loc[val_idx, :].reset_index(drop=True)
    test_dataset = Cancer_fold(test, args.folder)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    x_test, ytime, ystatus = norm_data(test_loader)
    IC25 = 0.
    for round in range(25):
        model = DAPLModel()
        print(f'ROUND:{round}')
        if args.mode=='all':
            train_idx = trn_idx.tolist()
        else:
            train_idx = sample(trn_idx.tolist(), k=20)
        train_se = df.loc[train_idx, :].reset_index(drop=True)
        train_dataset = Cancer_fold(train_se, args.folder)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        bic = 0
        save_final = os.path.join(args.output_folder, 'final')
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
        print(f'Cancer: {cancer_type} Round: {round} Cindex: {bic}')

if __name__=='__main__':
    args = config()
    main(args)
