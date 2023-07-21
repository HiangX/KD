import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.autograd import Variable
from .tools import CIndex
scaler = StandardScaler()


def do_final_learning(model, loader, lr_inner, reg_scale):
    model.train()
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=lr_inner, weight_decay=reg_scale)
    pbar = tqdm(loader)
    for feature, ystatus_ftrain, y_ftrain in pbar:
        x_ftrain = scaler.fit_transform(np.log(feature+1))
        x_batch = x_ftrain
        ystatus_batch = ystatus_ftrain
        y_batch = y_ftrain
        R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
        for i in range(y_batch.shape[0]):
            for j in range(y_batch.shape[0]):
                R_matrix_batch[i, j] = y_batch[j] >= y_batch[i]
        x_batch = Variable(torch.FloatTensor(x_batch), requires_grad=True)
        R_matrix_batch = Variable(torch.FloatTensor(R_matrix_batch), requires_grad=True)
        ystatus_batch = Variable(ystatus_batch, requires_grad=True)
        theta = model(x_batch)
        exp_theta = torch.reshape(torch.exp(theta), [x_batch.shape[0]])
        theta = torch.reshape(theta, [x_batch.shape[0]])

        # loss1
        loss = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))),
                                     torch.reshape(ystatus_batch, [x_batch.shape[0]])))
        # loss2
        # loss = -torch.mean((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))))
        # loss3
        # intent = (theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1)))
        # cnt=0
        # for item in intent:
        #     if item!=0:
        #         cnt+=1
        # loss = -(torch.sum(intent)/cnt)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return
def init_weight(model, loc, loader, lr_inner, reg_scale):
    model.train()
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=lr_inner, weight_decay=reg_scale)
    pbar = tqdm(loader)
    for feature, ystatus_ftrain, y_ftrain in pbar:
        x_ftrain = scaler.fit_transform(np.log(feature[:, loc]+1))
        x_batch = x_ftrain
        ystatus_batch = ystatus_ftrain
        y_batch = y_ftrain
        R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
        for i in range(y_batch.shape[0]):
            for j in range(y_batch.shape[0]):
                R_matrix_batch[i, j] = y_batch[j] >= y_batch[i]
        x_batch = Variable(torch.FloatTensor(x_batch), requires_grad=True)
        R_matrix_batch = Variable(torch.FloatTensor(R_matrix_batch), requires_grad=True)
        ystatus_batch = Variable(ystatus_batch, requires_grad=True)
        theta = model(x_batch)
        exp_theta = torch.reshape(torch.exp(theta), [x_batch.shape[0]])
        theta = torch.reshape(theta, [x_batch.shape[0]])

        # loss1
        loss = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))),
                                     torch.reshape(ystatus_batch, [x_batch.shape[0]])))
        # loss2
        # loss = -torch.mean((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))))
        # loss3
        # intent = (theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1)))
        # cnt=0
        # for item in intent:
        #     if item!=0:
        #         cnt+=1
        # loss = -(torch.sum(intent)/cnt)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return
def do_final_eval(trained_model, x_test, y_test, ystatus_test):
    trained_model.eval()
    x_batch = torch.FloatTensor(x_test)
    pred_batch_test = trained_model(x_batch)
    cind = CIndex(pred_batch_test, y_test, np.asarray(ystatus_test))
    return cind, pred_batch_test
def do_pair_learning(model1, model2, loader, lr_inner, reg_scale, location):
    model1.eval()
    model2.train()
    inner_optimizer = torch.optim.Adam(model2.parameters(), lr=lr_inner, weight_decay=reg_scale)
    pbar = tqdm(loader)
    for feature, ystatus_ftrain, y_ftrain in pbar:
        x_ftrain = scaler.fit_transform(np.log(feature+1))
        # x_gene = scaler.fit_transform(np.log(feature[:, location]+1))
        x_gene = scaler.fit_transform(np.log(feature[:, location]+1))
        x_batch = x_ftrain
        ystatus_batch = ystatus_ftrain
        x_batch = Variable(torch.FloatTensor(x_batch), requires_grad=True)
        x_gene = Variable(torch.FloatTensor(x_gene), requires_grad=True)
        y_reason = model1(x_gene)
        y_reason = y_reason.squeeze(-1)
        # y_batch = y_ftrain
        y_batch = y_reason
        R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
        for i in range(y_batch.shape[0]):
            for j in range(y_batch.shape[0]):
                R_matrix_batch[i, j] = y_batch[j] >= y_batch[i]

        R_matrix_batch = Variable(torch.FloatTensor(R_matrix_batch), requires_grad=True)
        ystatus_batch = Variable(ystatus_batch, requires_grad=True)

        hazard = model1(x_gene)
        theta = model2(x_batch)

        exp_theta = torch.reshape(torch.exp(theta), [x_batch.shape[0]])
        theta = torch.reshape(theta, [x_batch.shape[0]])
        MSE = torch.nn.MSELoss()
        # loss1
        # a = torch.mul(exp_theta, R_matrix_batch)
        # b = torch.sum(a, dim=1)
        # b2 = torch.log(b)
        # c = theta - b2
        # d = torch.reshape(ystatus_batch, [x_batch.shape[0]])
        # e = torch.mul(c, d)
        # loss1 = torch.mean(e)
        loss1 = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))),
                                     torch.reshape(ystatus_batch, [x_batch.shape[0]])))
        # loss2
        loss2 = -torch.mean((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))))
        loss3 = MSE(theta, hazard.squeeze(-1))


        loss = loss1
        # loss3
        # intent = (theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1)))
        # cnt=0
        # for item in intent:
        #     if item!=0:
        #         cnt+=1
        # loss = -(torch.sum(intent)/cnt)
        # print(loss)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return
def do_pair_learning_mix(model1, model2, model3, loader, lr_inner, reg_scale, location):
    model1.eval()
    model2.train()
    model3.eval()
    inner_optimizer = torch.optim.Adam(model2.parameters(), lr=lr_inner, weight_decay=reg_scale)
    pbar = tqdm(loader)
    for feature, ystatus_ftrain, y_ftrain in pbar:
        x_ftrain = scaler.fit_transform(np.log(feature+1))
        x_gene = scaler.fit_transform(np.log(feature[:, location]+1))
        x_batch = x_ftrain
        ystatus_batch = ystatus_ftrain
        x_batch = Variable(torch.FloatTensor(x_batch), requires_grad=True)
        x_gene = Variable(torch.FloatTensor(x_gene), requires_grad=True)
        y_reason = model1(x_gene)
        y_reason = y_reason.squeeze(-1)
        y_batch = y_reason
        R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
        for i in range(y_batch.shape[0]):
            for j in range(y_batch.shape[0]):
                R_matrix_batch[i, j] = y_batch[j] >= y_batch[i]

        R_matrix_batch = Variable(torch.FloatTensor(R_matrix_batch), requires_grad=True)
        ystatus_batch = Variable(ystatus_batch, requires_grad=True)

        hazard = model1(x_gene)
        theta = model2(x_batch)
        target = model3(x_batch)

        exp_theta = torch.reshape(torch.exp(theta), [x_batch.shape[0]])
        theta = torch.reshape(theta, [x_batch.shape[0]])
        MSE = torch.nn.MSELoss()

        loss1 = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))),
                                     torch.reshape(ystatus_batch, [x_batch.shape[0]])))
        # loss2
        loss2 = MSE(theta, target.squeeze(-1))


        loss = loss1 + loss2

        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return
def do_pair_learning_distill(model1, model2, loader, lr_inner, reg_scale):
    model1.eval()
    model2.train()
    inner_optimizer = torch.optim.Adam(model2.parameters(), lr=lr_inner, weight_decay=reg_scale)
    pbar = tqdm(loader)
    for feature, ystatus_ftrain, y_ftrain in pbar:
        x_ftrain = scaler.fit_transform(np.log(feature+1))
        # x_gene = scaler.fit_transform(np.log(feature[:, location]+1))
        x_gene = scaler.fit_transform(np.log(feature+1))
        x_batch = x_ftrain
        ystatus_batch = ystatus_ftrain
        x_batch = Variable(torch.FloatTensor(x_batch), requires_grad=True)
        x_gene = Variable(torch.FloatTensor(x_gene), requires_grad=True)
        y_reason = model1(x_gene)
        y_reason = y_reason.squeeze(-1)
        # y_batch = y_ftrain
        y_batch = y_reason
        R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
        for i in range(y_batch.shape[0]):
            for j in range(y_batch.shape[0]):
                R_matrix_batch[i, j] = y_batch[j] >= y_batch[i]
        R_matrix_batch = Variable(torch.FloatTensor(R_matrix_batch), requires_grad=True)
        ystatus_batch = Variable(ystatus_batch, requires_grad=True)
        hazard = model1(x_gene)
        theta = model2(x_batch)
        exp_theta = torch.reshape(torch.exp(theta), [x_batch.shape[0]])
        theta = torch.reshape(theta, [x_batch.shape[0]])
        MSE = torch.nn.MSELoss()
        # loss1
        # a = torch.mul(exp_theta, R_matrix_batch)
        # b = torch.sum(a, dim=1)
        # b2 = torch.log(b)
        # c = theta - b2
        # d = torch.reshape(ystatus_batch, [x_batch.shape[0]])
        # e = torch.mul(c, d)
        # loss1 = torch.mean(e)
        loss1 = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))),
                                     torch.reshape(ystatus_batch, [x_batch.shape[0]])))
        loss3 = MSE(theta, hazard.squeeze(-1))


        loss = loss1 + loss3
        # loss3
        # intent = (theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1)))
        # cnt=0
        # for item in intent:
        #     if item!=0:
        #         cnt+=1
        # loss = -(torch.sum(intent)/cnt)
        # print(loss)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return    model1.eval()
    model2.train()
    inner_optimizer = torch.optim.Adam(model2.parameters(), lr=lr_inner, weight_decay=reg_scale)
    pbar = tqdm(loader)
    for feature, ystatus_ftrain, y_ftrain in pbar:
        x_ftrain = scaler.fit_transform(np.log(feature+1))
        x_gene = scaler.fit_transform(np.log(feature+1))
        x_batch = x_ftrain
        ystatus_batch = ystatus_ftrain
        x_batch = Variable(torch.FloatTensor(x_batch), requires_grad=True)
        x_gene = Variable(torch.FloatTensor(x_gene), requires_grad=True)
        y_reason = model2(x_gene)
        y_reason = y_reason.squeeze(-1)
        # y_batch = y_ftrain
        y_batch = y_reason
        R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
        for i in range(y_batch.shape[0]):
            for j in range(y_batch.shape[0]):
                R_matrix_batch[i, j] = y_batch[j] >= y_batch[i]

        R_matrix_batch = Variable(torch.FloatTensor(R_matrix_batch), requires_grad=True)
        ystatus_batch = Variable(ystatus_batch, requires_grad=True)

        hazard = model1(x_gene)
        theta = model2(x_batch)
        exp_theta = torch.reshape(torch.exp(theta), [x_batch.shape[0]])
        theta = torch.reshape(theta, [x_batch.shape[0]])
        MSE = torch.nn.MSELoss()
        # loss1
        # a = torch.mul(exp_theta, R_matrix_batch)
        # b = torch.sum(a, dim=1)
        # b2 = torch.log(b)
        # c = theta - b2
        # d = torch.reshape(ystatus_batch, [x_batch.shape[0]])
        # e = torch.mul(c, d)
        # loss1 = torch.mean(e)
        loss1 = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))),
                                     torch.reshape(ystatus_batch, [x_batch.shape[0]])))
        # loss2
        # loss2 = -torch.mean((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))))
        loss3 = MSE(theta, hazard.squeeze(-1))

        loss = loss1+loss3
        # loss3
        # intent = (theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1)))
        # cnt=0
        # for item in intent:
        #     if item!=0:
        #         cnt+=1
        # loss = -(torch.sum(intent)/cnt)
        # print(loss)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return