import time
import math
import torch
from torch import nn, optim
from model import VGAE, DSIPredictor
from utils import try_gpu, loss_function
import copy


def train_VGAE(features, adj_norm, adj_label, epochs, outdim):

    pos_weight = float(adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) / adj_label.sum()
    norm = adj_label.shape[0] * adj_label.shape[0] / float((adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) * 2)

    model = VGAE(features.shape[1], 343, outdim, 0.1).to(device=try_gpu())
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    hidden_emb = None

    for epoch in range(epochs):
        t = time.time()
        model.train()
        recovered, z, mu, logstd = model(features, adj_norm)#, sigmoid = False)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logstd=logstd,
                             norm=norm, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        hidden_emb = mu

        if epoch % 10 == 0:
           print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),"time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    return model, hidden_emb


def train_DSIPredictor(vgae_dict, features, adj, X_train, Y_train, epochs):

    pro1_index = list(X_train[:,0])
    pro2_index = list(X_train[:,1])

    Y_train = torch.FloatTensor(Y_train).to(device=try_gpu())

    model = DSIPredictor(686, 1).to(device=try_gpu())
    model.gc1.weight.data = copy.deepcopy(vgae_dict['gc1.weight'])
    model.gc2.weight.data = copy.deepcopy(vgae_dict['gc2.weight'])

    loss_fcn = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    mini_batch = 64
    num_batch = int(math.floor(len(pro1_index) / mini_batch))

    for epoch in range(epochs):
        t = time.time()
        model.train()

        for k in range(0, num_batch):
            mini_pro1_index = pro1_index[k * mini_batch:(k + 1) * mini_batch]
            mini_pro2_index = pro2_index[k * mini_batch:(k + 1) * mini_batch]
            mini_Y_train = Y_train[k * mini_batch:(k + 1) * mini_batch]

            logits = model(features, adj, mini_pro1_index, mini_pro2_index)
            loss = loss_fcn(logits, mini_Y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "time=",
                  "{:.5f}".format(time.time() - t))

    return model


def train_eval_DSIPredictor(vgae_dict, features, adj, X_train, Y_train, X_test, epochs = 100):

    model = train_DSIPredictor(vgae_dict, features, adj, X_train, Y_train, epochs=epochs)

    pro1_index = list(X_test[:,0])
    pro2_index = list(X_test[:,1])

    model.eval()
    with torch.no_grad():
        y_prob = model(features, adj, pro1_index, pro2_index).data.cpu().numpy()

    return y_prob, model, pro1_index, pro2_index


def finetune_DSIPredictor(model, features, adj, X_train, Y_train, epochs):

    pro1_index = list(X_train[:,0])
    pro2_index = list(X_train[:,1])

    Y_train = torch.FloatTensor(Y_train).to(device=try_gpu())

    loss_fcn = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    mini_batch = 64
    num_batch = int(math.floor(len(pro1_index) / mini_batch))

    for epoch in range(epochs):
        t = time.time()
        model.train()

        for k in range(0, num_batch):
            mini_pro1_index = pro1_index[k * mini_batch:(k + 1) * mini_batch]
            mini_pro2_index = pro2_index[k * mini_batch:(k + 1) * mini_batch]
            mini_Y_train = Y_train[k * mini_batch:(k + 1) * mini_batch]

            logits = model(features, adj, mini_pro1_index, mini_pro2_index)
            loss = loss_fcn(logits, mini_Y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "time=",
                  "{:.5f}".format(time.time() - t))

    return model


def finetune_eval_DSIPredictor(model, features, adj, X_train, Y_train, X_test, epochs = 100):

    model = finetune_DSIPredictor(model, features, adj, X_train, Y_train, epochs=epochs)

    pro1_index = list(X_test[:,0])
    pro2_index = list(X_test[:,1])

    model.eval()
    with torch.no_grad():
        y_prob = model(features, adj, pro1_index, pro2_index).data.cpu().numpy()

    return y_prob, model, pro1_index, pro2_index
