import time
import numpy as np
import torch
from torch import nn, optim
from model import Classifier, AE
from utils import try_gpu

# modeling stuff.
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def train_MLP(features, X_train, Y_train, epochs):

    pro1_index = list(X_train[:,0])
    pro2_index = list(X_train[:,1])

    Y_train = torch.FloatTensor(Y_train).to(device=try_gpu())

    model = Classifier(686, 1).to(device=try_gpu())

    loss_fcn = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        t = time.time()
        model.train()
        logits = model(features, pro1_index, pro2_index)
        loss = loss_fcn(logits, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
           print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),"time=", "{:.5f}".format(time.time() - t))

    return model


def train_eval_MLP(features, X_train, Y_train, X_test, epochs):

    model = train_MLP(features, X_train, Y_train, epochs)

    pro1_index = list(X_test[:,0])
    pro2_index = list(X_test[:,1])

    model.eval()
    with torch.no_grad():
        y_prob = model(features, pro1_index, pro2_index).data.cpu().numpy()

    return y_prob, model, pro1_index, pro2_index


def concat_features(features, index):
    '''Concatenate features of samples paired with input'''

    pro1_index = list(index[:, 0])
    pro2_index = list(index[:, 1])

    return pro1_index, pro2_index, np.concatenate([features[pro1_index], features[pro2_index]], axis = 1)


def train_eval_AE(features):
    model = AE().to(device=try_gpu())

    loss_fcn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters())

    for epoch in range(100):
        model.train()
        logits = model(features)
        loss = loss_fcn(logits, features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        encoded_features = model.encoder(features)

    return encoded_features


def train_eval_RF(features, X_train, Y_train, X_test, config):
    _, _, X_train_fea = concat_features(features, X_train)
    model = RandomForestClassifier(**config)
    Y_train = np.ravel(Y_train)
    model.fit(X_train_fea, Y_train)

    pro1_index, pro2_index, X_test_fea = concat_features(features, X_test)
    test_probs = model.predict_proba(X_test_fea)

    return test_probs[:, 1][:, np.newaxis], model, pro1_index, pro2_index


def train_eval_SVM(features, X_train, Y_train, X_test, config):
    _, _, X_train_fea = concat_features(features, X_train)
    # model = SVC(kernel='linear', probability=True, **config)
    model = SVC(kernel='rbf', probability=True, **config)
    Y_train = np.ravel(Y_train)
    model.fit(X_train_fea, Y_train)

    pro1_index, pro2_index, X_test_fea = concat_features(features, X_test)
    test_probs = model.predict_proba(X_test_fea)

    return test_probs[:, 1][:, np.newaxis], model, pro1_index, pro2_index


def train_eval_XGBoost(features, X_train, Y_train, X_test, config):
    _, _, X_train_fea = concat_features(features, X_train)
    model = xgb.XGBClassifier(**config)
    Y_train = np.ravel(Y_train)
    model.fit(X_train_fea, Y_train)

    pro1_index, pro2_index, X_test_fea = concat_features(features, X_test)
    test_probs = model.predict_proba(X_test_fea)

    return test_probs[:, 1][:, np.newaxis], model, pro1_index, pro2_index


def train_eval_KNN(features, X_train, Y_train, X_test, config):
    _, _, X_train_fea = concat_features(features, X_train)
    model = KNeighborsClassifier(**config)
    Y_train = np.ravel(Y_train)
    model.fit(X_train_fea, Y_train)

    pro1_index, pro2_index, X_test_fea = concat_features(features, X_test)
    test_probs = model.predict_proba(X_test_fea)

    return test_probs[:, 1][:, np.newaxis], model, pro1_index, pro2_index


def train_eval_LR(features, X_train, Y_train, X_test, config):
    _, _, X_train_fea = concat_features(features, X_train)
    model = LogisticRegression(**config)
    Y_train = np.ravel(Y_train)
    model.fit(X_train_fea, Y_train)

    pro1_index, pro2_index, X_test_fea = concat_features(features, X_test)
    test_probs = model.predict_proba(X_test_fea)

    return test_probs[:, 1][:, np.newaxis], model, pro1_index, pro2_index
