import time
import torch
from torch import nn, optim
from model import Classifier
from utils import try_gpu

def train_Classifier(features, X_train, Y_train, epochs):

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

def extract_logits(features, X_train, Y_train, X_test, epochs):

    model = train_Classifier(features, X_train, Y_train, epochs)

    pro1_index = list(X_test[:,0])
    pro2_index = list(X_test[:,1])

    model.eval()
    with torch.no_grad():
        y_prob = model(features, pro1_index, pro2_index).data.cpu().numpy()

    return y_prob, model, pro1_index, pro2_index

