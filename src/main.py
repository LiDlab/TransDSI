import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from load import load_data, load_dataset
from train import train_VGAE, extract_logits
from utils import save_logits, evaluate_logits, try_gpu, youden_index

def cross_val(vgae_dict, features_tensor, adj_norm, AC, X, Y, epochs, save_path):

    random_idx = list(range(AC.shape[0]))
    np.random.seed(888)
    np.random.shuffle(random_idx)
    AC = AC[random_idx]
    X = X[random_idx]
    Y = Y[random_idx]

    all_idx = list(range(AC.shape[0]))
    rs = KFold(n_splits=5, shuffle=True)
    cv_index_set = rs.split(all_idx)

    all_Y_label = []
    all_Y_pred = []
    all_AC_val = []
    fold = 0

    for train_idx, val_idx in cv_index_set:
        X_train = X[train_idx]
        X_val = X[val_idx]
        Y_train = Y[train_idx]
        Y_val = Y[val_idx]
        AC_val = AC[val_idx]

        #print("###################################")
        #print("The" + str(fold) + "cross validation is underway")
        fold = fold + 1

        Y_pred, model, pro1_index, pro2_index= extract_logits(vgae_dict, features_tensor, adj_norm, X_train, Y_train, X_val, epochs)

        all_Y_label.append(Y_val)
        all_Y_pred.append(Y_pred)
        all_AC_val.append(AC_val)

    Y_val = np.concatenate(all_Y_label)
    Y_pred = np.concatenate(all_Y_pred)
    AC_val = np.concatenate(all_AC_val)
    cross_val_results = np.concatenate([AC_val, Y_val, Y_pred], axis=1)

    integrate = save_logits(cross_val_results, save_path, "DeepDSI", "UB2_crossval.csv", "UB2_DeepDSI_crossval.csv")
    perf_org, perf, perf_ub2 = evaluate_logits(Y_val, Y_pred, integrate, "DeepDSI")

    return perf_org, perf, perf_ub2


def ind_test(vgae_dict, features_tensor, adj_norm, X_train, Y_train, AC_test, X_test, Y_test, epochs, save_path):
    random_idx = list(range(X_train.shape[0]))
    np.random.seed(888)
    np.random.shuffle(random_idx)
    X_train = X_train[random_idx]
    Y_train = Y_train[random_idx]

    Y_pred, model, pro1_index, pro2_index = extract_logits(vgae_dict, features_tensor, adj_norm, X_train, Y_train, X_test, epochs)
    cross_val_results = np.concatenate([AC_test, Y_test, Y_pred], axis=1)

    integrate = save_logits(cross_val_results, save_path, "DeepDSI", "UB2_indtest.csv", "UB2_DeepDSI_indtest.csv")
    perf_org, perf, perf_ub2 = evaluate_logits(Y_test, Y_pred, integrate, "DeepDSI")


    return perf_org, perf, perf_ub2


if __name__ == "__main__":

    print("Importing data...")
    data_path = "../data/"
    uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")

    dataset_train, dataset_test = load_dataset(uniprot, data_path + "dataset/")
    AC_train = dataset_train[:, :2]
    X_train = dataset_train[:, 3:].astype(np.int64)
    Y_train = dataset_train[:, 2].astype(np.int64).reshape([-1, 1])
    AC_test = dataset_test[:, :2]
    X_test = dataset_test[:, 3:].astype(np.int64)
    Y_test = dataset_test[:, 2].astype(np.int64).reshape([-1, 1])
    X_all = np.concatenate([X_train, X_test])
    Y_all = np.concatenate([Y_train, Y_test])

    features, adj_norm, adj_label  = load_data(uniprot, data_path, is_CT = True)
    features = features.to(device=try_gpu())
    adj_norm = adj_norm.to(device=try_gpu())
    adj_label = adj_label.to(device=try_gpu())

    print("Train variational graph autoencoder")
    VGAE, _  = train_VGAE(features, adj_norm, adj_label, 100, 343)
    vgae_dict = VGAE.state_dict()
    del VGAE, _
    torch.cuda.empty_cache()

    perf_org, perf, perf_ub2 = cross_val(vgae_dict, features, adj_norm, AC_train, X_train, Y_train, epochs=100, save_path = "../results/roc/")
    print(perf_org)
    print(perf)
    print(perf_ub2)

    perf_org, perf, perf_ub2 = ind_test(vgae_dict, features, adj_norm, X_train, Y_train, AC_test, X_test, Y_test, epochs=100, save_path = "../results/roc/")
    print(perf_org)
    print(perf)
    print(perf_ub2)
