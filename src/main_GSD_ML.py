import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from train_baseline import train_eval_RF, train_eval_SVM, train_eval_XGBoost, train_eval_KNN, train_eval_LR
from load import load_data, load_GSD
from utils import save_logits_with_baseline, evaluate_logits_with_baseline, evaluate_logits, evaluate_logits_all

def cross_val_ML(features_tensor, AC, X, Y, model_train_fn, config, model_save_name = None, log = False):

    random_idx = list(range(AC.shape[0]))
    np.random.seed(888)
    np.random.shuffle(random_idx)
    AC = AC[random_idx]
    X = X[random_idx]
    Y = Y[random_idx]

    all_idx = list(range(AC.shape[0]))
    rs = KFold(n_splits=5, shuffle=True)
    cv_index_set = rs.split(all_idx)  # Five-fold cross validation and independent testing

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

        if log:
            print("###################################")
            print("Training on Fold {} of 5".format(fold + 1))
            fold = fold + 1

        Y_pred, model, pro1_index, pro2_index= model_train_fn(features_tensor, X_train, Y_train, X_val, config)

        all_Y_label.append(Y_val)
        all_Y_pred.append(Y_pred)
        all_AC_val.append(AC_val)

    Y_val = np.concatenate(all_Y_label)
    Y_pred = np.concatenate(all_Y_pred)
    AC_val = np.concatenate(all_AC_val)
    cross_val_results = np.concatenate([AC_val, Y_val, Y_pred], axis=1)

    # eval
    if model_save_name:
        perf = evaluate_logits_all(cross_val_results[:, 2].reshape(-1, 1).astype(np.float32), cross_val_results[:, 3].reshape(-1, 1).astype(np.float32))
        print(model_save_name + ':\t' + str(perf))

        # save
        cross_val_df = pd.DataFrame(cross_val_results, columns = ['DUB', 'Pro', 'label', 'prob'])
        cross_val_df.to_csv(model_save_name, index = False)

    return cross_val_results


def ind_test_ML(features_tensor, X_train, Y_train, AC_test, X_test, Y_test, model_train_fn, config, model_save_name = None, log = False):
    random_idx = list(range(X_train.shape[0]))
    np.random.seed(888)
    np.random.shuffle(random_idx)
    X_train = X_train[random_idx]
    Y_train = Y_train[random_idx]

    if log:
        print("###################################")
        print("Independent Testing")

    Y_pred, model, pro1_index, pro2_index = model_train_fn(features_tensor, X_train, Y_train, X_test, config)
    ind_test_results = np.concatenate([AC_test, Y_test, Y_pred], axis=1)

    # eval
    if model_save_name:
        perf = evaluate_logits_all(ind_test_results[:, 2].reshape(-1, 1).astype(np.float32),
                               ind_test_results[:, 3].reshape(-1, 1).astype(np.float32))
        print(model_save_name + ':\t' + str(perf))
        ind_test_df = pd.DataFrame(ind_test_results, columns=['DUB', 'Pro', 'label', 'prob'])
        ind_test_df.to_csv(model_save_name, index=False)

    return ind_test_results


def process_task(task):
    ML, features, X_train, Y_train, AC, X_test, Y_test, function_name, file1, file2 = task

    if function_name == 'cross_val_ML':
        out = cross_val_ML(features, AC, X_train, Y_train, MODEL_DICT[ML][0], MODEL_DICT[ML][1])
    elif function_name == 'ind_test_ML':
        out = ind_test_ML(features, X_train, Y_train, AC, X_test, Y_test, MODEL_DICT[ML][0], MODEL_DICT[ML][1])
    else:
        raise ValueError("Invalid function name")

    integrate = save_logits_with_baseline(out, "../results/performance/GSD/", ML, file1, file2)



if __name__ == "__main__":

    # load data
    print("Importing human protein sequences...", end = "")
    data_path = "../data/"
    dataset_path = "../data/dataset/"
    uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")
    print(" Done.")

    dataset_train, dataset_test = load_GSD(uniprot, dataset_path)
    AC_train = dataset_train[:, :2]
    X_train = dataset_train[:, 3:].astype(np.int64)
    Y_train = dataset_train[:, 2].astype(np.int64).reshape([-1, 1])
    AC_test = dataset_test[:, :2]
    X_test = dataset_test[:, 3:].astype(np.int64)
    Y_test = dataset_test[:, 2].astype(np.int64).reshape([-1, 1])
    X_all = np.concatenate([X_train, X_test])
    Y_all = np.concatenate([Y_train, Y_test])

    features, adj_norm, adj_label = load_data(uniprot, data_path, is_CT = True)

    MODEL_DICT = {
        'RF': (train_eval_RF, {'min_samples_leaf': 15}),
        'SVM': (train_eval_SVM, {'C': 100}),
        'XGBoost': (train_eval_XGBoost, {'learning_rate': 0.001, 'eval_metric': 'logloss', 'use_label_encoder' :False}),
        'KNN': (train_eval_KNN, {'n_neighbors': 13, 'weights': 'distance'}),
        'LR': (train_eval_LR, {'C': 100, 'max_iter': 5000})
    }

    tasks = [
        ('RF', features, X_train, Y_train, AC_train, None, None, 'cross_val_ML', "UB2_TransDSI_variant_crossval.csv",
         "UB2_TransDSI_RF_crossval.csv"),
        ('XGBoost', features, X_train, Y_train, AC_train, None, None, 'cross_val_ML', "UB2_TransDSI_RF_crossval.csv",
         "UB2_TransDSI_XGBoost_crossval.csv"),
        ('SVM', features, X_train, Y_train, AC_train, None, None, 'cross_val_ML', "UB2_TransDSI_XGBoost_crossval.csv",
         "UB2_TransDSI_SVM_crossval.csv"),
        ('KNN', features, X_train, Y_train, AC_train, None, None, 'cross_val_ML', "UB2_TransDSI_SVM_crossval.csv",
         "UB2_TransDSI_KNN_crossval.csv"),
        ('LR', features, X_train, Y_train, AC_train, None, None, 'cross_val_ML', "UB2_TransDSI_KNN_crossval.csv",
         "GSD_crossval_prob.csv"),

        ('RF', features, X_train, Y_train, AC_test, X_test, Y_test, 'ind_test_ML', "UB2_TransDSI_variant_indtest.csv",
         "UB2_TransDSI_RF_indtest.csv"),
        ('XGBoost', features, X_train, Y_train, AC_test, X_test, Y_test, 'ind_test_ML', "UB2_TransDSI_RF_indtest.csv",
         "UB2_TransDSI_XGBoost_indtest.csv"),
        ('SVM', features, X_train, Y_train, AC_test, X_test, Y_test, 'ind_test_ML', "UB2_TransDSI_XGBoost_indtest.csv",
         "UB2_TransDSI_SVM_indtest.csv"),
        ('KNN', features, X_train, Y_train, AC_test, X_test, Y_test, 'ind_test_ML', "UB2_TransDSI_SVM_indtest.csv",
         "UB2_TransDSI_KNN_indtest.csv"),
        ('LR', features, X_train, Y_train, AC_test, X_test, Y_test, 'ind_test_ML', "UB2_TransDSI_KNN_indtest.csv",
         "GSD_indtest_prob.csv")
    ]

    for task in tasks:
        process_task(task)
