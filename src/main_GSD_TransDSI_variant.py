import numpy as np
import pandas as pd
import torch

from main_GSD import cross_val, ind_test
from main_GSD_ML import cross_val_ML, ind_test_ML
from train import train_VGAE
from train_baseline import train_eval_MLP, train_eval_AE
from load import load_data, load_GSD
from utils import save_logits_with_baseline, evaluate_logits_with_baseline, try_gpu


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
    features = features.to(device=try_gpu())
    adj_norm = adj_norm.to(device=try_gpu())
    adj_label = adj_label.to(device=try_gpu())


    # TransDSI (w/o SSN)
    cross_val_results = cross_val_ML(features, AC_train, X_train, Y_train, train_eval_MLP, 500)
    integrate = save_logits_with_baseline(cross_val_results, "../results/performance/GSD/", "TransDSIwoSSN",
                                          "UB2_TransDSI_crossval.csv",
                                          "UB2_TransDSIwoSSN_crossval.csv")
    perf_org, perf, perf_ub2 = evaluate_logits_with_baseline(cross_val_results[:, 2].reshape(-1, 1).astype(np.float32),
                                                             cross_val_results[:, 3].reshape(-1, 1).astype(np.float32),
                                                             integrate, "TransDSIwoSSN")
    print("{}\n{}\n{}".format(perf_org, perf, perf_ub2))

    ind_test_results = ind_test_ML(features, X_train, Y_train, AC_test, X_test, Y_test, train_eval_MLP, 100)
    integrate = save_logits_with_baseline(ind_test_results, "../results/performance/GSD/", "TransDSIwoSSN",
                                          "UB2_TransDSI_indtest.csv",
                                          "UB2_TransDSIwoSSN_indtest.csv")
    perf_org, perf, perf_ub2 = evaluate_logits_with_baseline(ind_test_results[:, 2].reshape(-1, 1).astype(np.float32),
                                                             ind_test_results[:, 3].reshape(-1, 1).astype(np.float32),
                                                             integrate, "TransDSIwoSSN")
    print("{}\n{}\n{}".format(perf_org, perf, perf_ub2))


    # TransDSI (w/o CT)
    features = torch.eye(adj_norm.size(0)).to(adj_norm.device)
    features = train_eval_AE(features)

    print("Train variational graph autoencoder")
    VGAE, _ = train_VGAE(features, adj_norm, adj_label, 100, 343)
    vgae_dict = VGAE.state_dict()
    del VGAE, _
    torch.cuda.empty_cache()

    cross_val_results = cross_val(vgae_dict, features, adj_norm, AC_train, X_train, Y_train, epochs=100)
    integrate = save_logits_with_baseline(cross_val_results, "../results/performance/GSD/", "TransDSIwoCT",
                                          "UB2_TransDSIwoSSN_crossval.csv",
                                          "UB2_TransDSI_variant_crossval.csv")
    perf_org, perf, perf_ub2 = evaluate_logits_with_baseline(cross_val_results[:, 2].reshape(-1, 1).astype(np.float32),
                                                             cross_val_results[:, 3].reshape(-1, 1).astype(np.float32),
                                                             integrate, "TransDSIwoCT")
    print("{}\n{}\n{}".format(perf_org, perf, perf_ub2))

    ind_test_results = ind_test(vgae_dict, features, adj_norm, X_train, Y_train, AC_test, X_test, Y_test, epochs=100)
    integrate = save_logits_with_baseline(ind_test_results, "../results/performance/GSD/", "TransDSIwoCT",
                                          "UB2_TransDSIwoSSN_indtest.csv",
                                          "UB2_TransDSI_variant_indtest.csv")
    perf_org, perf, perf_ub2 = evaluate_logits_with_baseline(ind_test_results[:, 2].reshape(-1, 1).astype(np.float32),
                                                             ind_test_results[:, 3].reshape(-1, 1).astype(np.float32),
                                                             integrate, "TransDSIwoCT")
    print("{}\n{}\n{}".format(perf_org, perf, perf_ub2))

