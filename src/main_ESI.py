import numpy as np
import pandas as pd

from main_GSD import ind_test
from train import train_VGAE
from load import load_data, load_ESI
from utils import evaluate_logits_all, try_gpu


if __name__ == "__main__":

    # load data
    print("Importing human protein sequences...", end = "")
    data_path = "../data/"
    dataset_path = "../results/performance/ESI/"
    uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")
    print(" Done.")

    dataset_train, dataset_test = load_ESI(uniprot, dataset_path)
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


    ind_test_results = ind_test(vgae_dict, features, adj_norm, X_train, Y_train, AC_test, X_test, Y_test, epochs=500)
    perf = evaluate_logits_all(ind_test_results[:, 2].reshape(-1, 1).astype(np.float32),
                               ind_test_results[:, 3].reshape(-1, 1).astype(np.float32))
    print(dataset_path + "TransESI_it.csv" + ':\t' + str(perf))
    ind_test_results = pd.DataFrame(ind_test_results, columns=['E3', 'Pro', 'label', 'prob'])
    ind_test_results.to_csv(dataset_path + "TransESI_it.csv", index=False)

