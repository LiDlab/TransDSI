import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
import torch
import h5py

def SparseTensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj(adj):

    rowsum = adj.sum(1)+1e-15
    r_inv = np.power(rowsum, -1).flatten()/2
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    adj = r_mat_inv.dot(adj)

    adj_normalized = np.eye(adj.shape[0]) - np.diag(adj.sum(1).flatten()) + adj
    adj_sparse = sp.coo_matrix(adj_normalized)
    return adj_sparse

def load_features(features_separate, uniprot, is_CT: bool = True):

    features_all = []
    for i in tqdm(range(len(uniprot))):
        AC = uniprot.loc[i]["Entry"]
        if is_CT:
            features_all.append(np.array(features_separate[AC + "/E"], dtype=np.float32).reshape([1,-1]))
        else:
            features_all.append(torch.from_numpy(np.array(features_separate[AC + "/E"], dtype=np.float32)).unsqueeze(0))

    features = np.concatenate(features_all, axis=0)
    features = features/20
    return torch.FloatTensor(features)

def load_adj(data, gene_num):

    data = data.values
    adj = np.zeros((gene_num, gene_num))
    for x in tqdm(range(len(data))):
        if data[x, 2] > adj[int(data[x, 0]), int(data[x, 1])]:
            adj[int(data[x, 0]), int(data[x, 1])] = data[x, 2]
    adj = np.nan_to_num(adj)

    adj = adj - np.eye(gene_num)
    adj[adj < 0] = 0

    adj_norm = preprocess_adj(adj)

    adj_label = adj + np.eye(gene_num)
    adj_label[adj_label > 0] = 1
    return SparseTensor(adj_norm), torch.FloatTensor(adj_label)

def load_data(uniprot, path, is_CT: bool = True):

    print("Collect embeddings")
    features_separate = h5py.File(path + "TransDSI_features.hdf5", "r")
    features = load_features(features_separate, uniprot, is_CT)

    print("Calculate the sequence similarity matrix")
    blast = pd.read_csv(path + "ssn.txt", sep="\t")
    blast.columns = ['protein1', 'protein2', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart',
                     'send', 'evalue', 'score']
    ssnnetwork = blast[['protein1', 'protein2', 'pident']].copy()
    ssnnetwork.loc[:, 'pident'] = ssnnetwork['pident'] / 100
    adj_norm, adj_label = load_adj(ssnnetwork, uniprot.shape[0])

    return features, adj_norm, adj_label

def load_dataset(uniprot, path):

    uniprot_dict = dict([(v, k) for k, v in uniprot["Entry"].iteritems()])

    gsp_train = pd.read_table(path + "gsp_train.txt", delimiter="\t")
    gsp_train["pro1_index"] = gsp_train["DUB"].map(uniprot_dict)
    gsp_train["pro2_index"] = gsp_train["SUB"].map(uniprot_dict)
    gsp_train["source"] = gsp_train["source"].map(lambda x: 1)
    gsp_train = np.array(gsp_train)
    gsn_train = pd.read_table(path + "gsn_train.txt", delimiter="\t")
    gsn_train["pro1_index"] = gsn_train["Pro1"].map(uniprot_dict)
    gsn_train["pro2_index"] = gsn_train["Pro2"].map(uniprot_dict)
    gsn_train["source"] = gsn_train["source"].map(lambda x: 0)
    gsn_train = np.array(gsn_train)

    dataset_train = np.concatenate([gsp_train, gsn_train])

    gsp_test = pd.read_table(path + "gsp_test.txt", delimiter="\t")
    gsp_test["pro1_index"] = gsp_test["DUB"].map(uniprot_dict)
    gsp_test["pro2_index"] = gsp_test["SUB"].map(uniprot_dict)
    gsp_test["source"] = gsp_test["source"].map(lambda x: 1)
    gsp_test = np.array(gsp_test)
    gsn_test = pd.read_table(path + "gsn_test.txt", delimiter="\t")
    gsn_test["pro1_index"] = gsn_test["Pro1"].map(uniprot_dict)
    gsn_test["pro2_index"] = gsn_test["Pro2"].map(uniprot_dict)
    gsn_test["source"] = gsn_test["source"].map(lambda x: 0)
    gsn_test = np.array(gsn_test)

    dataset_test = np.concatenate([gsp_test, gsn_test])

    return dataset_train, dataset_test
