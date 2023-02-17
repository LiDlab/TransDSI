import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

def loss_function(preds, labels, mu, logstd, norm, pos_weight):
    # cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight =pos_weight)
    cost = norm * F.binary_cross_entropy(preds, labels)

    KLD = -0.5  * torch.mean(torch.sum(
        1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), 1))
    return cost + KLD


def save_logits(all_pred, path, item, filename, outname):
    pred = pd.DataFrame(all_pred, columns=["DUB", "Substrate", "label", item])
    pred.to_csv(path + item + "_" + filename.split("_")[-1], index = False)
    ub2 = pd.read_csv(path + filename, sep=",")
    intergrate = pd.merge(ub2, pred, on=["DUB", "Substrate", "label"])
    intergrate.to_csv(path + outname, index = False)

    return intergrate

def evaluate_logits(Y_val, Y_pred, intergrate, item):
    perf_org = dict()
    perf = dict()
    perf_ub2 = dict()

    perf_org["AUC"] = roc_auc_score(Y_val, Y_pred)
    perf_org["AUPR"] = average_precision_score(Y_val, Y_pred)

    perf["AUC"] = roc_auc_score(np.array(intergrate["label"],dtype=np.float32), np.array(intergrate[item],dtype=np.float32))
    perf["AUPR"] = average_precision_score(np.array(intergrate["label"],dtype=np.float32), np.array(intergrate[item],dtype=np.float32))

    perf_ub2["AUC"] = roc_auc_score(np.array(intergrate["label"], dtype=np.float32),
                                np.array(intergrate["Ubibrowser2"], dtype=np.float32))
    perf_ub2["AUPR"] = average_precision_score(np.array(intergrate["label"], dtype=np.float32),
                                           np.array(intergrate["Ubibrowser2"], dtype=np.float32))

    return perf_org, perf, perf_ub2

def precision_max_threshold(preds, labels):
    preds = np.round(preds, 3)
    labels = labels.astype(np.int32)
    p_max = 0
    threshold_spec = 0
    for t in range(1, 1000):
        threshold = t / 1000.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        precision = tp / (1.0 * (tp + fp))
        if p_max < precision:
            p_max = precision
            threshold_spec = threshold
    return threshold_spec,p_max

def f_max_threshold(preds, labels):
    preds = np.round(preds, 3)
    labels = labels.astype(np.int32)
    f_max = 0
    p_spec = 0
    r_spec = 0
    threshold_spec = 0
    for t in range(1, 1000):
        threshold = t / 1000.0
        predictions = (preds > threshold).astype(np.int32)
        p0 = (preds < threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        tn = np.sum(p0) - fn
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_spec = precision
            r_spec = recall
            threshold_spec = threshold
    return threshold_spec, f_max, p_spec, r_spec

def youden_index(preds, labels):
    preds = np.round(preds, 3)
    labels = labels.astype(np.int32)
    j_max = 0
    sn_spec = 0
    sp_spec = 0
    threshold_spec = 0
    for t in range(1, 1000):
        threshold = t / 1000.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        j = sn - fpr
        if j_max < j:
            j_max = j
            sn_spec = sn
            sp_spec = sp
            threshold_spec = threshold
    return threshold_spec, j_max, sn_spec, sp_spec
