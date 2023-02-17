import pandas as pd

def extract_intersection(path, dataset):
    inter_gsp = pd.read_table(path + "inter_gsp_" + dataset + ".txt", delimiter="\t")
    inter_gsn = pd.read_table(path + "inter_gsn_" + dataset + ".txt", delimiter="\t")
    inter_gsn[["DUB", "Substrate"]] = inter_gsn[["Pro1", "Pro2"]]
    inter_gsn = inter_gsn.drop(labels=["Pro1", "Pro2"], axis=1)
    inter = pd.concat([inter_gsp, inter_gsn])
    inter["label"] = inter["label"].apply(lambda x: 0 if x == -1 else 1)
    inter["Ubibrowser2"] = inter["integrate_lr"]
    inter["Ubibrowser2_domain_motif"] = inter["domain"] * inter["motif"]
    inter = inter[["DUB", "Substrate", "label", "Ubibrowser2", "Ubibrowser2_domain_motif"]]

    return inter


dataset_path = "../data/dataset/"

DSI_UB2_crossval = extract_intersection(dataset_path, "train")
DSI_UB2_indtest = extract_intersection(dataset_path, "test")

save_path = "../results/roc/"

DSI_UB2_crossval.to_csv(save_path + "UB2_crossval.csv", index = False)
DSI_UB2_indtest.to_csv(save_path + "UB2_indtest.csv", index = False)