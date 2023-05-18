import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("..")
from src.utils import try_gpu
from src.load import load_data
from src.model import DSIPredictor, PairExplainer

def attention2seq(sequence, node_feat_mask):
    classMap = {'G': '1', 'A': '1', 'V': '1', 'L': '2', 'I': '2', 'F': '2', 'P': '2',
                'Y': '3', 'M': '3', 'T': '3', 'S': '3', 'H': '4', 'N': '4', 'Q': '4', 'W': '4',
                'R': '5', 'K': '5', 'D': '6', 'E': '6', 'C': '7',
                'B': '8', 'O': '8', 'J': '8', 'U': '8', 'X': '8', 'Z': '8'}
    seq = ''.join([classMap[x] for x in sequence])
    length = len(seq)
    coding = np.zeros(length - 2)
    for i in range(length - 2):
        if (int(seq[i]) == 8 or int(seq[i + 1]) == 8 or int(seq[i + 2]) == 8):
            continue
        index = int(seq[i]) + (int(seq[i + 1]) - 1) * 7 + (int(seq[i + 2]) - 1) * 49 - 1
        coding[i] = node_feat_mask[index].item()
    return coding


if __name__ == "__main__":

    print("Importing data...")
    data_path = "../data/"
    uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")
    uniprot_dict = dict([(v, k) for k, v in uniprot["Entry"].iteritems()])

    features, adj_norm, adj_label = load_data(uniprot, data_path, is_CT=True)
    features = features.to(device=try_gpu())
    adj_norm = adj_norm.to(device=try_gpu())

    model = DSIPredictor(686, 1).to(device=try_gpu())

    results_path = "../results/"
    model_path = results_path + "model/"
    model_dict = torch.load(model_path + "DSIPredictor.pth")
    model.load_state_dict(model_dict)

    node1 = "Q93009"
    node2 = "P26358"

    node1_idx = uniprot_dict[node1]
    node2_idx = uniprot_dict[node2]

    # Initialize explainer
    explainer = PairExplainer(model, num_hops=2, feat_mask_obj = 'dsi')

    # Explain node
    node_feat_mask = explainer.explain(features, adj_norm, node1_idx, node2_idx, epochs=10000)

    sub_ct_attention = node_feat_mask[1].numpy()
    sub_ct_attention = sub_ct_attention / sub_ct_attention.max()

    sequence2 = uniprot.loc[uniprot["Entry"] == node2, "Sequence"].values[0]
    attention2 = attention2seq(sequence2, sub_ct_attention)

    save_path = results_path + "importance/"
    np.savetxt(save_path + node2 + ".csv", attention2, delimiter=",")
