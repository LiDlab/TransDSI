#!/usr/bin/env python3 -u

import argparse
import numpy as np
import pandas as pd
import torch
from src.utils import try_gpu
from src.load import load_data
from src.model import DSIPredictor, PairExplainer

from explain.explain import attention2seq
from explain.explain_vis import get_average_score_per_triplet

def create_parser():
    parser = argparse.ArgumentParser(
        description="Investigate the regions of the input DUB and/or candidate SUB sequence that contribute the most to the DeepDSI score"
    )
    parser.add_argument(
        "-obj", "--feat_mask_obj",
        type=str,
        default="dsi",
        choices= ["dsi", "dub", "sub"],
        help="The object of feature mask that will be learned (default='dsi')",
    )
    parser.add_argument(
        "-d", "--dub",
        type=str,
        required = True,
        help="Uniprot ID of the queried DUB",
    )
    parser.add_argument(
        "-s", "--candidate_sub",
        type=str,
        required = True,
        help="Uniprot ID of the candidate substrate corresponding to the queried DUB",
    )

    parser.add_argument(
        "-in", "--model_location",
        type=str,
        default="results/model/",
        help="DSIPredictor model file location",
    )
    parser.add_argument(
        "-out", "--output_location",
        type=str,
        default="results/importance/",
        help="PairExplainer output file location",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate to train PairExplainer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of epochs to train PairExplainer",
    )
    parser.add_argument(
        "--log",
        type=bool,
        default=True,
        help="Whether or not to print the learning progress of PairExplainer",
    )

    parser.add_argument("--nogpu", action="store_false", help="Do not use GPU even if available")
    return parser

def avg_attention(sequence, attention):
    if attention.max() > 0.05:
        attention = attention / attention.max()
    attention = attention2seq(sequence, attention).reshape([-1, 1])
    avg_att = get_average_score_per_triplet(attention)
    avg_att = list(np.round(np.array(avg_att), 2))
    return avg_att

def main(args):
    data_path = "data/"
    uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")
    uniprot_dict = dict([(v, k) for k, v in uniprot["Entry"].iteritems()])

    if (args.dub not in list(uniprot_dict.keys())) or (args.candidate_sub not in list(uniprot_dict.keys())):
        print("ERROR: The input DUB and candidate SUB have errors with the Uniprot ID  OR  The input DUB or candidate SUB is not human reviewed protein in the Uniprot database")
        return
    else:
        dub_idx = uniprot_dict[args.dub]
        sub_idx = uniprot_dict[args.candidate_sub]
    features, adj_norm, adj_label = load_data(uniprot, data_path, is_CT=True)

    model = DSIPredictor(686, 1)
    model_dict = torch.load(args.model_location + "DSIPredictor.pth")
    model.load_state_dict(model_dict)

    if torch.cuda.is_available() and args.nogpu:
        model = model.to(device=try_gpu())
        features = features.to(device=try_gpu())
        adj_norm = adj_norm.to(device=try_gpu())
        print("Transferred model and data to GPU")

    explainer = PairExplainer(model, lr = args.lr, feat_mask_obj = args.feat_mask_obj, log = args.log)

    # Explain node
    node_feat_mask = explainer.explain(features, adj_norm, dub_idx, sub_idx, epochs=args.epochs)

    sequence_dub = uniprot.loc[uniprot["Entry"] == args.dub, "Sequence"].values[0]
    sequence_sub = uniprot.loc[uniprot["Entry"] == args.candidate_sub, "Sequence"].values[0]

    if args.feat_mask_obj == "dsi":
        attention_dub = node_feat_mask[0].numpy()
        avg_att_dub = avg_attention(sequence_dub, attention_dub)
        attention_sub = node_feat_mask[1].numpy()
        avg_att_sub = avg_attention(sequence_sub, attention_sub)
        df = pd.DataFrame(data=[list(sequence_dub), avg_att_dub, list(sequence_sub), avg_att_sub],
                          index=["Amino acid (DUB)", "Importance (DUB)", "Amino acid (SUB)", "Importance (SUB)"])

    elif args.feat_mask_obj == "dub":
        attention_dub = node_feat_mask[0].numpy()
        avg_att_dub = avg_attention(sequence_dub, attention_dub)
        df = pd.DataFrame(data=[list(sequence_dub), avg_att_dub],
                          index=["Amino acid (DUB)", "Importance (DUB)"])

    else:
        attention_sub = node_feat_mask[0].numpy()
        avg_att_sub = avg_attention(sequence_sub, attention_sub)
        df = pd.DataFrame(data=[list(sequence_sub), avg_att_sub],
                          index=["Amino acid (SUB)", "Importance (SUB)"])

    path = args.output_location + args.dub + "_" + args.candidate_sub + ".csv"
    df.T.to_csv(path, index = False)

    print("The explainable result of " + args.dub + " and " + args.candidate_sub + " is saved in '" + path + "'.")



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
