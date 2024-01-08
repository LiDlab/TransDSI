#!/usr/bin/env python3 -u

import argparse
import numpy as np
import pandas as pd
import torch
from src.utils import try_gpu
from src.load import load_data
from src.model import DSIPredictor


def create_parser():
    parser = argparse.ArgumentParser(
        description="Obtain the prediction score of DSIPredictor on the query relation (DUB and candidate substrate)"
    )

    parser.add_argument(
        "-d", "--dub",
        type=str,
        default='Q14694',
        # required = True,
        help="Uniprot ID of the queried DUB",
    )
    parser.add_argument(
        "-s", "--candidate_sub",
        type=str,
        default='Q00987',
        # required=True,
        help="Uniprot ID of the candidate substrate corresponding to the queried DUB",
    )

    parser.add_argument(
        "-in", "--model_location",
        type=str,
        default="results/model/",
        help="DSIPredictor model file location",
    )

    parser.add_argument("--nogpu", action="store_false", help="Do not use GPU even if available")
    return parser


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

    model.eval()
    if torch.cuda.is_available() and args.nogpu:
        model = model.to(device=try_gpu())
        features = features.to(device=try_gpu())
        adj_norm = adj_norm.to(device=try_gpu())
        print("Transferred model and data to GPU")

    with torch.no_grad():
        pred = model(features, adj_norm, [dub_idx], [sub_idx])
        logits = np.log(pred.item() / (1 - pred.item()))
        scaled_pred = 1 / (1 + np.exp(-0.5 * logits))

    print("The TransDSI score of " + args.dub + " and " + args.candidate_sub + " is " + str(np.round(scaled_pred, 4)) + ".")



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
