import os
import collections
import random
import numpy as np
import pandas as pd


class GSDloader():
    def __init__(self, data_path, dataset_path, imbalance_ratio):

        '''
        :param data_path: the directory where the raw data is stored
        :param dataset_path: the directory where the dataset is stored
        :param imbalance_ratio: negative/positive ratio
        '''
        uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")
        self.randpro = set(uniprot['Entry'])
        self.uniprot_dict = dict([(v, k) for k, v in uniprot["Entry"].iteritems()])

        self.dpipro = collections.defaultdict(set)

        self.file2dpi(dataset_path + "gsp_train.txt")
        self.train_num = {protein: len(self.dpipro[protein])*imbalance_ratio for protein in self.dpipro}

        self.file2dpi(dataset_path + "gsp_test.txt")
        self.all_num = {protein: len(self.dpipro[protein])*imbalance_ratio for protein in self.dpipro}

        # calculate sample size
        self.dpipro = collections.defaultdict(set)
        self.file2dpi(dataset_path + "BIOGRID_DUB.txt")
        self.file_del_dpi(dataset_path + "gsp_train.txt")
        self.file_del_dpi(dataset_path + "gsp_test.txt")

        self.dataset_path = dataset_path

    def file2dpi(self, filename):
        with open(filename) as known_file:
            known_file.readline()  # skip header
            for line in known_file:
                proteins = line.rstrip('\r\n').split('\t')
                self.dpipro[proteins[0]].add(proteins[1])

    def file_del_dpi(self, filename):
        with open(filename) as known_file:
            known_file.readline()
            for line in known_file:
                proteins = line.rstrip('\r\n').split('\t')
                self.dpipro[proteins[0]].discard(proteins[1])

    def padding_sample(self, gsn_train, gsn_test):
        gsp_train = pd.read_table(self.dataset_path + "gsp_train.txt", delimiter="\t")
        gsp_test = pd.read_table(self.dataset_path + "gsp_test.txt", delimiter="\t")

        gsp_train["source"] = gsp_train["source"].map(lambda x: 1)
        gsp_train["pro1_index"] = gsp_train["DUB"].map(self.uniprot_dict)
        gsp_train["pro2_index"] = gsp_train["SUB"].map(self.uniprot_dict)
        gsn_train["source"] = gsn_train["Pro1"].map(lambda x: 0)
        gsn_train["pro1_index"] = gsn_train["Pro1"].map(self.uniprot_dict)
        gsn_train["pro2_index"] = gsn_train["Pro2"].map(self.uniprot_dict)
        dataset_train = np.concatenate([np.array(gsp_train), np.array(gsn_train)])

        gsp_test["source"] = gsp_test["source"].map(lambda x: 1)
        gsp_test["pro1_index"] = gsp_test["DUB"].map(self.uniprot_dict)
        gsp_test["pro2_index"] = gsp_test["SUB"].map(self.uniprot_dict)
        gsn_test["source"] = gsn_test["Pro1"].map(lambda x: 0)
        gsn_test["pro1_index"] = gsn_test["Pro1"].map(self.uniprot_dict)
        gsn_test["pro2_index"] = gsn_test["Pro2"].map(self.uniprot_dict)
        dataset_test = np.concatenate([np.array(gsp_test), np.array(gsn_test)])

        return dataset_train, dataset_test


    def sampling_RWD(self):
        gsn_train_dub = list()
        gsn_train_pro = list()
        gsn_test_dub = list()
        gsn_test_pro = list()
        for dub in self.all_num.keys():
            GSD = self.dpipro[dub]
            if self.all_num[dub] > len(GSD):        # impute proteins with random proteins
                GSD = GSD | set(random.sample(self.randpro, self.all_num[dub] - len(GSD)))

            gsn_all = set(random.sample(GSD, self.all_num[dub]))
            if dub in self.train_num.keys():
                gsn_train = set(random.sample(gsn_all, self.train_num[dub]))
            else:
                gsn_train = set()
            gsn_test = gsn_all - gsn_train
            gsn_train_dub += [dub] * len(gsn_train)
            gsn_train_pro += list(gsn_train)
            gsn_test_dub += [dub] * len(gsn_test)
            gsn_test_pro += list(gsn_test)
        gsn_train = pd.DataFrame(zip(gsn_train_dub, gsn_train_pro), columns = ['Pro1', 'Pro2'])
        gsn_test = pd.DataFrame(zip(gsn_test_dub, gsn_test_pro), columns=['Pro1', 'Pro2'])

        self.dataset_train, self.dataset_test = self.padding_sample(gsn_train, gsn_test)
        return self.dataset_train, self.dataset_test

    def saving_dataset(self, save_path):
        # save as .CSV file
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        pd.DataFrame(self.dataset_train, columns=["DUB", "Pro", "label", "DUB_idx", "Pro_idx"]).to_csv(
            save_path + "dataset_train.csv", index=False)
        pd.DataFrame(self.dataset_test, columns=["DUB", "Pro", "label", "DUB_idx", "Pro_idx"]).to_csv(
            save_path + "dataset_test.csv", index=False)

