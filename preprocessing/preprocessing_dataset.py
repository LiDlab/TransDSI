import pandas as pd
from tqdm import tqdm

# Generate gold standard positive | negative dataset
# gold standard positive dataset
print("Importing data...")
data_path = "../data/"
uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")

gsn = pd.read_csv(data_path + "BIOGRID-ALL-4.4.214.mitab.txt", sep="\t")
gsn = gsn[(gsn['Taxid Interactor A'] == "taxid:9606")&(gsn['Taxid Interactor B'] == "taxid:9606")].copy()
gsn.loc[:, "Pro1"] = gsn["Alt IDs Interactor A"].apply(lambda x : x.split("/swiss-prot:")[-1].split("|")[0])
gsn.loc[:, "Pro2"] = gsn["Alt IDs Interactor B"].apply(lambda x : x.split("/swiss-prot:")[-1].split("|")[0])
gsn.loc[:, "source"] = gsn["Publication Identifiers"].apply(lambda x: x.split(":")[-1])
gsn = gsn[["Pro1", "Pro2", "source"]].copy()
gsn = gsn[(gsn["Pro1"].isin(uniprot["Entry"].values))&(gsn["Pro2"].isin(uniprot["Entry"].values))].copy()
gsn_copy = gsn.copy()
gsn_copy.columns = ["Pro2", "Pro1", "source"]
gsn = pd.concat([gsn, gsn_copy])
gsn.sort_values(by="source", inplace = True)
gsn.drop_duplicates(subset=["Pro1", "Pro2"], keep="first", ignore_index=True, inplace = True)

ub2_path = data_path + "ubibrowser2/"
gsp = pd.read_csv(ub2_path + "literature.DUB.txt", sep="\t")
gsp = gsp[gsp["species"] == "H.sapiens"].copy()
gsp.rename(columns = dict(zip(["SwissProt AC (DUB)", "SwissProt AC (Substrate)", "SOURCEID"], ["DUB","SUB","source"])), inplace = True)
gsp = gsp[["DUB","SUB","source"]].copy()
gsp = gsp[(gsp["DUB"].isin(uniprot["Entry"].values))&(gsp["SUB"].isin(uniprot["Entry"].values))].copy()
gsp.reset_index(inplace = True, drop = True)

for i in range(len(gsp)):
    if "_HUMAN" in gsp.loc[i]["source"]:
        gsp.loc[i]["source"] = 11111111
    else:
        gsp.loc[i]["source"] = int(gsp.loc[i]["source"])

gsp.sort_values(by="source", inplace = True)
gsp.drop_duplicates(subset=["DUB","SUB"], keep="first", inplace = True, ignore_index=True)

gsn = gsn[gsn["Pro1"].isin(gsp["DUB"])].copy()

# The dataset used for 5-fold cross-validation comprised 616 DSIs that had been validated by literature up to June 2018. The independent test dataset was comprised of DSIs identified from June 2018 to August 2021.
gsp_train = gsp[gsp["source"] < 29803676].copy()
gsp_test = gsp[gsp["source"] >= 29803676].copy()
gsp_train.loc[gsp_train["source"] == 11111111, "source"] = "Uniprot"


UB2_train = pd.read_csv(ub2_path + "DUB_cross_validation.txt", sep="\t")
UB2_test = pd.read_csv(ub2_path + "DUB_independent_test.txt", sep="\t")
UB2_train = UB2_train[(UB2_train["DUB"].isin(uniprot["Entry"].values))&(UB2_train["Substrate"].isin(uniprot["Entry"].values))].copy()
UB2_test = UB2_test[(UB2_test["DUB"].isin(uniprot["Entry"].values))&(UB2_test["Substrate"].isin(uniprot["Entry"].values))].copy()

UB2_gsp_train = UB2_train[UB2_train["label"] == 1].copy()
UB2_gsn_train = UB2_train[UB2_train["label"] == -1].copy()
UB2_gsn_train.rename(columns = dict(zip(["DUB", "Substrate"], ["Pro1", "Pro2"])), inplace = True)
UB2_gsp_test = UB2_test[UB2_test["label"] == 1].copy()
UB2_gsn_test = UB2_test[UB2_test["label"] == -1].copy()
UB2_gsn_test.rename(columns = dict(zip(["DUB", "Substrate"], ["Pro1", "Pro2"])), inplace = True)

# cross_validation
new_gsp_train = list()
new_gsn_train = list()
inter_gsp_train = list()
inter_gsn_train = list()
train_dub = list(set(gsp_train["DUB"]))
for dub in tqdm(train_dub):
    our_gsp_train_sub = gsp_train[gsp_train["DUB"] == dub].copy()
    our_gsp_test_sub = gsp_test[gsp_test["DUB"] == dub].copy()
    UB2_gsp_train_sub = UB2_gsp_train[UB2_gsp_train["DUB"] == dub].copy()
    inter_gsp_train_sub = UB2_gsp_train_sub[UB2_gsp_train_sub["Substrate"].isin(our_gsp_train_sub["SUB"].values)].copy()

    new_gsp_train.append(our_gsp_train_sub)
    inter_gsp_train.append(inter_gsp_train_sub)

    UB2_gsn_train_sub = UB2_gsn_train[UB2_gsn_train["Pro1"] == dub].copy()
    UB2_gsn_test_sub = UB2_gsn_test[UB2_gsn_test["Pro2"] == dub].copy()

    if len(UB2_gsn_train_sub) >= len(our_gsp_train_sub):
        sample_UB2_gsn_train_sub = UB2_gsn_train_sub.sample(n = len(our_gsp_train_sub)).copy()
        inter_gsn_train.append(sample_UB2_gsn_train_sub)

        sample_UB2_gsn_train_sub.loc[:, "source"] = "Ubibrowser2"
        sample_UB2_gsn_train_sub = sample_UB2_gsn_train_sub[["Pro1", "Pro2", "source"]].copy()
        new_gsn_train.append(sample_UB2_gsn_train_sub)
    else:
        if len(UB2_gsn_train_sub) != 0:
            inter_gsn_train.append(UB2_gsn_train_sub)
            UB2_gsn_train_sub.loc[:, "source"] = "Ubibrowser2"
            UB2_gsn_train_sub = UB2_gsn_train_sub[["Pro1", "Pro2", "source"]].copy()
            new_gsn_train.append(UB2_gsn_train_sub)

        add_num = len(our_gsp_train_sub) - len(UB2_gsn_train_sub)
        gsn_sub = gsn[gsn["Pro1"] == dub]
        gsn_sub = gsn_sub[(~gsn_sub["Pro2"].isin(our_gsp_train_sub["SUB"].values)) & (
            ~gsn_sub["Pro2"].isin(our_gsp_test_sub["SUB"].values)) & (
                              ~gsn_sub["Pro2"].isin(UB2_gsn_train_sub["Pro2"].values)) & (
                              ~gsn_sub["Pro2"].isin(UB2_gsn_test_sub["Pro2"].values))]

        if add_num > len(gsn_sub):
            other_add_num = add_num - len(gsn_sub)
            new_gsn_train.append(gsn_sub)
            new_gsn_train.append(gsn.sample(other_add_num))
            print(add_num, len(gsn_sub))
        else:
            new_gsn_train.append(gsn_sub.sample(add_num))


new_gsp_train = pd.concat(new_gsp_train)
new_gsn_train = pd.concat(new_gsn_train)
inter_gsp_train = pd.concat(inter_gsp_train)
inter_gsn_train = pd.concat(inter_gsn_train)

# independent test
new_gsp_test = list()
new_gsn_test = list()
inter_gsp_test = list()
inter_gsn_test = list()
test_dub = list(set(gsp_test["DUB"]))
for dub in tqdm(test_dub):
    our_gsp_train_sub = gsp_train[gsp_train["DUB"] == dub].copy()
    our_gsp_test_sub = gsp_test[gsp_test["DUB"] == dub].copy()
    UB2_gsp_test_sub = UB2_gsp_test[UB2_gsp_test["DUB"] == dub].copy()
    inter_gsp_test_sub = UB2_gsp_test_sub[UB2_gsp_test_sub["Substrate"].isin(our_gsp_test_sub["SUB"].values)].copy()

    new_gsp_test.append(our_gsp_test_sub)
    inter_gsp_test.append(inter_gsp_test_sub)

    UB2_gsn_train_sub = UB2_gsn_train[UB2_gsn_train["Pro1"] == dub].copy()
    UB2_gsn_test_sub = UB2_gsn_test[UB2_gsn_test["Pro1"] == dub].copy()

    if len(UB2_gsn_test_sub) >= len(our_gsp_test_sub):
        sample_UB2_gsn_test_sub = UB2_gsn_test_sub.sample(n = len(our_gsp_test_sub)).copy()
        inter_gsn_test.append(sample_UB2_gsn_test_sub)

        sample_UB2_gsn_test_sub.loc[:, "source"] = "Ubibrowser2"
        sample_UB2_gsn_test_sub = sample_UB2_gsn_test_sub[["Pro1", "Pro2", "source"]].copy()
        new_gsn_test.append(sample_UB2_gsn_test_sub)
    else:
        if len(UB2_gsn_test_sub) != 0:
            inter_gsn_test.append(UB2_gsn_test_sub)
            UB2_gsn_test_sub.loc[:, "source"] = "Ubibrowser2"
            UB2_gsn_test_sub = UB2_gsn_test_sub[["Pro1", "Pro2", "source"]].copy()
            new_gsn_test.append(UB2_gsn_test_sub)

        add_num = len(our_gsp_test_sub) - len(UB2_gsn_test_sub)
        gsn_sub = gsn[gsn["Pro1"] == dub].copy()
        new_gsn_train_sub = new_gsn_train[new_gsn_train["Pro1"] == dub].copy()
        gsn_sub = gsn_sub[(~gsn_sub["Pro2"].isin(our_gsp_train_sub["SUB"].values)) & (
            ~gsn_sub["Pro2"].isin(our_gsp_test_sub["SUB"].values)) & (
                              ~gsn_sub["Pro2"].isin(UB2_gsn_train_sub["Pro2"].values)) & (
                              ~gsn_sub["Pro2"].isin(UB2_gsn_test_sub["Pro2"].values)) & (
                              ~gsn_sub["Pro2"].isin(new_gsn_train_sub["Pro2"].values))].copy()

        if add_num > len(gsn_sub):
            other_add_num = add_num - len(gsn_sub)
            new_gsn_test.append(gsn_sub)
            new_gsn_test.append(gsn.sample(other_add_num))
            print(add_num, len(gsn_sub))
        else:
            new_gsn_test.append(gsn_sub.sample(add_num))

new_gsp_test = pd.concat(new_gsp_test)
new_gsn_test = pd.concat(new_gsn_test)
inter_gsp_test = pd.concat(inter_gsp_test)
inter_gsn_test = pd.concat(inter_gsn_test)

# Get ubibrowser predictions
new_inter_gsp_test = list()
new_inter_gsn_test = list()

ub2_pred = pd.read_table(data_path + "ubibrowser2/" + "H.sapiens.result.txt")
ub2_pred = ub2_pred[(ub2_pred["enyz"].isin(list(gsp_test["DUB"].values) + list(new_gsn_test["Pro1"]))) & (ub2_pred['sub'].isin(list(gsp_test['SUB']) + list(new_gsn_test["Pro2"])))]

for i in tqdm(range(len(gsp_test))):
    i_gsp_test = gsp_test.iloc[i]
    i_inter_gsp_test = inter_gsp_test[(inter_gsp_test['DUB'] == i_gsp_test['DUB']) & (inter_gsp_test['Substrate'] == i_gsp_test['SUB'])]

    if len(i_inter_gsp_test):
        new_inter_gsp_test.append(i_inter_gsp_test)
    else:
        i_ub2_pred = ub2_pred[(ub2_pred['enyz'] == i_gsp_test['DUB']) & (ub2_pred['sub'] == i_gsp_test['SUB'])].copy()
        if len(i_ub2_pred):
            i_ub2_pred.loc[:,'label'] = 1
            i_ub2_pred = i_ub2_pred[['enyz','sub','label','motifLR','domainLR','goLR','netLR','interLR']].copy()
            i_ub2_pred.rename(columns = dict(zip(i_ub2_pred.columns, i_inter_gsp_test.columns)), inplace = True)
            new_inter_gsp_test.append(i_ub2_pred)

for i in tqdm(range(len(new_gsn_test))):
    i_gsn_test = new_gsn_test.iloc[i]
    i_inter_gsn_test = inter_gsn_test[(inter_gsn_test['Pro1'] == i_gsn_test['Pro1']) & (inter_gsn_test['Pro2'] == i_gsn_test['Pro2'])]

    if len(i_inter_gsn_test):
        new_inter_gsn_test.append(i_inter_gsn_test)
    else:
        i_ub2_pred = ub2_pred[(ub2_pred['enyz'] == i_gsn_test['Pro1']) & (ub2_pred['sub'] == i_gsn_test['Pro2'])].copy()
        if len(i_ub2_pred):
            i_ub2_pred.loc[:,'label'] = -1
            i_ub2_pred = i_ub2_pred[['enyz','sub','label','motifLR','domainLR','goLR','netLR','interLR']].copy()
            i_ub2_pred.rename(columns = dict(zip(i_ub2_pred.columns, i_inter_gsn_test.columns)), inplace = True)
            new_inter_gsn_test.append(i_ub2_pred)

new_inter_gsp_test = pd.concat(new_inter_gsp_test)
new_inter_gsn_test = pd.concat(new_inter_gsn_test)

save_path = data_path + "dataset/"

new_gsp_train.to_csv(save_path + "gsp_train.txt", index=False, header=True, sep="\t")
new_gsn_train.to_csv(save_path + "gsn_train.txt", index=False, header=True, sep="\t")
new_gsp_test.to_csv(save_path + "gsp_test.txt", index=False, header=True, sep="\t")
new_gsn_test.to_csv(save_path + "gsn_test.txt", index=False, header=True, sep="\t")
inter_gsp_train.to_csv(save_path + "inter_gsp_train.txt", index=False, header=True, sep="\t")
inter_gsn_train.to_csv(save_path + "inter_gsn_train.txt", index=False, header=True, sep="\t")
new_inter_gsp_test.to_csv(save_path + "inter_gsp_test.txt", index=False, header=True, sep="\t")
new_inter_gsn_test.to_csv(save_path + "inter_gsn_test.txt", index=False, header=True, sep="\t")
