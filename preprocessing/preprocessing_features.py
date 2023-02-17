import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# def find_amino_acid(x):
#     return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)
def write_fasta(database, save_path):
    filename = save_path + "uniprot_seq.fas"
    with open(filename, "w") as f:
        for i,row in database.iterrows():
            f.write(">" + str(i) + "\n")
            f.write(row['Sequence'] + "\n")

def CT(sequence):
    classMap = {'G': '1', 'A': '1', 'V': '1', 'L': '2', 'I': '2', 'F': '2', 'P': '2',
                'Y': '3', 'M': '3', 'T': '3', 'S': '3', 'H': '4', 'N': '4', 'Q': '4', 'W': '4',
                'R': '5', 'K': '5', 'D': '6', 'E': '6', 'C': '7',
                'B': '8', 'O': '8', 'J': '8', 'U': '8', 'X': '8', 'Z': '8'}
    seq = ''.join([classMap[x] for x in sequence])
    length = len(seq)
    coding = np.zeros(343, dtype=int)
    for i in range(length - 2):
        if(int(seq[i])==8 or int(seq[i+1])==8 or int(seq[i+2])==8):
            continue
        index = int(seq[i]) + (int(seq[i + 1]) - 1) * 7 + (int(seq[i + 2]) - 1) * 49 - 1
        coding[index] = coding[index] + 1

    if sum(coding) == 0:
        coding[0] = 1
    # coding[coding > 20] = 20

    return coding

print("Start processing data from the Uniprot database...")
print("Importing data...")
data_path = "../data/"
uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")
# uniprot = uniprot.loc[~uniprot['Sequence'].apply(find_amino_acid)]
# uniprot.reset_index(drop = True, inplace = True)
uniprot.loc[:,'features_seq'] = uniprot['Sequence'].apply(CT)

print("Collect all CT embeddings")
features = h5py.File(data_path + "DeepDSI_features.hdf5", "w")
for i in tqdm(range(len(uniprot))):
    group = features.create_group(uniprot.loc[i, "Entry"])
    group.create_dataset("E", data=uniprot.loc[i, "features_seq"])
    dt = h5py.special_dtype(vlen=str)
    group.create_dataset("S", dtype=dt, data=uniprot.loc[i, "Sequence"])
features.close()

# print("Start generating FAS files for building local BLAST")
write_fasta(uniprot, save_path = data_path + "blast/")

print("The end")