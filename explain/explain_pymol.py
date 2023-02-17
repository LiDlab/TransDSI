import pandas as pd
from explain_vis import get_average_score_per_triplet

sub = "P26358"
save_path = "../results/importance/"

sub_imp = pd.read_csv(save_path + sub + ".csv", names = [sub], delimiter=",")
sub_imp = sub_imp.to_numpy()

# The average score per triplet
avg_imp = get_average_score_per_triplet(sub_imp)

PDB_ID = "4yoc"
structure_file = open(save_path + PDB_ID + '.pdb', "r")

all_line = list()
for line in structure_file:
    if line.startswith("ATOM"):
        chain = line[21:22].strip()
        idx = int(line[22:26].strip())

        if chain == 'A':
        #修改预测值
            line = list(line)
            line[60:66] = list("{:,.2f}".format(avg_imp[idx - 1]).rjust(6))
            line = "".join(line)
    all_line.append(line)

new_file = open(save_path + "new_" + PDB_ID + ".pdb", "w")
for i in all_line:
    new_file.write(i)

new_file.close()


