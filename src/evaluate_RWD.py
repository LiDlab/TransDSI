import pandas as pd
import os
from utils import evaluate_logits_all

folder_path = "../results/performance/"
save_path = "../results/performance/RWD/"

cv_results = {}
it_results = {}

for ratio_folder in os.listdir(folder_path):
    if ratio_folder.startswith("RWD_ratio"):
        ratio_folder_path = os.path.join(folder_path, ratio_folder)
        for repeat_folder in os.listdir(ratio_folder_path):
            repeat_folder_path = os.path.join(ratio_folder_path, repeat_folder)

            for method_file in os.listdir(repeat_folder_path):
                if method_file.endswith("_cv.csv"):
                    method_name = method_file.split("_cv.csv")[0]
                    method_file_path = os.path.join(repeat_folder_path, method_file)
                    method_results = pd.read_csv(method_file_path)
                    eval_results_cv = evaluate_logits_all(method_results.iloc[:, 2], method_results.iloc[:, 3])
                    cv_results[f"{ratio_folder}_{repeat_folder}_{method_name}"] = pd.Series(eval_results_cv)

                    cv_results[f"{ratio_folder}_{repeat_folder}_{method_name}"]['ratio'] = int(ratio_folder.replace("RWD_ratio", ""))
                    cv_results[f"{ratio_folder}_{repeat_folder}_{method_name}"]['repeat'] = int(repeat_folder.replace("repeat", ""))
                    cv_results[f"{ratio_folder}_{repeat_folder}_{method_name}"]['method'] = method_name

                elif method_file.endswith("_it.csv"):
                    method_name = method_file.split("_it.csv")[0]
                    method_file_path = os.path.join(repeat_folder_path, method_file)
                    method_results = pd.read_csv(method_file_path)
                    eval_results_it = evaluate_logits_all(method_results.iloc[:, 2], method_results.iloc[:, 3])
                    it_results[f"{ratio_folder}_{repeat_folder}_{method_name}"] = pd.Series(eval_results_it)

                    it_results[f"{ratio_folder}_{repeat_folder}_{method_name}"]['ratio'] = int(ratio_folder.replace("RWD_ratio", ""))
                    it_results[f"{ratio_folder}_{repeat_folder}_{method_name}"]['repeat'] = int(repeat_folder.replace("repeat", ""))
                    it_results[f"{ratio_folder}_{repeat_folder}_{method_name}"]['method'] = method_name


cv_results = pd.DataFrame(cv_results).T
it_results = pd.DataFrame(it_results).T

# Sort
method_order = ["TransDSI", "RF", "XGBoost", "SVM", "LR", "KNN"]
cv_results['method'] = pd.Categorical(cv_results['method'], categories=method_order, ordered=True)
it_results['method'] = pd.Categorical(it_results['method'], categories=method_order, ordered=True)
cv_results = cv_results.sort_values(by=['ratio', 'repeat', 'method'], axis=0)
it_results = it_results.sort_values(by=['ratio', 'repeat', 'method'], axis=0)

cv_results.to_csv(save_path + "cv_results.csv", index = False)
it_results.to_csv(save_path + "it_results.csv", index = False)


cv_results = cv_results.drop(['repeat'], axis=1)
it_results = it_results.drop(['repeat'], axis=1)

grouped = cv_results.groupby(['ratio', 'method'])
means = grouped.mean().round(2)
stds = grouped.std().round(2)

cv_results_table = pd.DataFrame()
for col in ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-score']:
    # cv_results_table[col] = means[col].astype(str) + '±' + stds[col].astype(str)
    cv_results_table[col] = means[col].apply(lambda x: f"{x:.2f}") + '±' + stds[col].apply(lambda x: f"{x:.2f}")

cv_results_table = cv_results_table.T
methods = cv_results_table.index
cv_results_table.insert(0, "Ratio", methods)

cv_results_table.to_csv(save_path + "RWD_crossval.csv", index=False)


grouped = it_results.groupby(['ratio', 'method'])
means = grouped.mean().round(2)
stds = grouped.std().round(2)

it_results_table = pd.DataFrame()
for col in ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-score']:
    # it_results_table[col] = means[col].astype(str) + '±' + stds[col].astype(str)
    it_results_table[col] = means[col].apply(lambda x: f"{x:.2f}") + '±' + stds[col].apply(lambda x: f"{x:.2f}")

it_results_table = it_results_table.T
methods = it_results_table.index
it_results_table.insert(0, "Ratio", methods)

it_results_table.to_csv(save_path + "RWD_indtest.csv", index=False)