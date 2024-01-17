import pandas as pd
from evaluate_GSD import evaluate_methods

# load data
print("Importing data...")
results_path = "../results/performance/ESI/"

models = ["TransESI_it", "RF_it", "SVM_it", "XGBoost_it", "LR_it", "KNN_it"]
all_results = pd.DataFrame()

for model in models:
    print(f"Reading {model} data...")
    results = pd.read_csv(results_path + f"{model}.csv")
    results_df = evaluate_methods(results)
    results_df.rename(index={'prob': model.split('_')[0]}, inplace=True)
    all_results = pd.concat([all_results, results_df], axis=0)  # 将当前模型的结果追加到汇总DataFrame中

# Sort and modify column headers
sorted_results = all_results.sort_values(by="AUROC", ascending=False)
sorted_results = sorted_results.reset_index()
sorted_results.rename(columns={"index": "Methods"}, inplace=True)
sorted_results.iloc[:, 1:] = sorted_results.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")
sorted_results = sorted_results[['Methods', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-score']]

sorted_results.to_csv(results_path + "ESI_indtest.csv", index = False)

# Save prob
first = True
all_results = pd.DataFrame()
for model in models:
    print(f"Reading {model} data...")
    results = pd.read_csv(results_path + f"{model}.csv")
    results.rename(columns={"prob": model.split('_')[0]}, inplace=True)
    if first:
        all_results = pd.concat([all_results, results], axis = 1)
        first = False
    else:
        all_results = pd.concat([all_results, results[model.split('_')[0]]], axis = 1)

all_results.to_csv(results_path + "ESI_indtest_prob.csv", index = False)
