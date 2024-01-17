import pandas as pd
from utils import evaluate_logits_all

def evaluate_methods(df):
    methods = df.columns[3:]    # Get the column names of evaluation methods
    results = pd.DataFrame(index=methods,
                           columns=["Sensitivity", "Specificity", "PPV", "NPV", "F1-score", "AUROC", "AUPRC"])
    Y_val = df["label"]

    for method in methods:
        Y_pred_prob = df[method]
        perf = evaluate_logits_all(Y_val, Y_pred_prob)
        results.loc[method] = perf

    return results


if __name__ == "__main__":

    # load data
    print("Importing data...")
    results_path = "../results/performance/GSD/"

    results = pd.read_csv(results_path + "GSD_crossval_prob.csv")
    results_df = evaluate_methods(results)
    # Sort and modify column headers
    sorted_results = results_df.sort_values(by="AUROC", ascending=False)
    sorted_results = sorted_results.reset_index()
    sorted_results.rename(columns={"index": "Methods"}, inplace=True)
    sorted_results.iloc[:, 1:] = sorted_results.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")
    sorted_results = sorted_results[['Methods', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-score']]

    sorted_results.to_csv(results_path + "GSD_crossval.csv", index = False)


    results = pd.read_csv(results_path + "GSD_indtest_prob.csv")
    results_df = evaluate_methods(results)
    # Sort and modify column headers
    sorted_results = results_df.sort_values(by="AUROC", ascending=False)
    sorted_results = sorted_results.reset_index()
    sorted_results.rename(columns={"index": "Methods"}, inplace=True)
    sorted_results.iloc[:, 1:] = sorted_results.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")
    sorted_results = sorted_results[['Methods', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-score']]

    sorted_results.to_csv(results_path + "GSD_indtest.csv", index = False)
