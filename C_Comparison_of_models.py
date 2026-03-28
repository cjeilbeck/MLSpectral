from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
import pandas as pd

rf = pd.read_csv('PREDICTIONS_(INDEX)/rf_predictions.csv')
svm = pd.read_csv('PREDICTIONS_(INDEX)/svm_predictions.csv')
mlp = pd.read_csv('PREDICTIONS_(INDEX)/mlp_predictions.csv')

all = pd.DataFrame({
    'y_true': rf['y_true'],
    'y_pred_rf': rf['y_pred_rf'],
    'y_pred_svm': svm['y_pred_svm'],
    'y_pred_mlp': mlp['y_pred_mlp']
})

svm_correct = all['y_true'] == all['y_pred_svm']
rf_correct = all['y_true'] == all['y_pred_rf']
mlp_correct = all['y_true'] == all['y_pred_mlp']

correct = pd.DataFrame({'svm':svm_correct,'rf':rf_correct,'mlp':mlp_correct})

resultQ = cochrans_q(correct)

print('Cochran:')
print(f"Q-statistic: {resultQ.statistic:.4f}")
print(f"p-value: {resultQ.pvalue:.4f}")



#svm vs rf
contingency_svmrf = pd.crosstab(svm_correct, rf_correct)
result_svmrf = mcnemar(contingency_svmrf, exact=True)
print("SVM vs RF:")
print(f"Contingency Table:\n{contingency_svmrf}")
print(contingency_svmrf)
print(f"McNemar's test statistic: {result_svmrf.statistic}")
print(f"p-value: {result_svmrf.pvalue}")


#svm vs mlp
contingency_svmmlp = pd.crosstab(svm_correct, mlp_correct)
result_svmmlp = mcnemar(contingency_svmmlp, exact=True)
print("SVM vs MLP:")
print(f"Contingency Table:\n{contingency_svmmlp}")
print(f"McNemar's test statistic: {result_svmmlp.statistic}")
print(f"p-value: {result_svmmlp.pvalue}")

#rf vs mlp
contingency_rfmlp = pd.crosstab(rf_correct, mlp_correct)
result_rfmlp = mcnemar(contingency_rfmlp, exact=True)
print("RF vs MLP:")
print(f"Contingency Table:\n{contingency_rfmlp}")
print(f"McNemar's test statistic: {result_rfmlp.statistic}")
print(f"p-value: {result_rfmlp.pvalue}")

