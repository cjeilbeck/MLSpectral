from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
import pandas as pd


Comparison = 'classifier' #  'model','problem', 'classifier'

#problem
SVM = False
RF = False
MLP = False

#model
INDEX = False
HYP = False
CUSTOM = False
OAK = True
BINARY = False

if Comparison == 'model':
    

    if INDEX:
        rf = pd.read_csv('PREDICTIONS_(INDEX)/rf_predictions.csv')
        svm = pd.read_csv('PREDICTIONS_(INDEX)/svm_predictions.csv')
        mlp = pd.read_csv('PREDICTIONS_(INDEX)/mlp_predictions.csv')
    elif HYP:
        rf = pd.read_csv('PREDICTIONS_(HYP)/rf_predictionsHYP.csv')
        svm = pd.read_csv('PREDICTIONS_(HYP)/svm_predictionsHYP.csv')
        mlp = pd.read_csv('PREDICTIONS_(HYP)/mlp_predictionsHYP.csv')

    elif CUSTOM:
        rf = pd.read_csv('PREDICTIONS_(CUSTOM)/rf_predictionsCUSTOM.csv')
        svm = pd.read_csv('PREDICTIONS_(CUSTOM)/svm_predictionsCUSTOM.csv')
        mlp = pd.read_csv('PREDICTIONS_(CUSTOM)/mlp_predictionsCUSTOM.csv')
    elif OAK:
        rf = pd.read_csv('PREDICTIONS_(OAK)/rf_predictionsOAK.csv')
        svm = pd.read_csv('PREDICTIONS_(OAK)/svm_predictionsOAK.csv')
        mlp = pd.read_csv('PREDICTIONS_(OAK)/mlp_predictionsOAK.csv')
    elif BINARY:
        rf = pd.read_csv('PREDICTIONS_(BINARY)/rf_predictionsBINARY.csv')
        svm = pd.read_csv('PREDICTIONS_(BINARY)/svm_predictionsBINARY.csv')
        mlp = pd.read_csv('PREDICTIONS_(BINARY)/mlp_predictionsBINARY.csv')



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

elif Comparison == 'problem':

    if SVM:
        custom = pd.read_csv('PREDICTIONS_(CUSTOM)/svm_predictionsCUSTOM.csv')
        indices = pd.read_csv('PREDICTIONS_(INDEX)/svm_predictions.csv')
        hyp = pd.read_csv('PREDICTIONS_(HYP)/svm_predictionsHYP.csv')
        pred_col = 'y_pred_svm'

    elif RF:
        custom = pd.read_csv('PREDICTIONS_(CUSTOM)/rf_predictionsCUSTOM.csv')
        indices = pd.read_csv('PREDICTIONS_(INDEX)/rf_predictions.csv')
        hyp = pd.read_csv('PREDICTIONS_(HYP)/rf_predictionsHYP.csv')
        pred_col = 'y_pred_rf'

    elif MLP:
        custom = pd.read_csv('PREDICTIONS_(CUSTOM)/mlp_predictionsCUSTOM.csv')
        indices = pd.read_csv('PREDICTIONS_(INDEX)/mlp_predictions.csv')
        hyp = pd.read_csv('PREDICTIONS_(HYP)/mlp_predictionsHYP.csv')
        pred_col = 'y_pred_mlp'

    y_true = custom['y_true']
    custom_correct = y_true == custom[pred_col]
    index_correct = y_true == indices[pred_col]
    hyp_correct = y_true == hyp[pred_col]

    correct = pd.DataFrame({'custom':custom_correct,'index':index_correct,'hyp':hyp_correct})

    resultQ=cochrans_q(correct)
    print('Cochran:')
    print(f"Q-statistic: {resultQ.statistic:.4f}")
    print(f"p-value: {resultQ.pvalue:.4f}")

    # custom vs index
    ct = pd.crosstab(custom_correct, index_correct)
    res = mcnemar(ct, exact=True)
    print(f"CUSTOM vs INDEX:\n{ct}")
    print(f"statistic: {res.statistic}, p-value: {res.pvalue}")

    # custom vs hyp
    ct = pd.crosstab(custom_correct, hyp_correct)
    res = mcnemar(ct, exact=True)
    print(f"CUSTOM vs HYP:\n{ct}")
    print(f"statistic: {res.statistic}, p-value: {res.pvalue}")

    # index vs hyp
    ct = pd.crosstab(index_correct, hyp_correct)
    res = mcnemar(ct, exact=True)
    print(f"INDEX vs HYP:\n{ct}")
    print(f"statistic: {res.statistic}, p-value: {res.pvalue}")

elif Comparison == 'classifier':

    if OAK:
        rf  = pd.read_csv('PREDICTIONS_(OAK)/rf_predictionsOAK.csv')
        svm = pd.read_csv('PREDICTIONS_(OAK)/svm_predictionsOAK.csv')
        mlp = pd.read_csv('PREDICTIONS_(OAK)/mlp_predictionsOAK.csv')
        clf = pd.read_csv('PREDICTIONS_CLASSIFIER/classifier_oak.csv')

    elif BINARY:
        rf  = pd.read_csv('PREDICTIONS_(BINARY)/rf_predictionsBINARY.csv')
        svm = pd.read_csv('PREDICTIONS_(BINARY)/svm_predictionsBINARY.csv')
        mlp = pd.read_csv('PREDICTIONS_(BINARY)/mlp_predictionsBINARY.csv')
        clf = pd.read_csv('PREDICTIONS_CLASSIFIER/classifier_gum.csv')



    all = pd.DataFrame({
        'y_true':     rf['y_true'],
        'y_pred_rf':  rf['y_pred_rf'],
        'y_pred_svm': svm['y_pred_svm'],
        'y_pred_mlp': mlp['y_pred_mlp'],
        'y_pred_clf': clf['y_pred_class'],
    })

    svm_correct = all['y_true'] == all['y_pred_svm']
    rf_correct  = all['y_true'] == all['y_pred_rf']
    mlp_correct = all['y_true'] == all['y_pred_mlp']
    clf_correct = all['y_true'] == all['y_pred_clf']

    correct = pd.DataFrame({
        'svm': svm_correct,
        'rf':  rf_correct,
        'mlp': mlp_correct,
        'clf': clf_correct,
    })

    resultQ = cochrans_q(correct)
    print('Cochran:')
    print(f"Q-statistic: {resultQ.statistic:.4f}")
    print(f"p-value: {resultQ.pvalue:.4f}")

    pairs = [
        ('SVM', svm_correct, 'RF',  rf_correct),
        ('SVM', svm_correct, 'MLP', mlp_correct),
        ('SVM', svm_correct, 'CLF', clf_correct),
        ('RF',  rf_correct,  'MLP', mlp_correct),
        ('RF',  rf_correct,  'CLF', clf_correct),
        ('MLP', mlp_correct, 'CLF', clf_correct),
    ]

    for name_a, a, name_b, b in pairs:
        ct = pd.crosstab(a, b)
        res = mcnemar(ct, exact=True)
        print(f"\n{name_a} vs {name_b}:")
        print(f"Contingency Table:\n{ct}")
        print(f"McNemar's test statistic: {res.statistic}")
        print(f"p-value: {res.pvalue}")


"""
SVM vs CLF:
McNemar's test statistic: 2.0
p-value: 0.03857421875

RF vs CLF:
McNemar's test statistic: 3.0
p-value: 0.5078125

MLP vs CLF:
McNemar's test statistic: 3.0
p-value: 0.5078125




SVM vs CLF:
McNemar's test statistic: 1.0
p-value: 0.01171875

RF vs CLF:
McNemar's test statistic: 0.0
p-value: 0.03125

MLP vs CLF:
McNemar's test statistic: 4.0
p-value: 0.3876953125"""