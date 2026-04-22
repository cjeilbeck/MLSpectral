import matplotlib.pyplot as plt
from pathlib import Path
import sys
cdir = Path(__file__).parent
root = cdir.parent
sys.path.append(str(root))
from FUNCTIONS.A_functions import read_data, indicesmav, indexplot1D, problemtype, KDEfunction, customlabels,indicescustomSVM,indicescustomrf,indicescustomMLP
from sklearn.model_selection import train_test_split, StratifiedKFold
plt.rcParams.update({'font.size': 12})
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, cohen_kappa_score

plt.rcParams.update({                
    'font.size': 9,           
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1,   
    'axes.linewidth': 0.5,    
    'xtick.direction': 'out',   
    'ytick.direction': 'out',
    'xtick.top': False,         
    'ytick.right': False   
})

filename = 'CSVfiles/datacalibrated.csv'

x, y = read_data(filename)
indices = indicesmav(x)
#prior controls prior weighting, indices only limits to indices rather than bands
INDICES_ONLY = False
TEST = False 
prior = False

if INDICES_ONLY:
    indices = indices[indices.columns[-5:]]

plot = indices.copy()
plot['leaf_type'] = y

BINARY = False
OAK = True
ALL = False
GUM = False
PINE = False

#F1 recall kind of dont work would not recommend
ACCURACY = True
F1 = False
RECALL = False

SINGLE = False 
N_FOLDS = 5

SAVEFIG = False
name = "classifier_gum.csv"

if BINARY:
    plot = problemtype(plot, 'binary')
    customweights = {'Gum': 7, 'Other': 8}
   
elif OAK:
    plot = problemtype(plot, 'oak')
    customweights = {'Oak': 4, 'Other': 15}
elif ALL:
    plot = problemtype(plot, 'all')
elif GUM:
    plot = problemtype(plot, 'gum')
    customweights = {'Young Gum': 1, 'Old Gum': 1}
elif PINE:
    plot = problemtype(plot, 'pine')
    customweights = {'Pine': 1, 'Other': 1}

#toggle to mess with priors 
customweights = None
train, test = train_test_split(plot, test_size=0.2, random_state=42, stratify=plot['leaf_type'])
cols = list(indices.columns) #for indicesonly automation

def get_metric(y_true, y_pred):
    if ACCURACY:
        return accuracy_score(y_true, y_pred)
    elif F1:
        return f1_score(y_true, y_pred, average='macro')
    elif RECALL:
        return recall_score(y_true, y_pred, average='macro')


def evalsingle(data, trainidx, validx, INDEXA):
    train_fold = data.loc[trainidx]
    val_fold = data.loc[validx]

    Athresholds, Alabels = KDEfunction(train_fold, INDEXA, prioradj=prior, priorarr = customweights)
    if not Athresholds:
        return None

    min_val, max_val = data[INDEXA].min(), data[INDEXA].max()
    bins = sorted(set([min_val] + Athresholds + [max_val]))
    if len(bins)-1 != len(Alabels):
        return None


    preds = pd.cut(val_fold[INDEXA], bins=bins, labels=Alabels, ordered=False, include_lowest=True).astype(str)
    return get_metric(val_fold['leaf_type'], preds)

def evaltwo(data, trainidx, validx, INDEXA):
    train_fold = data.loc[trainidx]
    val_fold = data.loc[validx]

    Athresholds, Alabels = KDEfunction(train_fold, INDEXA, prioradj=prior, priorarr=customweights)
    if not Athresholds:
        return None,None

    min_val, max_val = data[INDEXA].min(), data[INDEXA].max()
    bins = sorted(set([min_val] + Athresholds + [max_val]))
    if len(bins)-1 != len(Alabels):
        return None,None

    fold = data.loc[trainidx.union(validx)].copy()
    fold['region_A'] = pd.cut(fold[INDEXA],bins=bins, labels=range(len(Alabels)),ordered=False, include_lowest=True)
    fold['pred_B'] = pd.cut(fold[INDEXA], bins=bins, labels=Alabels, ordered=False, include_lowest=True).astype(str)

    region_groups = {i: {'label': label, 'data': fold[fold['region_A'] == i]} for i,label in enumerate(Alabels)}
    region_best = {}

    for region_id, region_info in region_groups.items():
        region_data = region_info['data']
        region_train = train_fold[train_fold.index.isin(region_data.index)]

        if len(region_train) < 5:
            continue
        counts = region_train['leaf_type'].value_counts()
        if (counts < 2).any():
            continue
   
        best_trainacc=0
        best_split=None

        for INDEXB in cols:
            if INDEXB == INDEXA:
                continue

            thresholdsB, labelsB = KDEfunction(region_train, INDEXB, prioradj=prior, priorarr=customweights)
            if not thresholdsB:
                continue

            minB, maxB = region_data[INDEXB].min(), region_data[INDEXB].max()
            binsB = sorted(set([minB] + thresholdsB + [maxB]))
            if len(binsB)-1!= len(labelsB):
                continue

            train_preds = pd.cut(region_train[INDEXB], bins=binsB, labels=labelsB, ordered=False, include_lowest=True)
            acc = get_metric(region_train['leaf_type'], train_preds)

            if acc > best_trainacc:
                best_trainacc = acc
                best_split = (thresholdsB, labelsB, INDEXB, binsB)
                region_best[region_id] = INDEXB

        if best_split is not None:
            thresholdsB, labelsB, INDEXB_used, binsB = best_split
            sub_preds = pd.cut(region_data[INDEXB_used],bins=binsB, labels=labelsB, ordered=False, include_lowest=True)
            fold.loc[region_data.index, 'pred_B'] = sub_preds.astype(str)

    val_preds = fold.loc[validx, 'pred_B']
    score = get_metric(val_fold['leaf_type'], val_preds)
    
    return score, region_best

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
best_score = 0
best_indexA = None

for INDEXA in cols:
    fold_scores = []
    fold_indices = []

    for train_idx, val_idx in skf.split(train, train['leaf_type']):
        train_index = train.iloc[train_idx].index
        val_index = train.iloc[val_idx].index

        if SINGLE:
            score = evalsingle(train, train_index, val_index, INDEXA)
        else:
            score, rbi = evaltwo(train, train_index, val_index, INDEXA)
            fold_indices.append(rbi)

        if score is not None:
            fold_scores.append(score)

    if fold_scores:
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"{INDEXA}: CV = {mean_score:.4f} ± {std_score/np.sqrt(5):.4f}")
        if not SINGLE:
            for rbi in fold_indices:
                print(f"{rbi}")

        if mean_score > best_score:
            best_score = mean_score
            best_indexA = INDEXA

print(f"Best INDEXA: {best_indexA} (CV: {best_score:.4f})")
INDEXA = best_indexA
Athresholds, Alabels = KDEfunction(train, INDEXA, prioradj=prior, priorarr = customweights)
min_val, max_val = plot[INDEXA].min(), plot[INDEXA].max()
bins = sorted(set([min_val] + Athresholds + [max_val]))

plot['region_A'] = pd.cut(plot[INDEXA], bins=bins, labels=range(len(Alabels)), ordered=False, include_lowest=True)
plot['pred_A'] = pd.cut(plot[INDEXA], bins=bins, labels=Alabels, ordered=False, include_lowest=True).astype(str)
plot['pred_B'] = plot['pred_A'].astype(str)

region_groups = {i: {'label': label, 'data': plot[plot['region_A'] == i]} for i, label in enumerate(Alabels)}
sub_splits = {}
region_bests = {}

if not SINGLE:
    for region_id, region_info in region_groups.items():
        region_data = region_info['data']
        region_train = train[train.index.isin(region_data.index)]

        if len(region_train) < 5:
            continue
        counts = region_train['leaf_type'].value_counts()
        if (counts < 2).any():
            continue
        best_acc = 0

        for INDEXB in cols:
            if INDEXB == INDEXA:
                continue

            thresholdsB, labelsB = KDEfunction(region_train, INDEXB, prioradj=prior, priorarr = customweights)
            if not thresholdsB:
                continue

            minB, maxB = region_data[INDEXB].min() - 1e-10, region_data[INDEXB].max() + 1e-10
            binsB = sorted(set([minB] + thresholdsB + [maxB]))
            if len(binsB) - 1 != len(labelsB):
                continue

            preds = pd.cut(region_train[INDEXB], bins=binsB, labels=labelsB, ordered=False, include_lowest=True)
            acc = get_metric(region_train['leaf_type'], preds)

            if acc > best_acc:
                best_acc = acc
                sub_splits[region_id] = (thresholdsB, labelsB, INDEXB)
                region_bests[region_id] = INDEXB

        if region_id in sub_splits:
            thresholdsB, labelsB, INDEXB_used = sub_splits[region_id]
            minB, maxB = region_data[INDEXB_used].min(), region_data[INDEXB_used].max()
            binsB = sorted([minB] + thresholdsB + [maxB])
            if len(binsB) - 1 != len(labelsB):
                continue
            sub_regions = pd.cut(region_data[INDEXB_used], bins=binsB, labels=labelsB, ordered=False, include_lowest=True)
            plot.loc[region_data.index, 'pred_B'] = sub_regions.astype(str)

pred_col = 'pred_A' if SINGLE else 'pred_B'

acc_test = accuracy_score(plot.loc[test.index, 'leaf_type'], plot.loc[test.index, pred_col])
acc_train = accuracy_score(plot.loc[train.index, 'leaf_type'], plot.loc[train.index, pred_col])
kappa_test = cohen_kappa_score(plot.loc[test.index, 'leaf_type'], plot.loc[test.index, pred_col])

y_pred = plot.loc[test.index, pred_col]
y_test = plot.loc[test.index, 'leaf_type']

if SAVEFIG:
    results_df = pd.DataFrame({'y_true': y_test,'y_pred_class': y_pred})
    results_df.to_csv(name, index=False)
    print("Saved to CSV")

print(f"Final - Train: {acc_train:.4f}, Test: {acc_test:.4f}, Kappa: {kappa_test:.4f}")
print(f"Best indexA: {INDEXA}")
if not SINGLE:
    for region_id, INDEXB in region_bests.items():
        print(f"  Region {region_id} ({Alabels[region_id]}): best indexB = {INDEXB}")

print(classification_report(plot.loc[test.index, 'leaf_type'], plot.loc[test.index, pred_col],digits=4))
labels = sorted(plot['leaf_type'].unique())
cm = confusion_matrix(plot.loc[test.index, 'leaf_type'], plot.loc[test.index, pred_col], labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Greens)
plt.show()

if ALL:
    customlabels1 = customlabels
else:
    customlabels1 = None

if SINGLE:
    fig, ax = plt.subplots(figsize=(3.5, 2.65),dpi=300)
    indexplot1D(plot, INDEXA, Athresholds, Alabels, ax, testing=False, customlabels=customlabels1, highlightindex=True)
else:
    fig, ax = plt.subplots(figsize=(3.5, 2.65), dpi=300)
    indexplot1D(plot, INDEXA, Athresholds, Alabels, ax, testing=False, customlabels=customlabels1, highlightindex=True)
    plt.tight_layout()
    plt.show() 

    for region_id, region_info in region_groups.items():
        fig, ax = plt.subplots(figsize=(3.5, 2.65), dpi=300)
        region_data = region_info['data']
        
        if region_id not in sub_splits:
            indexplot1D(region_data, INDEXA, [], [Alabels[region_id]], ax, testing=False, customlabels=customlabels1)
        else:
            INDEXB = region_bests.get(region_id)
            thresholds_B, labels_B, _ = sub_splits[region_id]
            indexplot1D(region_data, INDEXB, thresholds_B, labels_B, ax, testing=False, customlabels=customlabels1, multiple=True)
            
        plt.tight_layout()
        plt.show()



