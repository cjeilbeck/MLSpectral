import pandas as pd
from pathlib import Path
import sys
cdir = Path(__file__).parent
root = cdir.parent
sys.path.append(str(root))
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline            
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt  
from scipy.signal import savgol_filter
from FUNCTIONS.A_functions import haircut,read_data,multispectral,indicesmav, indicescustomrf,problemtype, indicescustom, customlabels, indicescustomSVM, indicescustomMLP
import numpy as np
import os
import random
import seaborn as sns
from scipy.stats import entropy


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


def apply_savgol(x):
    return savgol_filter(x, window_length=51, polyorder=3, axis=1)

USE_SCALING = False
USE_SAVGOL = True
HAIRCUT = True 

CUSTOMBANDPLOTS = True
SAVEFIG = False
name = 'rf_predictionsOAK.csv'
z= 42

os.environ['PYTHONHASHSEED'] = str(z)
random.seed(z)
np.random.seed(z)  

left= 206
right=910

test_sizeinput = 0.2

#indices is commercial data, custom is custom bands; both off for hyperspectral processing
INDICES = True
CUSTOMRF = False
GUMCOMB = False
OAKONLY = False


if INDICES:
    HAIRCUT = False
    USE_SAVGOL = False

if CUSTOMRF:
    INDICES = False
    HAIRCUT = False
    USE_SAVGOL = False

data=pd.read_csv("CSVfiles/datacalibrated.csv")



if GUMCOMB:
    problem = 'binary'
elif OAKONLY:
    problem = 'oak'
else:
    problem = 'all'

data = problemtype(data, problem)
y = data['leaf_type']
x = data.drop(columns=['leaf_type','sample_id'])    

if HAIRCUT:
    x = haircut(x,left,right)
    print(f"trimmed wav:",x.columns[0],x.columns[-1])
if INDICES:
    x = indicesmav(x)
if CUSTOMRF:
    x = indicescustomrf(x)

pipeline_steps = []

if USE_SAVGOL:
    pipeline_steps.append(('savgol', FunctionTransformer(apply_savgol)))

if USE_SCALING:
    pipeline_steps.append(('scaler', StandardScaler()))

#updated based off gridsearch results
TEST=False

if TEST:
    rf = RandomForestClassifier(n_estimators=150, random_state=z)
elif INDICES and GUMCOMB and TEST==False:
    rf = RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=15,max_features='sqrt',
                                class_weight='balanced',random_state=z)
elif INDICES and TEST==False:
    rf = RandomForestClassifier(n_estimators=50, criterion='gini',max_depth=15,max_features='sqrt',
                                class_weight=None,random_state=z)
elif CUSTOMRF and TEST==False:
    rf = RandomForestClassifier(n_estimators=100, criterion='log_loss',max_depth=10,max_features='sqrt',
                                class_weight=None,random_state=z)
elif OAKONLY and TEST==False:
    rf = RandomForestClassifier(n_estimators=50,criterion='gini', max_depth=10, max_features='sqrt',
                                class_weight=None,random_state=z)
else:
    rf = RandomForestClassifier(n_estimators=150,criterion='log_loss',max_depth=10,max_features='log2',
                                class_weight='balanced',random_state=z)



pipeline_steps.append(('rf_model', rf))
model = Pipeline(pipeline_steps)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_sizeinput,random_state=z,stratify=y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=z)
cv_scores = cross_val_score(model, x_train, y_train, cv=skf)

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {(cv_scores.std())/np.sqrt(5):.4f}")

model.fit(x_train, y_train)
probs = model.predict_proba(x_test)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Testing Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")

if SAVEFIG:
    results_df = pd.DataFrame({'y_true': y_test,'y_pred_rf': y_pred})
    results_df.to_csv(name, index=False)
    print("Saved to CSV")

importances =[]

for train_index, val_index in skf.split(x_train, y_train):
    x_tr_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
    y_tr_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    model.fit(x_tr_fold, y_tr_fold)
    rf = model.named_steps['rf_model']
    importances.append(rf.feature_importances_)
        
mean_importance = np.mean(importances, axis=0)
std_importance = np.std(importances, axis=0)

err_importance = std_importance / np.sqrt(5)

featurefile = pd.DataFrame({'index': x_train.columns,'importance': mean_importance,'std': std_importance})

if INDICES:
    featurefile.to_csv('RFindeximportances.csv', index=False)
elif CUSTOMRF:
    featurefile.to_csv('RFcustomimportances.csv', index=False)


if INDICES or CUSTOMRF:
    plt.figure(figsize=(12, 6))
    #featurefile = featurefile.sort_values(by='importance', ascending=False)
    colors = [(0.4,0.6,0.2,0.8) if i<4 else (0.2,0.4,0.8,0.8) for i in range(len(featurefile))]
    meanallimportance = featurefile['importance'].mean()
    plt.axhline(meanallimportance, color='black', alpha=0.5,linestyle='--')
    plt.bar(featurefile['index'], featurefile['importance'], yerr=featurefile['std'], color=colors,capsize=10,error_kw={'alpha': 1, 'linewidth': 2})
    plt.xlabel("Spectral Feature")
    plt.ylabel("Importance Score")
    plt.show()

else:
    featurefile['index'] = pd.to_numeric(featurefile['index'])
    featurefile['smoothed'] = savgol_filter(featurefile['importance'], window_length=51, polyorder=3, axis=-1)
    featurefile['smoothed_std'] = savgol_filter(featurefile['std'], window_length=51, polyorder=3, axis=-1)
    featurefile['smoothed_std']=featurefile['smoothed_std']/np.sqrt(5)
    centres=[415,545,675,715]
    widths =[32,32,32,32]


    plt.figure(figsize=(3.5, 2.65),dpi=300)
    plt.plot(featurefile['index'], featurefile['smoothed'], color='red')
    if CUSTOMBANDPLOTS:
        for i, (c, w) in enumerate(zip(centres, widths)):
            plt.axvspan(c-w/2, c+w/2, facecolor=(1,0.8,0,0.15),edgecolor=(0,0,0,1), linewidth=0.2,linestyle='--')
    
            #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.fill_between(featurefile['index'], featurefile['smoothed'] - featurefile['smoothed_std'], featurefile['smoothed'] + featurefile['smoothed_std'], color='red', alpha=0.2,edgecolor='none')
    
  
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Feature Importance")
   
    plt.tight_layout()
    plt.show()

    wavs = featurefile['index'].values
    imps = featurefile['importance'].values
    half = 16  
    centers = []
    windowtot = []
    for center in np.arange(wavs.min()+half,wavs.max()-half, 1):
        mask = (wavs >= center-half) & (wavs <= center+half)
        centers.append(center)
        windowtot.append(imps[mask].sum())
    centers = np.array(centers)
    windowtot = np.array(windowtot)
    plt.figure(figsize=(15, 5))
    plt.plot(centers, windowtot, color='blue', linewidth=1.5)
    plt.show()

shentropy = entropy(probs, axis=1,base=2)
entropyfile = pd.DataFrame({'Class': y_test,'Entropy': shentropy})


if not OAKONLY and not GUMCOMB:
        entropyfile['Class'] = entropyfile['Class'].map(customlabels)
        displaylabels = [customlabels[c] for c in model.named_steps['rf_model'].classes_]
else:
        displaylabels = model.named_steps['rf_model'].classes_

plt.figure(figsize=(12, 6))
sns.boxplot(x='Class', y='Entropy', data=entropyfile, palette="Set2", hue='Class', legend=False, linewidth=1.2, width=0.5, notch=False)
plt.ylabel('Entropy')
plt.show()

fig, ax = plt.subplots(figsize=(3, 3), dpi=300)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=displaylabels)
disp.plot(cmap=plt.cm.Reds,ax=ax,colorbar=False, values_format='d')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
ax.set_xlabel('Predicted Label', fontsize=9)
ax.set_ylabel('True Label', fontsize=9)
plt.tight_layout()
plt.show()


