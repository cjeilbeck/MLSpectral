import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline            
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt  
from scipy.signal import savgol_filter, find_peaks, peak_widths
from A_functions import haircut,read_data,multispectral,indicesmav, indicescustomrf,problemtype, indicescustom
import numpy as np
import os
import random
import seaborn as sns
from scipy.stats import entropy

plt.rcParams.update({'font.size': 25})
def apply_savgol(x):
    return savgol_filter(x, window_length=51, polyorder=3, axis=1)

USE_SCALING = False
USE_SAVGOL = True
HAIRCUT = True 

CUSTOMBANDPLOTS = False

z= 42

os.environ['PYTHONHASHSEED'] = str(z)
random.seed(z)
np.random.seed(z)  

left= 206
right=910

test_sizeinput = 0.2

INDICES = True 
CUSTOMRF = False


if INDICES:
    HAIRCUT = False
    USE_SAVGOL = False

if CUSTOMRF:
    INDICES = False
    HAIRCUT = False
    USE_SAVGOL = False

data=pd.read_csv("CSVfiles/datacalibrated.csv")

GUMCOMB = False
OAKONLY = False

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
    rf = RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=15,max_features='sqrt',class_weight='balanced',random_state=z)
elif INDICES and TEST==False:
    rf = RandomForestClassifier(n_estimators=200, criterion='log_loss',max_depth=15,max_features='sqrt',class_weight='balanced',random_state=z)
elif CUSTOMRF and TEST==False:
    rf = RandomForestClassifier(n_estimators=150,criterion='gini',max_depth=10,max_features='sqrt',class_weight='balanced',random_state=z)
elif OAKONLY and TEST==False:
    rf = RandomForestClassifier(n_estimators=150,criterion='log_loss', max_depth=15, max_features='sqrt',class_weight='balanced',random_state=z)
else:
    rf = RandomForestClassifier(n_estimators=150,criterion='log_loss',max_depth=10,max_features='sqrt',class_weight=None,random_state=z)



pipeline_steps.append(('rf_model', rf))
model = Pipeline(pipeline_steps)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_sizeinput,random_state=z,stratify=y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=z)
cv_scores = cross_val_score(model, x_train, y_train, cv=skf)

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Testing Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.named_steps['rf_model'].classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
"""
if INDICES and GUMCOMB==False and CUSTOMRF==False:
    results_df = pd.DataFrame({'y_true': y_test,'y_pred_rf': y_pred})
    results_df.to_csv('rf_predictions.csv', index=False)
    print("Saved to CSV")
"""
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

if INDICES or CUSTOMRF:
    plt.figure(figsize=(12, 6))
    plt.bar(featurefile['index'], featurefile['importance'], yerr=[np.zeros(len(featurefile)), featurefile['std']], color='purple',capsize=10,error_kw={'alpha': 1, 'linewidth': 2})
    plt.xlabel("Index")
    plt.ylabel("Importance Score")
    plt.show()
else:
    featurefile['index'] = pd.to_numeric(featurefile['index'])
    featurefile['smoothed'] = savgol_filter(featurefile['importance'], window_length=51, polyorder=3, axis=-1)
    featurefile['smoothed_std'] = savgol_filter(featurefile['std'], window_length=51, polyorder=3, axis=-1)


    centres=[420,555,676,710]
    widths =[32,20,20,32]


    plt.figure(figsize=(12, 6))
    plt.plot(featurefile['index'], featurefile['smoothed'], color='red', linewidth=2)
    if CUSTOMBANDPLOTS:
        for i, (c, w) in enumerate(zip(centres, widths)):
            plt.axvspan(c-w/2, c+w/2, alpha=0.25, color='steelblue', label=f'Centre: {c}nm')
    
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.fill_between(featurefile['index'], featurefile['smoothed'] - featurefile['smoothed_std'], featurefile['smoothed'] + featurefile['smoothed_std'], color='red', alpha=0.2, label='±1 Std. Dev.')
    
  
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()


probs = model.named_steps['rf_model'].predict_proba(x_test)
shentropy = entropy(probs, axis=1,base=2)
entropyfile = pd.DataFrame({'Class': y_test,'Entropy': shentropy})
plt.figure(figsize=(12, 6))
sns.boxplot(x='Class', y='Entropy', data=entropyfile, palette="rocket",hue='Class',legend=False)

plt.ylabel('Entropy')
plt.show()
