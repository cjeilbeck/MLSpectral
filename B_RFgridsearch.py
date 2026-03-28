import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline            
from scipy.signal import savgol_filter
from A_functions import haircut,read_data,indicesmav, indicescustomrf
import numpy as np
import os
import random
import torch

def apply_savgol(x):
    return savgol_filter(x, window_length=51, polyorder=3, axis=1)

USE_SCALING = False
USE_PCA = False
USE_SAVGOL = True
HAIRCUT = True 

z = 42
os.environ['PYTHONHASHSEED'] = str(z)
random.seed(z)
np.random.seed(z)
torch.manual_seed(z)
if torch.cuda.is_available():
    torch.cuda.manual_seed(z)
    torch.cuda.manual_seed_all(z)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

left= 200
right=900

test_sizeinput = 0.2

INDICES = True
if INDICES:
    HAIRCUT = False
    USE_SAVGOL = False

CUSTOMRF = True
if CUSTOMRF:
    INDICES = False
    HAIRCUT = False
    USE_SAVGOL = False

x,y=read_data("CSVfiles/datacalibrated.csv")

GUMCOMB = False
OAKONLY = False

if GUMCOMB:
    y = y.replace(['Gum_young', 'Gum_old'], 'Gum')
    y = y.replace(['Pine', 'Oakcork'], 'Other')

if OAKONLY:
    y = y.replace(['Gum_young', 'Gum_old', 'Pine'], 'Other')

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

pipeline_steps.append(('rf_model', RandomForestClassifier(random_state=z)))
model = Pipeline(pipeline_steps)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_sizeinput, random_state=z, stratify=y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=z)

param_grid = {'rf_model__criterion': ['gini', 'log_loss'],'rf_model__n_estimators': [50, 100, 150,200],'rf_model__max_depth': [10, 15, 20, None],
              'rf_model__max_features': ['sqrt', 'log2'],'rf_model__class_weight': ['balanced', None]
}

print("Running GridSearch...")
grid_search = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1,verbose=2)
grid_search.fit(x_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")


#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__min_samples_leaf': 1, 'rf_model__n_estimators': 50} FOR INDEX
#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 10, 'rf_model__max_features': 'log2', 'rf_model__min_samples_leaf': 1, 'rf_model__n_estimators': 150} FOR NORMAL



#revampedsearch

#Best Parameters: {'rf_model__class_weight': None, 'rf_model__criterion': 'gini', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 50} INDEXES
#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'gini', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 200} #INDEXGUMCOMB
#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'gini', 'rf_model__max_depth': 10, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 150} #INDEXESCUSTOM
#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 10, 'rf_model__max_features': 'log2', 'rf_model__min_samples_leaf': 1, 'rf_model__n_estimators': 150} FOR NORMAL
