import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline            
from scipy.signal import savgol_filter
from A_functions import haircut,read_data,indicesmav, indicescustomrf, problemtype
import numpy as np
import os
import random


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

left= 206
right=910

test_sizeinput = 0.2

INDICES = False
if INDICES:
    HAIRCUT = False
    USE_SAVGOL = False

CUSTOMRF = True
if CUSTOMRF:
    INDICES = False
    HAIRCUT = False
    USE_SAVGOL = False

data=pd.read_csv("CSVfiles/datacalibrated.csv")

GUMCOMB = False
OAKONLY = False
GUM = False
if GUMCOMB:
    problem = 'binary'
elif OAKONLY:
    problem = 'oak'
else:
    problem = 'all'
if GUM:
    problem = 'gum'

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

results = pd.DataFrame(grid_search.cv_results_)
pd.set_option('display.max_colwidth', None)
results = results.sort_values(by='mean_test_score', ascending=False)
print(results[['params', 'mean_test_score', 'std_test_score']])


#revampedsearch

#Best Parameters: {'rf_model__class_weight': None, 'rf_model__criterion': 'gini', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 50} INDEXES    # does imbalanced class weight help due to easy classification of young gum which would otherwise be prioritised?
#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'gini', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 200} #INDEXGUMCOMB

#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 10, 'rf_model__max_features': 'log2', 'rf_model__min_samples_leaf': 1, 'rf_model__n_estimators': 150} FOR NORMAL


#OAK
"""
64             {'rf_model__class_weight': None, 'rf_model__criterion': 'gini', 'rf_model__max_depth': 10, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 50}         0.943333        0.016159
42  {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 150}         0.943333        0.016159
46  {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 15, 'rf_model__max_features': 'log2', 'rf_model__n_estimators': 150}         0.943333        0.016159
50  {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 20, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 150}         0.943333        0.016159
54  {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 20, 'rf_model__max_features': 'log2', 'rf_model__n_estimators': 150}         0.943333        0.016159

"""


#CUSTOM
"""
{'rf_model__class_weight': None, 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 10, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 100}         0.980000        0.013540
117        {'rf_model__class_weight': None, 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 20, 'rf_model__max_features': 'log2', 'rf_model__n_estimators': 100}         0.980000        0.013540
38   {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 10, 'rf_model__max_features': 'log2', 'rf_model__n_estimators': 150}         0.980000        0.013540"""


















#SEARCHAFTER
#INDEXES
#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 200} INDEX
#Best Parameters: {'rf_model__class_weight': None, 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 10, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 150} HYPER
#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'gini', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 200} GUMCOMB
#Best Parameters: {'rf_model__class_weight': 'balanced', 'rf_model__criterion': 'log_loss', 'rf_model__max_depth': 15, 'rf_model__max_features': 'sqrt', 'rf_model__n_estimators': 150} OAKONLY