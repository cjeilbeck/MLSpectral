import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from skorch import NeuralNetClassifier
from A_functions import read_data, indicesmav, indicescustomMLP, haircut, problemtype
from sklearn.preprocessing import FunctionTransformer
from scipy.signal import savgol_filter
from skorch.callbacks import EarlyStopping
import pandas as pd

def apply_savgol(x):
    return savgol_filter(x, window_length=51, polyorder=3, axis=1)

HAIRCUT = True
left, right = 206, 910
INDICES = True
SAVGOL=True
CUSTOM = False 

#typesofdatasetup
GUMCOMB = False
OAKONLY = False
if GUMCOMB: 
    problem = 'binary'
elif OAKONLY: 
    problem = 'oak'
else: 
    problem = 'all'

if INDICES:
    HAIRCUT = False
    SAVGOL = False

if CUSTOM:
    INDICES = False
    HAIRCUT = False
    SAVGOL = False

import random
import os

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


data = pd.read_csv("CSVfiles/datacalibrated.csv")
data = problemtype(data, problem)
y = data['leaf_type']
x = data.drop(columns=['leaf_type','sample_id'])

if HAIRCUT:
    x = haircut(x, left, right)
if INDICES:
    x = indicesmav(x)
if CUSTOM:
    x = indicescustomMLP(x)


labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)



idim = x_train.shape[1]
odim = len(np.unique(y_encoded))

class Brad(nn.Module):
    def __init__(self, input_N, classes_N, layer_sizes=(x, y), drop=0.2):
        super().__init__()
        n1, n2 = layer_sizes
        self.layers = torch.nn.Sequential(nn.Linear(input_N, n1),nn.Dropout(drop), nn.ReLU(), nn.Linear(n1, n2), nn.ReLU(),nn.Linear(n2, classes_N))
        
    def forward(self, x):
        return self.layers(x)


patience = 40

early_stopping = EarlyStopping(
    monitor='valid_loss', 
    patience=patience,          
    threshold=0.0001,     # Min imrpov
    lower_is_better=True)

net = NeuralNetClassifier(module=Brad,module__input_N=idim,module__classes_N=odim,criterion=nn.CrossEntropyLoss,optimizer=optim.AdamW,verbose=0,max_epochs=1000,callbacks=[early_stopping])

def epochcount(estimator,x,y):
    return len(estimator.named_steps['net'].history)

scoring = {
    'primary_score':'accuracy', 
    'epochs': epochcount}

pipeline_steps = []

if SAVGOL:
    pipeline_steps.append(('savgol', FunctionTransformer(apply_savgol)))
pipeline_steps.append(('scaler', StandardScaler()))
pipeline_steps.append(('net', net))
pipe = Pipeline(pipeline_steps)

params = {'net__module__layer_sizes': [(128,64),(64,32),(32,16)],
          'net__lr': [0.01,0.001],'net__module__drop': [0.1,0.2]}

#params = {'net__module__layer_sizes': [(128,64)],'net__lr': [0.01],'net__module__drop': [0.1]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, params, refit='primary_score', verbose=2, cv=cv, n_jobs=1, scoring=scoring)
gs.fit(x_train.astype(np.float32), y_train.astype(np.int64))

print("x--" * 60)
print(f"Best Accuracy: {gs.best_score_:.4f}")
print(f"Best Parameters: {gs.best_params_}")


results = pd.DataFrame(gs.cv_results_)
pd.set_option('display.max_colwidth', None)
results = results.sort_values(by='mean_test_primary_score', ascending=False)
print(results[['params', 'mean_test_primary_score', 'std_test_primary_score', 'mean_test_epochs', 'std_test_epochs']])


best_std = gs.cv_results_['std_test_primary_score'][gs.best_index_]
best_error = best_std / np.sqrt(5)
print(patience)
print(f"Best Error: {best_error:.4f}")


#SPECTRA (deterministic)                                                                                   params  mean_test_primary_score  std_test_primary_score  error             mean_test_epochs  std_test_epochs
7    #{'net__lr': 0.001, 'net__module__drop': 0.1, 'net__module__layer_sizes': (64, 32)}                 0.991667                0.010541             246.0        79.701945
8    #{'net__lr': 0.001, 'net__module__drop': 0.1, 'net__module__layer_sizes': (32, 16)}                 0.991667                0.012910             227.8        64.857999


#indexes
0    #{'net__lr': 0.01, 'net__module__drop': 0.1, 'net__module__layer_sizes': (128, 64)}                 0.938333                0.022111              55.6         7.964923
3    #{'net__lr': 0.01, 'net__module__drop': 0.2, 'net__module__layer_sizes': (128, 64)}                 0.938333                0.017951              55.4         7.391887



#indexescustom 

#0    {'net__lr': 0.01, 'net__module__drop': 0.1, 'net__module__layer_sizes': (128, 64)}                 0.986667                0.008498              79.2        32.145917




#indexes on gumcomb 

1    # {'net__lr': 0.01, 'net__module__drop': 0.1, 'net__module__layer_sizes': (64, 32)}                 0.973333                0.006236             130.8        15.276125
3    #{'net__lr': 0.01, 'net__module__drop': 0.2, 'net__module__layer_sizes': (128, 64)}                 0.973333                0.013333             118.0        34.047026


#indexes on oakonly

#   {'net__lr': 0.01, 'net__module__drop': 0.2, 'net__module__layer_sizes': (128, 64)}                 0.950000                0.013944              81.2        65.967871