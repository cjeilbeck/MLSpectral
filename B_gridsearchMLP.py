import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from skorch import NeuralNetClassifier
from A_functions import read_data, indicesmav
from sklearn.preprocessing import FunctionTransformer
from scipy.signal import savgol_filter
from A_functions import read_data, indicesmav, haircut

def apply_savgol(x):
    return savgol_filter(x, window_length=51, polyorder=3, axis=1)

HAIRCUT = True
left, right = 200, 900
INDICES = False
SAVGOL=True

if INDICES:
    HAIRCUT = False
    SAVGOL = False

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


x, y = read_data("CSVfiles/datacalibrated.csv")

if HAIRCUT:
    x = haircut(x, left, right)
if INDICES:
    x = indicesmav(x)


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

net = NeuralNetClassifier(module=Brad,module__input_N=idim,module__classes_N=odim,criterion=nn.CrossEntropyLoss,optimizer=optim.AdamW,verbose=0)

pipeline_steps = []

if SAVGOL:
    pipeline_steps.append(('savgol', FunctionTransformer(apply_savgol)))
pipeline_steps.append(('scaler', StandardScaler()))
pipeline_steps.append(('net', net))
pipe = Pipeline(pipeline_steps)

params = {'net__module__layer_sizes': [(64, 32),(128,64),(32,16)],'net__lr': [0.01,0.001],'net__max_epochs': [100,300,600],'net__module__drop': [0.1,0.2]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, params, refit=True, verbose=2, cv=cv, n_jobs=-1, scoring='accuracy')
gs.fit(x_train.astype(np.float32), y_train.astype(np.int64))

print("-" * 10)
print(f"Best Accuracy: {gs.best_score_:.6f}")
print(f"Best Parameters: {gs.best_params_}")


results = pd.DataFrame(gs.cv_results_)
pd.set_option('display.max_colwidth', None)
results = results.sort_values(by='mean_test_score', ascending=False)
print(results[['params', 'mean_test_score']])

#INDICES:    {'net__lr': 0.01, 'net__max_epochs': 600, 'net__module__drop': 0.2, 'net__module__layer_sizes': (128, 64)}         0.956667
#SPECTRA     {'net__lr': 0.01, 'net__max_epochs': 300, 'net__module__drop': 0.1, 'net__module__layer_sizes': (32, 16)}         0.998333