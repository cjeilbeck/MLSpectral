import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from skorch import NeuralNetClassifier
from A_functions import read_data, indicesmav, scaling

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

x, y = read_data("CSVfiles/datacalibrated.csv")
x = indicesmav(x)

labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)

x_train,x_test = scaling(x_train,x_test)

idim = x_train.shape[1]
odim = len(np.unique(y_encoded))

class Brad(nn.Module):
    def __init__(self, input_N, classes_N, layer_sizes=(x, y), drop=0.1):
        super().__init__()
        n1, n2 = layer_sizes
        self.layers = torch.nn.Sequential(nn.Linear(input_N, n1),nn.Dropout(drop), nn.ReLU(), nn.Linear(n1, n2), nn.ReLU(),nn.Linear(n2, classes_N))
        
    def forward(self, x):
        return self.layers(x)

net = NeuralNetClassifier(module=Brad,module__input_N=idim,module__classes_N=odim,criterion=nn.CrossEntropyLoss,optimizer=optim.AdamW,verbose=0)
params = {'module__layer_sizes': [(64, 32)],'lr': [0.01],'module__drop': [0.2],'max_epochs': [300]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(net, params, refit=True, verbose=2, cv=cv, n_jobs=-1, scoring='accuracy')

gs.fit(x_train.astype(np.float32), y_train.astype(np.int64))

print("-" * 25)
print(f"Best Accuracy: {gs.best_score_:.6f}")
print(f"Best Parameters: {gs.best_params_}")


results = pd.DataFrame(gs.cv_results_)
pd.set_option('display.max_colwidth', None)
results = results.sort_values(by='mean_test_score', ascending=False)
print(results[['params', 'mean_test_score']])