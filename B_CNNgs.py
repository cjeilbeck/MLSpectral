import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import savgol_filter
from skorch import NeuralNetClassifier

from A_functions import read_data, haircut, indicesmav

HAIRCUT = True
left, right = 200, 900
INDICES = True
USE_SAVGOL = True
seed = 42

if INDICES:
    HAIRCUT = False
    USE_SAVGOL = False

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


x, y = read_data("CSVfiles/datacalibrated.csv")

if HAIRCUT:
    x = haircut(x, left, right)
if INDICES:
    x = indicesmav(x)

labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
)

idim = x_train.shape[1]  # input_L
odim = len(np.unique(y_encoded))  # classes_N


class Oliver(nn.Module):
    def __init__(self, input_L, classes_N, neurons1=64, neurons2=128, 
                 kernel1=3, kernel2=3, poolkernel=2, dropprob=0.1):
        super().__init__() 
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=neurons1, kernel_size=kernel1, padding=(kernel1-1)//2)
        self.pool = nn.MaxPool1d(kernel_size=poolkernel) 
        self.conv2 = nn.Conv1d(in_channels=neurons1, out_channels=neurons2, kernel_size=kernel2, padding=(kernel2-1)//2)
        

        self.final_len = input_L // (poolkernel**2)
        self.flattened = neurons2 * self.final_len

        self.fc = nn.Linear(self.flattened, classes_N)
        self.drop = nn.Dropout1d(dropprob)
        
    def forward(self, x):
      
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = self.pool(x)  
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  
        x = self.fc(x)
        return x

def apply_savgol(X):
    return savgol_filter(X, window_length=51, polyorder=3, axis=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = NeuralNetClassifier(module=Oliver,module__input_L=idim,module__classes_N=odim,criterion=nn.CrossEntropyLoss,optimizer=optim.Adam,device=device,verbose=0)

pipeline_steps = []
if USE_SAVGOL:
    pipeline_steps.append(('savgol', FunctionTransformer(apply_savgol)))
pipeline_steps.append(('scaler', StandardScaler()))
pipeline_steps.append(('net', net))

pipe = Pipeline(pipeline_steps)


params = {'net__module__neurons1': [16,32, 64],'net__module__neurons2': [32,64, 128],'net__module__dropprob': [0.1, 0.2],'net__lr': [0.01, 0.001],'net__max_epochs': [150,300,600]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

gs = GridSearchCV(pipe, params, refit=True, verbose=2, cv=cv, n_jobs=1, scoring='accuracy')


gs.fit(x_train.astype(np.float32), y_train.astype(np.int64))

print("-" * 20)
print(f"Best Accuracy: {gs.best_score_:.6f}")
print(f"Best Parameters: {gs.best_params_}")

results = pd.DataFrame(gs.cv_results_)
pd.set_option('display.max_colwidth', None)
results = results.sort_values(by='mean_test_score', ascending=False)
print(results[['params', 'mean_test_score']].head())


best_model = gs.best_estimator_
y_pred = best_model.predict(x_test.astype(np.float32))

predlabels = labelencoder.inverse_transform(y_pred)
testlabels = labelencoder.inverse_transform(y_test)


accuracy = accuracy_score(testlabels, predlabels)
print(f"Accuracy: {accuracy:.6f}")
print("Classification Report:")
print(classification_report(testlabels, predlabels))