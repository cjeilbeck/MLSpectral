import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import FunctionTransformer
from A_functions import read_data,multiregion, multispectral
from sklearn.model_selection import StratifiedKFold, cross_val_score
def apply_savgol(x):  
    return savgol_filter(x, window_length=11,polyorder=3,axis=1)

USE_SCALING = True
USE_SAVGOL = False
seed = 42     # Seed
comp = 6    # PCA components

KERNEL = 'linear' #try 'linear', 'poly', 'rbf', 'sigmoid' 
Cs = 10     
GAMMA = 1e-4

#3648 wavelength values for reference, 345 - 1038nm

trainsplit = True  #if false does cv
window = 32
step = 8
x,y=read_data("CSVfiles/datacalibrated.csv")
wav = pd.to_numeric(x.columns)
start = wav.min()
end = wav.max()


plt.figure(figsize=(15, 6))

for c in Cs:
    print(f"C:{c}")
    accuracies = []
    centers = []
    win_start=start
    pipeline_steps = []
    if USE_SAVGOL:
        pipeline_steps.append(('savgol', FunctionTransformer(apply_savgol)))
    if USE_SCALING:
        pipeline_steps.append(('scaler', StandardScaler()))

    pipeline_steps.append(('svm_model', SVC(kernel=KERNEL,C=c,gamma=GAMMA)))

    while win_start + window <= end:
        print(win_start)
        win_end = win_start+window
        center = win_start+(window/2.0)
        
        x_window = multispectral(x, np.array([center]), window, Dif=False)
        n_features = x_window.shape[1]
        
        if trainsplit:
            x_train,x_test,y_train,y_test = train_test_split(x_window,y,test_size=0.2,random_state=seed,stratify=y)
            model = Pipeline(pipeline_steps)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            centers.append(center)
            win_start += step
        
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            model = Pipeline(pipeline_steps)
            scores = cross_val_score(model, x_window, y, cv=cv, scoring='accuracy',n_jobs=-1)
            accuracy = scores.mean()
            accuracies.append(accuracy)
            centers.append(center)
            win_start += step
    
    plt.plot(centers, accuracies, marker='.',linestyle='-',label=f'C = {c}')

plt.xlabel('Wavelength Window Center (nm)')
plt.ylabel('Model Accuracy')
plt.legend() 
plt.show()