import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline            
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt  
from scipy.signal import savgol_filter
from sklearn.preprocessing import FunctionTransformer
from A_functions import haircut,read_data,multispectral,indicesmav
import numpy as np

def apply_savgol(x):
    return savgol_filter(x, window_length=51, polyorder=3, axis=1)

USE_SCALING = False
USE_PCA = False
USE_SAVGOL = True
HAIRCUT = True 
z = 42  # Seed

left= 200
right=900
#3648 wavelength values for reference, 345 - 1038nm

test_sizeinput = 0.2
MULTIREGION = False
centers = [560,650, 730,860]  
width = [32,32,32,26]

INDICES = False 

n_estimators = 150

x,y=read_data("CSVfiles/datacalibrated.csv")

if HAIRCUT:
    x = haircut(x,left,right)
    print(f"trimmed wav:",x.columns[0],x.columns[-1])
if MULTIREGION:
    x = multispectral(x, centers, width)
if INDICES:
    x = indicesmav(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_sizeinput,random_state=z,stratify=y)

pipeline_steps = []

if USE_SAVGOL:
    pipeline_steps.append(('savgol', FunctionTransformer(apply_savgol)))

if USE_SCALING:
    pipeline_steps.append(('scaler', StandardScaler()))


pipeline_steps.append(('rf_model', RandomForestClassifier(n_estimators, random_state=z)))
model = Pipeline(pipeline_steps)


model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


    
