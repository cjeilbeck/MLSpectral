import numpy as np
import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import savgol_filter
from A_functions import haircut,multiregion,read_data,scaling,gradanal, multispectral, indicesmav, indicespaper
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams.update({'font.size': 14})

USE_SCALING = True 
USE_PCA=False 
ncomp = 4
USE_SAVGOL = False
smooth =51

HAIRCUT = False 
left = 200  #this changes accuracy a lot with minor tweaks
right = 900

MULTIREGION = False
centers = [560,650, 730,860]
width = [32,32,32,26]
GRADANALYSIS = True

INDICES = True

neurons2=64
neurons1=2*neurons2

epochs = 300
noisefactor = 0.0
lr=0.01
seed = 42
test_sizeinput = 0.2
drop=0.2
torch.manual_seed(seed)
np.random.seed(seed)

x,y=read_data("CSVfiles/datacalibrated.csv")

if INDICES:
    x = indicesmav(x)
if HAIRCUT:
    x = haircut(x,left, right)
    print(f"trimmed wav:",x.columns[0],x.columns[-1])
if MULTIREGION:
    x = multispectral(x, centers, width)

labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=test_sizeinput,random_state=seed,stratify=y_encoded)


if USE_SAVGOL:
    x_train = savgol_filter(x_train, window_length=smooth, polyorder=3, axis=1)
    x_test = savgol_filter(x_test, window_length=smooth, polyorder=3, axis=1)
if USE_SCALING:
    x_train,x_test = scaling(x_train,x_test)
if USE_PCA:
    pca = PCA(n_components=ncomp)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
idim = x_train.shape[1]  #input/output dims
odim = len(np.unique(y_encoded)) 

x_traint = torch.tensor(x_train, dtype=torch.float32)
y_traint = torch.tensor(y_train, dtype=torch.long)
x_testt = torch.tensor(x_test, dtype=torch.float32)
y_testt = torch.tensor(y_test, dtype=torch.long)

class Brad(nn.Module):
    def __init__(self, input_N, classes_N):
        super().__init__()

        self.layers = torch.nn.Sequential(nn.Linear(input_N, neurons1),nn.Dropout(drop), nn.ReLU(), nn.Linear(neurons1, neurons2), nn.ReLU(),nn.Linear(neurons2, classes_N))
        
    def forward(self, x):
        z = self.layers(x)
        return z

model = Brad(idim, odim)
optimizer = optim.AdamW(model.parameters(), lr=lr)
lossfunc = nn.CrossEntropyLoss()

train_losses = []
val_losses = [] #training validation losses addition

for epoch in range(epochs):
   
    model.train() 
    optimizer.zero_grad()
    outputs = model(x_traint)
    loss = lossfunc(outputs, y_traint)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    model.eval()   
    with torch.no_grad(): 
        val_outputs = model(x_testt)
        val_loss = lossfunc(val_outputs, y_testt)
        val_losses.append(val_loss.item())


    #if epoch%10 == 0:
        #print(epoch)

noise = torch.randn_like(x_testt)*noisefactor
x_testn = x_testt + noise

model.eval()
with torch.no_grad():
    test_outputs = model(x_testn)
    _, y_pred = torch.max(test_outputs, 1)

    #entropy
    probs = torch.softmax(test_outputs, 1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=1).numpy()




y_prednp = y_pred.numpy()
y_testnp = y_testt.numpy()

predlabels = labelencoder.inverse_transform(y_prednp)
testlabels = labelencoder.inverse_transform(y_testnp)

entropyfile = pd.DataFrame({'Entropy': entropy,'Class': testlabels})

accuracy = accuracy_score(testlabels, predlabels)
print(neurons1,neurons2,epochs,lr,drop)
print(f"Accuracy: {accuracy:.6f}")
print("Classification Report:")
print(classification_report(testlabels, predlabels))

if USE_PCA:

    PC = np.arange(pca.n_components) + 1
    plt.plot(PC, pca.explained_variance_ratio_, color='red')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance contribution')
    plt.show()

if GRADANALYSIS:
    wav, smoothed_attr = gradanal(model,x,x_testt,y_testt,smooth,left,right,0,savgol=USE_SAVGOL,islabel=INDICES)
    if INDICES:
        plt.figure(figsize=(10, 5))
        plt.bar(wav, smoothed_attr, color='purple')
        plt.xticks(rotation=90)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Mean Absolute Attribution")
        plt.show()
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(wav, smoothed_attr, color='purple')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Mean Absolute Attribution")
        plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(entropy, bins=10, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Entropy')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Class', y='Entropy', data=entropyfile, palette="rocket")
plt.xticks(rotation=45)
plt.ylabel('Entropy')
plt.show()