import numpy as np
import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import savgol_filter
from A_functions import haircut,read_data,scaling,gradanal, indicesmav, indicescustomMLP, problemtype, indicespaper
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams.update({'font.size': 25})
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import os

USE_SCALING = True
USE_PCA=False
ncomp = 4
USE_SAVGOL = True
smooth =51

TESTGRAPHS = True
CVGRAPHS = True

HAIRCUT = True
left = 206  #this changes accuracy a lot with minor tweaks
right = 910



GRADANALYSIS = True

#typesofdatasetup
GUMCOMB = False
if GUMCOMB: problem = 'binary'

OAKONLY = False
if OAKONLY: problem = 'oak'

else: problem = 'all'

INDICES = True
CUSTOM = False

if INDICES:
    HAIRCUT = False
    USE_SAVGOL = False

if CUSTOM:
    INDICES = False
    HAIRCUT = False
    USE_SAVGOL = False



test_sizeinput = 0.2

    
if INDICES and GUMCOMB:
    lr=0.01

    drop=0.1
    neurons2=32
    neurons1=2*neurons2
    epochs = 131

elif INDICES and OAKONLY:
    lr=0.01

    drop=0.2
    neurons2=64
    neurons1=2*neurons2
    epochs = 81

elif INDICES or CUSTOM:
    
    lr=0.01
    neurons2=64
    neurons1=2*neurons2
    if INDICES:
        epochs = 55
        drop=0.2
    if CUSTOM:
        epochs = 79
        drop = 0.1

else:
    epochs = 228
    lr=0.001

    drop=0.1
    neurons2=16
    neurons1=2*neurons2

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

if INDICES:
    x = indicesmav(x)    
if CUSTOM:
    x = indicescustomMLP(x)
if HAIRCUT:
    x = haircut(x,left, right)
    print(f"trimmed wav:",x.columns[0],x.columns[-1])




labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=test_sizeinput,random_state=seed,stratify=y_encoded)




if USE_SAVGOL:
    x_train = savgol_filter(x_train, window_length=smooth, polyorder=3, axis=1)
    x_test = savgol_filter(x_test, window_length=smooth, polyorder=3, axis=1)

if isinstance(x_train, pd.DataFrame):
    x_train = x_train.values
if isinstance(x_test, pd.DataFrame):
    x_test = x_test.values

idim = x_train.shape[1]  #input/output dims
odim = len(np.unique(y_encoded))





class Brad(nn.Module):
    def __init__(self, input_N, classes_N):
        super().__init__()


        self.layers = torch.nn.Sequential(nn.Linear(input_N, neurons1),nn.Dropout(drop), nn.ReLU(), nn.Linear(neurons1, neurons2), nn.ReLU(),nn.Linear(neurons2, classes_N))
       
    def forward(self, x):
        z = self.layers(x)
        return z




skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

all_trainloss = []
all_valloss = []
fold_accuracies = []
all_entropy = []
all_testlabels = []
all_predlabels = []
all_attributions = []




for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
    xfold_train = x_train[train_idx]
    yfold_train = y_train[train_idx]
    xfold_val = x_train[val_idx]
    yfold_val = y_train[val_idx]

    xfold_train,xfold_val = scaling(xfold_train, xfold_val)

    x_traint = torch.tensor(xfold_train, dtype=torch.float32)
    y_traint = torch.tensor(yfold_train, dtype=torch.long)
    x_valt = torch.tensor(xfold_val, dtype=torch.float32)
    y_valt = torch.tensor(yfold_val, dtype=torch.long)

    model = Brad(idim, odim)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossfunc = nn.CrossEntropyLoss()


    train_losses = []
    val_losses = [] 


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
            val_outputs = model(x_valt)
            val_loss = lossfunc(val_outputs, y_valt)
            val_losses.append(val_loss.item())




    #if epoch%10 == 0:
        #print(epoch)




    model.eval()
    with torch.no_grad():
        test_outputs = model(x_valt)
        _, y_pred = torch.max(test_outputs, 1)


    #entropy
        probs = torch.softmax(test_outputs, 1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=1).numpy()


    y_prednp = y_pred.numpy()
    y_valtnp = y_valt.numpy()
    predlabels = labelencoder.inverse_transform(y_prednp)
    testlabels = labelencoder.inverse_transform(y_valtnp)

    foldacc = accuracy_score(testlabels, predlabels)
    fold_accuracies.append(foldacc)
    all_trainloss.append(train_losses)
    all_valloss.append(val_losses)
    all_entropy.extend(entropy)
    all_testlabels.extend(testlabels)
    all_predlabels.extend(predlabels)



    if GRADANALYSIS:
        wav, smoothed_attr = gradanal(model,x,x_valt,y_valt,smooth,left,right,0,savgol=USE_SAVGOL,islabel=INDICES or CUSTOM)
        all_attributions.append((wav, smoothed_attr))


print("-"*25)
print(f"CV Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies)/np.sqrt(len(fold_accuracies)):.4f}")
print("error quoted not std")
print("-"*25)
print("fold accuracies:", fold_accuracies)
print("-"*25)



if CVGRAPHS:
    cm = confusion_matrix(all_testlabels, all_predlabels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labelencoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


    if GRADANALYSIS:
        attributions = np.array([attr for _, attr in all_attributions])
        wav = all_attributions[0][0]  
        mean_attr = attributions.mean(axis=0)
        std_attr = attributions.std(axis=0)

        if INDICES or CUSTOM:
            plt.figure(figsize=(10, 5))
            plt.bar(wav, mean_attr, yerr=std_attr, color='purple', capsize=3, alpha=0.8)
            plt.xlabel("Index/Band")
            plt.ylabel("Mean Absolute Attribution")
            plt.show()
        else:
            wav = np.array(wav, dtype=float)
            plt.figure(figsize=(10, 5))
            plt.plot(wav, mean_attr, color='purple')
            plt.fill_between(wav, mean_attr - std_attr, mean_attr + std_attr, alpha=0.2, color='purple')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Mean Absolute Attribution")
            plt.show()

    mean_train = np.array(all_trainloss).mean(axis=0)
    std_train = np.array(all_trainloss).std(axis=0)
    mean_val = np.array(all_valloss).mean(axis=0)
    std_val = np.array(all_valloss).std(axis=0)
    ep = np.arange(epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(mean_train, color='blue', label='Train Loss')
    plt.fill_between(ep, mean_train - std_train, mean_train + std_train, alpha=0.2, color='blue', label='Std. Dev.')
    plt.plot(mean_val, color='red', label='Validation Loss')
    plt.fill_between(ep, mean_val - std_val, mean_val + std_val, alpha=0.2, color='red', label='Std. Dev.')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



entropyfile = pd.DataFrame({'Entropy': all_entropy,'Class': all_testlabels})

plt.figure(figsize=(8, 5))
plt.hist(all_entropy, bins=10, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Entropy')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Class', y='Entropy', data=entropyfile, palette="rocket", hue='Class', legend=False)
#plt.xticks(rotation=45)
plt.ylabel('Entropy')
plt.show()





#============
#test set bit

if USE_SCALING:
    x_train,x_test = scaling(x_train,x_test)

x_traint = torch.tensor(x_train, dtype=torch.float32)
y_traint = torch.tensor(y_train, dtype=torch.long)
x_testt = torch.tensor(x_test, dtype=torch.float32)
y_testt = torch.tensor(y_test, dtype=torch.long)


modeltest = Brad(idim, odim)
optimizer = optim.AdamW(modeltest.parameters(), lr=lr, weight_decay=0.01)
lossfunc = nn.CrossEntropyLoss()

train_losses2 = []
test_losses2 = []
for epoch in range(epochs):
    modeltest.train()
    optimizer.zero_grad()
    outputs = modeltest(x_traint)
    loss = lossfunc(outputs, y_traint)
    loss.backward()
    optimizer.step()
    train_losses2.append(loss.item())
    modeltest.eval()
    with torch.no_grad():
        val_outputs = modeltest(x_testt)
        val_loss = lossfunc(val_outputs, y_testt)
        test_losses2.append(val_loss.item())

modeltest.eval()
with torch.no_grad():
    test_out = modeltest(x_testt)
    _, y_predtest = torch.max(test_out, 1)

    probs = torch.softmax(test_out, 1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=1).numpy()

testpred_labels = labelencoder.inverse_transform(y_predtest.numpy())
testtrue_labels = labelencoder.inverse_transform(y_testt.numpy())

print("-"*50)
test_acc = accuracy_score(testtrue_labels, testpred_labels)
print(f"Test Accuracy: {test_acc:.4f}")
print("Classification Report:")
print(classification_report(testtrue_labels, testpred_labels))

cm = confusion_matrix(testtrue_labels, testpred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labelencoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion test set")
plt.show()

entropyfile = pd.DataFrame({'Entropy': entropy,'Class': testtrue_labels})
'''
if INDICES and OAKONLY==False and GUMCOMB==False and CUSTOM==False:
    results_df = pd.DataFrame({'y_true': testtrue_labels,'y_pred_mlp': testpred_labels})
    results_df.to_csv('mlp_predictions.csv', index=False)
    print("saved to CSV")
'''

if TESTGRAPHS:

    if GRADANALYSIS:
        wav, smoothed_attr = gradanal(modeltest,x,x_testt,y_testt,smooth,left,right,0,savgol=USE_SAVGOL,islabel=INDICES or CUSTOM)
        if INDICES or CUSTOM:
            plt.figure(figsize=(10, 5))
            plt.bar(wav, smoothed_attr, color='purple')
            #plt.xticks(rotation=90)
            plt.xlabel("Index/Band")
            plt.ylabel("Mean Absolute Attribution")
            plt.show()
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(wav, smoothed_attr, color='purple')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Mean Absolute Attribution")
            plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses2, label='Training Loss', color='blue')
    plt.plot(test_losses2, label='Testing Loss', color='red')
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
    sns.boxplot(x='Class', y='Entropy', data=entropyfile, palette="rocket",hue='Class',legend=False)
    #plt.xticks(rotation=45)
    plt.ylabel('Entropy')
    plt.show()
