import numpy as np
import torch
import sys
from pathlib import Path
cdir = Path(__file__).parent
root = cdir.parent
sys.path.append(str(root))
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import savgol_filter
from FUNCTIONS.A_functions import haircut, indicescustomSVM,read_data,scaling,gradanal, indicesmav, indicescustomMLP, problemtype, indicespaper,indicescustom, customlabels,indicescustomrf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams.update({'font.size': 35})
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from scipy.stats import entropy
import random
import os

plt.rcParams.update({                
    'font.size': 9,           
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1,   
    'axes.linewidth': 0.5,    
    'xtick.direction': 'out',   
    'ytick.direction': 'out',
    'xtick.top': False,         
    'ytick.right': False   
})


USE_SCALING = True
USE_PCA=False
ncomp = 4
USE_SAVGOL = True
smooth =51


#USE CV GRAPHS for cross validation metrics, testgraphs for holdout set metrics

TESTGRAPHS = False
CVGRAPHS = True
CUSTOMBANDPLOTS = True

HAIRCUT = True
left = 206  #this changes accuracy a lot with minor tweaks
right = 910

SAVEFIG = False
name = 'mlp_predictionsOAK.csv'

GRADANALYSIS = True

#indices is commercial data, custom is custom bands; both off for hyperspectral processing
INDICES = False
CUSTOM = False
CUSTOM1 = True
GUMCOMB = False
OAKONLY = False



if GUMCOMB: problem = 'binary'

elif OAKONLY: problem = 'oak'

else: problem = 'all'



if INDICES:
    HAIRCUT = False
    USE_SAVGOL = False

if CUSTOM or CUSTOM1:
    INDICES = False
    HAIRCUT = False
    USE_SAVGOL = False


test_sizeinput = 0.2

#problem = 'gum'

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

elif INDICES:
    
    lr=0.01
    neurons2=64
    neurons1=2*neurons2
    drop = 0.2
    epochs = 121

elif CUSTOM:
    lr=0.001
    neurons2=32
    neurons1=2*neurons2
    drop = 0.2
    epochs = 246 

elif CUSTOM1:
    lr=0.01
    neurons2=64
    neurons1=2*neurons2
    drop = 0.2
    epochs = 64

else:
    epochs = 235
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
if CUSTOM or CUSTOM1:
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


#MODEL

class Brad(nn.Module):
    def __init__(self, input_N, classes_N):
        super().__init__()

        self.layers = torch.nn.Sequential(nn.Linear(input_N, neurons1),nn.Dropout(drop), nn.ReLU(), nn.Linear(neurons1, neurons2), nn.ReLU(),nn.Linear(neurons2, classes_N))
       
    def forward(self, x):
        z = self.layers(x)
        return z



#Cross validation

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

    model.eval()
    with torch.no_grad():
        test_outputs = model(x_valt)
        y_pred = torch.argmax(test_outputs,1)


    #entropy
        probs = torch.softmax(test_outputs, 1)
        shentropy = entropy(probs.numpy(), base=2, axis=1)

    y_prednp = y_pred.numpy()
    y_valtnp = y_valt.numpy()
    predlabels = labelencoder.inverse_transform(y_prednp)
    testlabels = labelencoder.inverse_transform(y_valtnp)

    foldacc = accuracy_score(testlabels, predlabels)
    fold_accuracies.append(foldacc)
    all_trainloss.append(train_losses)
    all_valloss.append(val_losses)
    all_entropy.extend(shentropy)
    all_testlabels.extend(testlabels)
    all_predlabels.extend(predlabels)

    if GRADANALYSIS:
        wav, smoothed_attr = gradanal(model,x,x_valt,y_valt,smooth,left,right,0,savgol=USE_SAVGOL,islabel=INDICES or CUSTOM or CUSTOM1)
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
        std_attr = std_attr/np.sqrt(5)

        if INDICES or CUSTOM or CUSTOM1:
            colors = [(0.4,0.6,0.2,0.8) if i<4 else (0.2,0.4,0.8,0.8) for i in range(len(wav))]
            meanallimportance = mean_attr.mean()
            plt.axhline(meanallimportance, color='black', alpha=0.5,linestyle='--')
            
            plt.bar(wav, mean_attr, yerr=std_attr, color=colors,capsize=10,error_kw={'alpha': 1, 'linewidth': 2})
            plt.xlabel("Spectral Feature")
            plt.ylabel("Importance Score")
            plt.show()
            featurefile = pd.DataFrame({'index': wav,'importance': mean_attr,'std': std_attr})
            if CUSTOM or CUSTOM1:
                featurefile.to_csv('mlp_customfeature_importances.csv', index=False)
            else:
                featurefile.to_csv('mlp_indexfeature_importances.csv', index=False)
        else:
            wav = np.array(wav, dtype=float)
            plt.figure(figsize=(3.5, 2.65),dpi=300)
            plt.plot(wav, mean_attr, color='red')

            if CUSTOMBANDPLOTS:
                centres=[425,555,665,700]
                widths =[32,32,32,32]

                for i, (c, w) in enumerate(zip(centres, widths)):
                    plt.axvspan(c-w/2, c+w/2, facecolor=(1,0.8,0,0.15),edgecolor=(0,0,0,1),linewidth = 0.2,linestyle='--')
        
                    
            plt.fill_between(wav, mean_attr - std_attr, mean_attr + std_attr, alpha=0.2, color='red')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Mean Absolute Attribution")
            plt.tight_layout()
            plt.show()

            wavs = wav
            imps = mean_attr
            half = 16  
            centers = []
            windowtot = []
            for center in np.arange(wavs.min()+half,wavs.max()-half, 1):
                mask = (wavs >= center-half) & (wavs <= center+half)
                centers.append(center)
                windowtot.append(imps[mask].sum())
            centers = np.array(centers)
            windowtot = np.array(windowtot)
            plt.figure(figsize=(15, 5))
            plt.plot(centers, windowtot, color='blue', linewidth=1.5)

    mean_train = np.array(all_trainloss).mean(axis=0)
    std_train = np.array(all_trainloss).std(axis=0)
    mean_val = np.array(all_valloss).mean(axis=0)
    std_val = np.array(all_valloss).std(axis=0)
    std_val = std_val/np.sqrt(5)
    ep = np.arange(epochs)
    plt.figure(figsize=(3.5, 2.65),dpi=300)
    plt.plot(mean_train, color='blue', linestyle='--')
    plt.fill_between(ep, mean_train - std_train, mean_train + std_train, alpha=0.2, color='blue')
    plt.plot(mean_val, color='red')
    plt.fill_between(ep, mean_val - std_val, mean_val + std_val, alpha=0.2, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
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
    y_predtest = torch.argmax(test_out, 1)

    probs = torch.softmax(test_out, 1)
    shentropy = entropy(probs.numpy(), base=2, axis=1)

testpred_labels = labelencoder.inverse_transform(y_predtest.numpy())
testtrue_labels = labelencoder.inverse_transform(y_testt.numpy())

print("-"*50)
test_acc = accuracy_score(testtrue_labels, testpred_labels)
print(f"Test Accuracy: {test_acc:.4f}")
print("Classification Report:")
print(classification_report(testtrue_labels, testpred_labels,digits=4))
print(f"Cohen's Kappa: {cohen_kappa_score(testtrue_labels, testpred_labels):.4f}")



if SAVEFIG:
    results_df = pd.DataFrame({'y_true': testtrue_labels,'y_pred_mlp': testpred_labels})
    results_df.to_csv(name, index=False)
    print("saved to CSV")

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


    entropyfile = pd.DataFrame({'Entropy': shentropy,'Class': testtrue_labels})

    if not OAKONLY and not GUMCOMB:
        entropyfile['Class'] = entropyfile['Class'].map(customlabels)
        displaylabels = [customlabels[c] for c in labelencoder.classes_]

    cm = confusion_matrix(testtrue_labels, testpred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=displaylabels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Class', y='Entropy', data=entropyfile, palette="rocket",hue='Class',legend=False)
    #plt.xticks(rotation=45)
    plt.ylabel('Entropy')
    plt.show()
    
 
"""
    plt.figure(figsize=(8, 5))
    plt.hist(shentropy, bins=10, color='orange', edgecolor='black', alpha=0.7)
    plt.xlabel('Entropy')
    plt.ylabel('Count')
    plt.show()"""