import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA    
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import savgol_filter
from A_functions import haircut, multiregion,scaling,read_data,cudacheck,gradanal


DIAGNOSTICS=False
device=cudacheck(DIAGNOSTICS)
USE_SCALING = True 
GRADANALYSIS = True
USE_PCA=False 
ncomp = 4
USE_SAVGOL = True
smooth =51
poly=3
HAIRCUT = True 
left = 600
right = 800

MULTIREGION = False
centers = [560,650, 730,860]
width = [32,32,32,26]


neurons1=16
neurons2=8
kernel1=3
kernel2=3  #convolutional kernels
poolkernel=2 #pooling kernel
noisefactor=0
dropprob=0.5 #dropout layer    might be unneccesary
epochs=400
lr=0.001
seed = 49
test_sizeinput = 0.2
torch.manual_seed(seed)
np.random.seed(seed)
x,y=read_data("CSVfiles/datacalibrated.csv")

if HAIRCUT:
    x = haircut(x, left, right)
    print(f"trimmed wav:",x.columns[0],x.columns[-1])
if MULTIREGION:
    x = multiregion(x, centers, width)

labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=test_sizeinput,random_state=seed,stratify=y_encoded)

if USE_SAVGOL:
    x_train = savgol_filter(x_train,window_length=smooth,polyorder=poly, axis=1)
    x_test = savgol_filter(x_test,window_length=smooth,polyorder=poly, axis=1)

if USE_SCALING:
    x_train,x_test=scaling(x_train, x_test)
if USE_PCA:
    pca = PCA(n_components=ncomp)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

idim = x_train.shape[1]  # inputL
odim = len(np.unique(y_encoded))  #output classes
x_traint = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
y_traint = torch.tensor(y_train, dtype=torch.long).to(device)
x_testt = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)
y_testt = torch.tensor(y_test, dtype=torch.long).to(device)

class Oliver(nn.Module):
    def __init__(self, input_L, classes_N):
        super().__init__() 
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=neurons1,kernel_size=kernel1, padding=(kernel1-1)//2)
        self.pool = nn.MaxPool1d(kernel_size=poolkernel) # maxpool or avgpool? 
        self.conv2 =nn.Conv1d(in_channels=neurons1,out_channels=neurons2,kernel_size=kernel2, padding=(kernel2-1)//2)
        self.final_len = input_L//poolkernel**2
        self.flattened = neurons2 * self.final_len

        self.fc = nn.Linear(self.flattened, classes_N)

        self.drop = nn.Dropout1d(dropprob)
        
    def forward(self, x):
        x = torch.tanh(self.conv1(x)) #tanh or Relu?
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)  
        x = self.drop(self.fc(x))
        return x

model = Oliver(idim,odim)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
lossfunc = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_traint)
    loss = lossfunc(outputs, y_traint)
    loss.backward()
    optimizer.step()
    print(epoch) if epoch%10 == 0 else None

noise = torch.randn_like(x_testt)*noisefactor #gaussian noise
x_testn = x_testt + noise

model.eval()
with torch.no_grad():
    test_outputs = model(x_testn)
    _, y_pred = torch.max(test_outputs, 1)

y_pred = y_pred.cpu()
y_testt = y_testt.cpu()

y_prednp = y_pred.numpy()
y_testnp = y_testt.numpy()

predlabels = labelencoder.inverse_transform(y_prednp)
testlabels = labelencoder.inverse_transform(y_testnp)

accuracy = accuracy_score(testlabels, predlabels)
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
    wav, smoothed_attr = gradanal(model,x,x_testt,y_testt, 51,left,right, device,batch=True)
    plt.figure(figsize=(10, 5))
    plt.plot(wav, smoothed_attr, color='purple')
    plt.title("Attributions across dataset")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Importance")
    plt.show()