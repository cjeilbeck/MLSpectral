import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from captum.attr import IntegratedGradients
from scipy.signal import savgol_filter
import math

def haircut(x,left,right):   
    total_cols = x.shape[1]
    x = x.iloc[: ,left+1:total_cols-right]
    return x

def multiregion(x,centers,width):
    x.columns = pd.to_numeric(x.columns)
    rad = width / 2
    all_regions = []
    for c in centers:
        start_wav = c - rad
        end_wav = c + rad
        region = (x.columns>=start_wav) & (x.columns<=end_wav)
        all_regions.append(region)
    final_region = np.logical_or.reduce(all_regions)
    x = x.loc[:,final_region]
    return x

def scaling(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

def read_data(filename):
    data = pd.read_csv(filename)
    y = data['leaf_type']
    x = data.drop(columns=['leaf_type', 'sample_id'])
    return x, y

def cudacheck(diag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if diag:
        print(f"PyTorch Version:{torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Version: {torch.version.cuda}")
        print(device)
    return device



def gradanal(model, x, x_testt, y_testt, smooth, left, right, device, batch=False):
    ig = IntegratedGradients(model)
    x_testt.requires_grad_()

    if batch:
        input_shape = x_testt.shape[1:]
        total_attributions = torch.zeros(input_shape).to(device)
        size = 32 
        n = math.ceil(len(x_testt)/size)
        print(f"Processing {len(x_testt)} items in {n} batches...")

        for i in range(n):
            start = i * size
            end = min((i + 1) * size, len(x_testt))
            batch_in = x_testt[start:end].to(device)
            labels = y_testt[start:end].to(device)
            batch_in.requires_grad_()
            attrs_batch = ig.attribute(batch_in, target=labels,n_steps=50)
            total_attributions += torch.sum(torch.abs(attrs_batch), dim=0)

        total_attributions = total_attributions/n
        finalat = total_attributions.squeeze().cpu().detach().numpy()
        smoothed_attr = savgol_filter(finalat, window_length=smooth, polyorder=3)

    else:
        ig = IntegratedGradients(model)
        x_testt.requires_grad_()

        attributions = ig.attribute(x_testt, target=y_testt, n_steps=50)

        total_attributions = torch.mean(torch.abs(attributions), dim=0)

        finalat = total_attributions.squeeze().detach().numpy()
        smoothed_attr = savgol_filter(finalat, window_length=smooth, polyorder=3)

    wav = pd.to_numeric(x.columns)   
    return wav, smoothed_attr
    

