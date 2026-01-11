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

def multiregion(x,centres, width, Dif=True):
    x.columns = pd.to_numeric(x.columns)
    if Dif:
        width=np.array(width)
    rad = width / 2
    all_regions = []
    if Dif:
        for c, r in zip(centres, rad):
            start_wav = c - r
            end_wav = c + r
            region = (x.columns>=start_wav) & (x.columns<=end_wav)
            all_regions.append(region)
    else:
        for c in centres:
            start_wav = c - rad
            end_wav = c + rad
            region = (x.columns>=start_wav) & (x.columns<=end_wav)
            all_regions.append(region)

    final_region = np.logical_or.reduce(all_regions)
    x = x.loc[:,final_region]   
    return x

def multispectral(x,centres,width,Dif=True):
    x.columns = pd.to_numeric(x.columns)
    if Dif:
        width=np.array(width)
    rad = width / 2
    intensities = []
    if Dif:
        for c, r in zip(centres, rad):
            start_wav = c - r
            end_wav = c + r
            region = (x.columns>=start_wav) & (x.columns<=end_wav)
            xwindow = x.loc[:,region]
            intensity = xwindow.sum(axis=1)
            intensities.append(intensity)
    else:
        for c in centres:
            start_wav = c - rad
            end_wav = c + rad
            region = (x.columns>=start_wav) & (x.columns<=end_wav)
            xwindow = x.loc[:,region]
            intensity = xwindow.sum(axis=1)
            intensities.append(intensity)
            
    x=pd.concat(intensities, axis=1, keys=centres)
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



def gradanal(model, x, x_testt, y_testt, smooth, left, right, device, batch=False,savgol=True,islabel=False):
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

        total_attributions = total_attributions/len(x_testt)
        finalat = total_attributions.squeeze().cpu().detach().numpy()
        if savgol:
            smoothed_attr = savgol_filter(finalat, window_length=smooth, polyorder=3)
        else:
            smoothed_attr = finalat

    else:
        ig = IntegratedGradients(model)
        x_testt.requires_grad_()

        attributions = ig.attribute(x_testt, target=y_testt, n_steps=50)

        total_attributions = torch.mean(torch.abs(attributions), dim=0)

        finalat = total_attributions.squeeze().detach().numpy()
        if savgol:
            smoothed_attr = savgol_filter(finalat, window_length=smooth, polyorder=3)
        else:
            smoothed_attr = finalat
    if islabel:
        wav = x.columns
    else:

        wav = pd.to_numeric(x.columns)   
    return wav, smoothed_attr
    


def moving_average(input,window):

    input = pd.Series(input)
    windows = input.rolling(window)

    movingav = windows.mean()
    movingav = movingav.tolist()
    movingav = movingav[window-1:]

    return movingav



def indicespaper(x):

    def M(w):
        intensity = multispectral(x, [w], width=5, Dif=False)
        return intensity.iloc[:, 0]

    ids = {}
    ids['NDVI']= (M(830)-M(650))/(M(830)+M(650))
    ids['ARI1']= (1/M(550))-(1/M(700))
    ids['ARI2']= M(800)*((1/M(550))-(1/M(700)))
    ids['GNDVIhyper1']= (M(750)-M(550))/(M(750)+M(550))
    ids['GNDVIhyper2']= (M(800)-M(550))/(M(800)+M(550))
    ids['mNDVI705']= (M(750)-M(705))/(M(750)+M(705)-2*M(445))
    ids['CCI']= (M(777)-M(747))/M(673)
    ids['VOG2']= (M(734)-M(747))/(M(715)+M(726))
    ids['Carter1']= M(695)/M(420)
    ids['Carter2']= M(695)/M(760)
    ids['Carter3']= M(605)/M(760)
    ids['Carter4']= M(710)/M(760)
    ids['Carter5']= M(695)/M(670)
    ids['Datt1']= (M(850)-M(710))/(M(850)-M(680))
    ids['Datt2']= M(850)/M(710)
    ids['Datt3']= M(754)/M(704)
    ids['EVI']= 2.5*((M(800)-M(670))/(M(800)-6*M(670)-7.5*M(475)+1))
    ids['MCARI']= ((M(700)-M(670))-0.2*(M(700)-M(550)))*(M(700)/M(670))
    ids['MTVI1']= 1.2*(1.2*(M(800)-M(550))-2.5*(M(670)-M(550)))
    ids['NDCI']= (M(762)-M(527))/(M(762)+M(527))
    ids['PSRI']= (M(678)-M(500))/M(750)
    ids['RDVI']= (M(800)-M(670))/np.sqrt(M(800)+M(670))
    ids['REP']= 700+40*((M(670)+M(780))/2-M(700))/(M(740)-M(700))
    ids['SPVI1']= 0.4*3.7*(M(800)-M(670))-1.2*np.abs(M(530)-M(670))
    ids['SRPI']= M(430)/M(680)
    ids['SR_440_690']= M(440)/M(690)
    ids['SR_700_670']= M(700)/M(670)
    ids['SR_750_550']= M(750)/M(550)
    ids['SR_750_700']= M(750)/M(700)
    ids['SR_750_710']= M(750)/M(710)
    ids['SR_752_690']= M(752)/M(690)
    ids['SR_800_680']= M(800)/M(680)
    tcari = 3*((M(700)-M(670))-0.2*(M(700)-M(550))*(M(700)/M(670)))
    osavi = (1+0.16)*(M(800)-M(670))/(M(800)+M(670)+0.16)
    ids['OSAVI'] = osavi
    ids['TCARI'] = tcari
    ids['TCARI_OSAVI'] = tcari / osavi
    ids['TVI']= 0.5*(120*(M(750)-M(550))-200*(M(670)-M(550)))
    ids['LCI']= (M(850)-M(710))/(M(850)-M(680))
    ids['SIPI1']= (M(800)-M(445))/(M(800)-M(680))
    ids['SIPI2']= (M(800)-M(505))/(M(800)-M(690))
    ids['SIPI3']= (M(800)-M(470))/(M(800)-M(680))
    ids['RERVI']= M(840)/M(717)
    ids['RENDVI']= (M(840)-M(717))/(M(840)+M(717))
    ids['GRVI']= M(840)/M(560)
    ids['MTCI']= (M(753)-M(708))/(M(708)-M(681))
    ids['CI_green']= (M(780)/M(550))-1
    ids['RVI']= M(765)/M(720)

    return pd.DataFrame(ids, index=x.index)

def indicesmav(x):   #assuming mavic 3M
    def M(w, width=16):
        intensity = multispectral(x, [w], width=width, Dif=False)
        return intensity.iloc[:, 0]

    ids = {}
    ids['Green'] = M(560) 
    ids['Red'] = M(650)
    ids['Red-edge'] = M(730)
    ids['NIR'] = M(860, width=26)
    ids['NDVI']= (M(860, width=26)-M(650))/(M(860, width=26)+M(650))
    ids['GNDVI']=(M(730)-M(560))/(M(730)+M(560))
    ids['OSAVI']= (M(860, width=26)-M(650))/(M(860, width=26)+M(650)+0.16)
    ids['LCI']= (M(860, width=26)-M(730))/(M(860, width=26)+M(650))
    ids['NDRE']= (M(860, width=26)-M(730))/(M(860, width=26)+M(730))
 
    return pd.DataFrame(ids, index=x.index) 