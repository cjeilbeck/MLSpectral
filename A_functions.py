import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from captum.attr import IntegratedGradients
from scipy.signal import savgol_filter
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde


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
    def M(w, width=32):
        intensity = multispectral(x, [w], width=width, Dif=False)
        return intensity.iloc[:, 0]

    ids = {}
    ids['Green'] = M(560) 
    ids['Red'] = M(650)
    ids['Red-edge'] = M(730)
    ids['NIR'] = M(860, width=52)
    ids['NDVI']= (M(860, width=52)-M(650))/(M(860, width=52)+M(650))
    ids['GNDVI']=(M(860,width=52)-M(560))/(M(860,width=52)+M(560))
    ids['OSAVI']= (M(860, width=52)-M(650))/(M(860, width=52)+M(650)+0.16)
    ids['LCI']= (M(860, width=52)-M(730))/(M(860, width=52)+M(650))
    ids['NDRE']= (M(860, width=52)-M(730))/(M(860, width=52)+M(730))
 
    return pd.DataFrame(ids, index=x.index) 


def indices_single(x, index):
    def M(w, width=32):
        intensity = multispectral(x, [w], width=width, Dif=False)
        return intensity.iloc[:, 0]
    
    ids = {}
    if index == 'NDVI':
        ids['NDVI']= (M(860, width=52)-M(650))/(M(860, width=52)+M(650))
    elif index == 'GNDVI':
        ids['GNDVI']=(M(730)-M(560))/(M(730)+M(560))
    elif index == 'OSAVI':
        ids['OSAVI']= (M(860, width=52)-M(650))/(M(860, width=52)+M(650)+0.16)
    elif index == 'LCI':
        ids['LCI']= (M(860, width=52)-M(730))/(M(860, width=52)+M(650))
    elif index == 'NDRE':
        ids['NDRE']= (M(860, width=52)-M(730))/(M(860, width=52)+M(730))
    elif index == 'Green':
        ids['Green'] = M(560)
    elif index == 'Red':
        ids['Red'] = M(650)
    elif index == 'Red-edge':
        ids['Red-edge'] = M(730)
    elif index == 'NIR':
        ids['NIR'] = M(860, width=26)
    elif index == 'Total Intensity':
        ids['Total Intensity']=M(640, width = 500)
    
    return pd.DataFrame(ids, index=x.index)
    

def indicescustomrf(x):  
    def M(w, width=32):
        intensity = multispectral(x, [w], width=width, Dif=False)
        return intensity.iloc[:, 0]
    green = M(415)
    red = M(545)  
    re = M(675)  
    nir = M(715)

    ids = {}
    
    ids['Green'] = green
    ids['Red'] = red
    ids['Red-edge'] = re
    ids['NIR'] = nir
    ids['NDVI'] = (nir-red) / (nir+red)
    ids['GNDVI'] = (nir -green) / (nir+ green)
    ids['OSAVI'] = (nir-red) / (nir+red + 0.16)
    ids['LCI'] = (nir-re) / (nir +red)
    ids['NDRE'] = (nir-re) / (nir+re)
 
    return pd.DataFrame(ids, index=x.index)

def indicescustomMLP(x):  
    def M(w, width=32):
        intensity = multispectral(x, [w], width=width, Dif=False)
        return intensity.iloc[:, 0]
    green = M(425)
    red = M(485)  
    re = M(665)  
    nir = M(700)

    ids = {}
    
    ids['Green'] = green
    ids['Red'] = red
    ids['Red-edge'] = re
    ids['NIR'] = nir
    ids['NDVI'] = (nir-red) / (nir+red)
    ids['GNDVI'] = (nir -green) / (nir+ green)
    ids['OSAVI'] = (nir-red) / (nir+red + 0.16)
    ids['LCI'] = (nir-re) / (nir +red)
    ids['NDRE'] = (nir-re) / (nir+re)
 
    return pd.DataFrame(ids, index=x.index)

def indicescustomSVM(x):   
    def M(w, width=32):
        intensity = multispectral(x, [w], width=width, Dif=False)
        return intensity.iloc[:, 0]
    green = M(440)
    red = M(555)
    re = M(675)
    nir = M(710)

    ids = {}
    
    ids['Green'] = green
    ids['Red'] = red
    ids['Red-edge'] = re
    ids['NIR'] = nir
    ids['NDVI'] = (nir-red) / (nir+red)
    ids['GNDVI'] = (nir -green) / (nir+ green)
    ids['OSAVI'] = (nir-red) / (nir+red + 0.16)
    ids['LCI'] = (nir-re) / (nir +red)
    ids['NDRE'] = (nir-re) / (nir+re)
 
    return pd.DataFrame(ids, index=x.index)

def indicescustom(x):   
    def M(w, width=32):
        intensity = multispectral(x, [w], width=width, Dif=False)
        return intensity.iloc[:, 0]
    green = M(440)
    red = M(675)
    re = M(710)
    nir = M(820)

    ids = {}
    
    ids['Green'] = green
    ids['Red'] = red
    ids['Red-edge'] = re
    ids['NIR'] = nir
    ids['NDVI'] = (nir-red) / (nir+red)
    ids['GNDVI'] = (nir -green) / (nir+ green)
    ids['OSAVI'] = (nir-red) / (nir+red + 0.16)
    ids['LCI'] = (nir-re) / (nir +red)
    ids['NDRE'] = (nir-re) / (nir+re)
 
    return pd.DataFrame(ids, index=x.index)

def indexplot1D(file, index, thresholds, labels, ax, labelsize, legendsize, truelabels='leaf_type', testing=True):
    
    plot = file.copy()
    min_x = plot[index].min()
    max_x = plot[index].max()
    bins = [min_x] + thresholds + [max_x]
    print(bins)
    plot['predicted'] = pd.cut(plot[index], bins=bins, labels=labels, ordered=False, include_lowest=True)

    if testing:
        t,test = train_test_split(plot, test_size=0.2, random_state=42, stratify = plot['leaf_type'])
        acc = accuracy_score(test['predicted'], test[truelabels])
        print(f"accuracy:{acc:.4f}")
        print(classification_report(test['predicted'], test[truelabels]))

    fig = ax.get_figure()
    sns.histplot(data=plot, x=index, hue='leaf_type', kde=True, palette='Set2',ax=ax)

    colour_map = {
        
        'Young Gum': 'lightsalmon',
        'Old Gum': 'lightblue',
        'Oak': 'lightgreen',
        'Pine': 'thistle',
        'Gum': 'lightblue',
        'Other': 'thistle'

    }

    currentylim = ax.get_ylim()[1]
    for left,right, label in zip(bins[:-1], bins[1:] ,labels):
        ax.axvspan(left,right,color = colour_map[label], alpha = 0.3)
        mid = (left+right)/2
        ax.text(mid, currentylim*0.95, label, ha="center", fontsize = labelsize, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    for thresh in thresholds:
        ax.axvline(x=thresh, color='black', linestyle='--', linewidth=2, label=f'Bound:{thresh:.3f}')

    ax.set_xlabel(index)
    handles, llabels = ax.get_legend_handles_labels()
    if testing:
        ax.legend(handles=handles, fontsize = legendsize, title_fontsize = legendsize, title=f"Test Accuracy: {acc:.3f}", labels=llabels, bbox_to_anchor=(1.05, 1))

    else:
        ax.legend(handles=handles, labels=llabels, bbox_to_anchor=(1.05, 1))
    fig.tight_layout()

    ax.set_xlim(min_x - 0.05*((max_x-min_x)/2), max_x + 0.05*(max_x-min_x)/2) #optional padding but might have to remove for sake of space when 9 plots
    
  
    return ax

def problemtype(file, problemtype):
    data = file.copy()
    if problemtype == 'binary':
        data['leaf_type'] = data['leaf_type'].replace(['Gum_young', 'Gum_old'], 'Gum')
        data['leaf_type'] = data['leaf_type'].replace(['Pine', 'Oakcork'], 'Other')
    elif problemtype == 'gum':
        data = data[data['leaf_type'].isin(['Gum_young', 'Gum_old'])].copy()
        data['leaf_type'] = data['leaf_type'].replace({'Gum_young': 'Young Gum', 'Gum_old': 'Old Gum'})
    elif problemtype == 'oak':
        data['leaf_type'] = data['leaf_type'].replace(['Gum_young', 'Gum_old', 'Pine'], 'Other')
        data['leaf_type'] = data['leaf_type'].replace(['Oakcork'], 'Oak')
    return data


customlabels = {'Gum_old':  'Old Gum',
    'Gum_young': 'Young Gum',
    'Oakcork':  'Oak',
    'Pine':     'Pine'}
    
def KDEpriorfunction(file, index, truelabels='leaf_type'):
    classes = file[truelabels].unique()
    nsamples = len(file[index])

    kdes = {}
    priors = {}

    min = file[index].min()
    max = file[index].max()

    base = np.linspace(min,max,2000)

    for c in classes:
        data = file[file[truelabels]==c][index]
        kdes[c] = gaussian_kde(data)
        print(len(data))
        print(nsamples)
        priors[c] = len(data)/nsamples

    prob = np.zeros((len(classes), len(base)))

    for i, c in enumerate(classes):
        if c in kdes:
            prob[i,:] = kdes[c](base)*priors[c]
        
    best = np.argmax(prob, axis=0)
    boundary = np.where(np.diff(best) !=0)[0]

    boundaries = [base[idx] for idx in boundary] 
    orderlabels = [classes[best[0]]]
    for idx in boundary:
        orderlabels.append(classes[best[idx+1]])

    return boundaries, orderlabels
