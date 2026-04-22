import numpy as np
from pathlib import Path
import sys
cdir = Path(__file__).parent
root = cdir.parent
sys.path.append(str(root))
import itertools
from FUNCTIONS.A_functions import read_data, indicesmav, indicescustomMLP, indicesmavbands,indicesmavindices, indices_single, customlabels, indicescustomMLP_single, indicescustomrf_single, indicescustomSVM_single
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

INDICES = True

def bhattacharyya_distance(mu1, mu2, sigma1, sigma2):
    mu1, mu2 = np.asarray(mu1), np.asarray(mu2)
    sigma1, sigma2 = np.asarray(sigma1), np.asarray(sigma2)
    
    sigma = (sigma1 + sigma2) / 2.0
    sigma_inv = np.linalg.inv(sigma)
    
    det_sigma = np.linalg.det(sigma)
    det_sigma1 = np.linalg.det(sigma1)
    det_sigma2 = np.linalg.det(sigma2)
    
    diff_mu = mu1 - mu2
    
    term1 = 0.125*np.dot(np.dot(diff_mu.T, sigma_inv), diff_mu)
    term2 = 0.5*np.log(det_sigma/np.sqrt(det_sigma1 * det_sigma2))
    
    return term1 + term2

def jeffries_matusita_distance(mu1, mu2, sigma1, sigma2):
    b_dist = bhattacharyya_distance(mu1, mu2, sigma1, sigma2)
    return 2 * (1-np.exp(-b_dist))

filename = 'CSVfiles/datacalibrated.csv'


x, y = read_data(filename)

results = [] 

indices = indicesmav(x)
plot = indices.copy()
plot['leaf_type'] = y.map(customlabels)
feature_cols = [col for col in plot.columns if col != 'leaf_type']
classes = sorted(plot['leaf_type'].unique())
class_pairs = [(classes[i], classes[j]) for i in range(len(classes)) for j in range(i+1, len(classes))]

results = []

for class1, class2 in class_pairs:
    data1 = plot[plot['leaf_type'] == class1][feature_cols]
    data2 = plot[plot['leaf_type'] == class2][feature_cols]
    mu1, sigma1 = data1.mean().values, data1.cov().values
    mu2, sigma2 = data2.mean().values, data2.cov().values
    dist = jeffries_matusita_distance(mu1, mu2, sigma1, sigma2)

    results.append({ 'Pair': f"{class1}v{class2}", 'JM_Distance': dist})

results_df = pd.DataFrame(results)
print(results_df)

if INDICES:
    bands = ['Green','Red','Red-edge','NIR','NDVI','GNDVI','OSAVI','LCI','NDRE']
else:
    bands = ['Band1','Band2','Band3','Band4','CVI42','CVI41','CVI42a','CVI432','CVI43']
results = []
customlabels = {'Gum_old':  'OG',
    'Gum_young': 'YG',
    'Oakcork':  'O',
    'Pine':     'P'}
for band in bands:
    if INDICES:
        indices = indices_single(x, band)
    else:
        indices = indicescustomMLP_single(x, band)
    plot = indices.copy()
    plot['leaf_type'] = y.map(customlabels)

    feature_cols = [col for col in plot.columns if col != 'leaf_type']
    classes = sorted(plot['leaf_type'].unique())
    class_pairs = [(classes[i], classes[j]) for i in range(len(classes)) for j in range(i+1, len(classes))]

    for class1, class2 in class_pairs:
        data1 = plot[plot['leaf_type'] == class1][feature_cols]
        data2 = plot[plot['leaf_type'] == class2][feature_cols]
        mu1, sigma1 = data1.mean().values, data1.cov().values
        mu2, sigma2 = data2.mean().values, data2.cov().values

        dist = jeffries_matusita_distance(mu1, mu2, sigma1, sigma2)
        results.append({'Band': band,'Pair': f"{class1}v{class2}",'JM_Distance': dist})

df = pd.DataFrame(results)
table = df.pivot(index='Pair', columns='Band', values='JM_Distance')
table = table[bands]  
fig, ax = plt.subplots(figsize=(3.5, 2.65),dpi=300)
sns.heatmap(table,annot=True,annot_kws={"size": 5.2},fmt="#.3g",cmap='Blues',vmin=0, vmax=np.sqrt(2),ax=ax)
ax.set_xlabel('Spectral Feature')
ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
