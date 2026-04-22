import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CUSTOM = False
INDICES = True

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


if CUSTOM:
    rf  = pd.read_csv('IMPORTANCES/RFcustomimportances.csv',  index_col=0)
    svm = pd.read_csv('IMPORTANCES/svm_customfeatureweights.csv',  index_col=0)
    mlp = pd.read_csv('IMPORTANCES/mlp_customfeature_importances.csv', index_col=0)
if INDICES:
    rf  = pd.read_csv('IMPORTANCES/RFindeximportances.csv',  index_col=0)
    svm = pd.read_csv('IMPORTANCES/svm_index_importances.csv',  index_col=0)
    mlp = pd.read_csv('IMPORTANCES/mlp_indexfeature_importances.csv', index_col=0)

means = pd.DataFrame({
    'RF':  rf['importance'],
    'SVM':  svm['importance'],
    'MLP':  mlp['importance'],
})
stds = pd.DataFrame({
    'RF':  rf['std'],
    'SVM':  svm['std'],
    'MLP':  mlp['std'],
})

scale = means.sum()
means_n = means / scale
err = stds /np.sqrt(5)
stds_n  = err  / scale

colors  = ['#0072B2', '#E69F00', '#009E73']

#blue RF, yellow SVM, green MLP

n_models = len(means_n.columns)
n_features = len(means_n.index)
fig, ax = plt.subplots(figsize=(3.5, 2.65),dpi=300)  


offsets = [-0.15, 0, 0.15]
markers = ['o', 's', 'D']

for i, (model, color, offset, marker) in enumerate(zip(means_n.columns, colors, offsets, markers)):
    x = [j + offset for j in range(n_features)]
    ax.errorbar(x, means_n[model], yerr=stds_n[model],fmt=marker, markersize = 2, color=color,label=model, linestyle='none', zorder=3,capsize=1,capthick=0.5,elinewidth=0.8)

for j in range(n_features):
    y_vals = [means_n[model].iloc[j] for model in means_n.columns]
    ax.vlines(j, min(y_vals), max(y_vals),color='gray', alpha=0.1, zorder=1)

ax.set_xticks(range(n_features))
ax.set_xticklabels(means_n.index, rotation=45, ha='right')
ax.set_ylabel('Normalised importance')
ax.legend(frameon=False, fontsize=7, markerscale=1.5)
if CUSTOM:
    ax.set_xlabel('Spectral Features')
elif INDICES:
    ax.set_xlabel('Spectral Features')


plt.tight_layout()
plt.show()