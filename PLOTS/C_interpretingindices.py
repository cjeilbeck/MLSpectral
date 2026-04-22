from pathlib import Path
import sys
cdir = Path(__file__).parent
root = cdir.parent
sys.path.append(str(root))

from FUNCTIONS.A_functions import read_data, indicesmav, indices_single, indicescustomSVM,indexplot1D, problemtype, KDEfunction,customlabels
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

"""  ids['Green'] = M(560) 
    ids['Red'] = M(650)
    ids['Red-edge'] = M(730)
    ids['NIR'] = M(860, width=26)
    ids['NDVI']= (M(860, width=26)-M(650))/(M(860, width=26)+M(650))
    ids['GNDVI']=(M(860)-M(560))/(M(860)+M(560))
    ids['OSAVI']= (M(860, width=26)-M(650))/(M(860, width=26)+M(650)+0.16)
    ids['LCI']= (M(860, width=26)-M(730))/(M(860, width=26)+M(650))
    ids['NDRE']= (M(860, width=26)-M(730))/(M(860, width=26)+M(730))"""

filename = 'CSVfiles/datacalibrated.csv'


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

#plt.figure(figsize=(3.5, 2.65),dpi=300)

x,y = read_data(filename)
indices=indicesmav(x)

plot = indices.copy()

plot['leaf_type']=y



fig = plt.figure(figsize=(3.5,2.65),dpi=300)
ax = fig.add_subplot()

features = ['NDRE','Red-edge']
markers = ['o', 's', '^', 'D']
groups = plot.groupby('leaf_type')
for i ,(name, group) in enumerate(groups):
    marker = markers[i]
    name = customlabels.get(name, name)
    ax.scatter(group[features[0]], group[features[1]],label=name,marker=marker, alpha=0.9,s=2,edgecolors=None)

ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.legend(frameon=False, markerscale=2)
plt.tight_layout()
plt.show()

#blue old gum, orange young gum, green oak, red pine


OPTION =False

if OPTION:

    corr_matrix = plot.corr(method='pearson', numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 30}, fmt=".2f")        
    plt.show()







    indices = indices_single(x, 'Total Intensity')
    plot = indices.copy()
    plot['leaf_type']=y
    sns.histplot(data=plot, x='Total Intensity', hue='leaf_type', kde=True, palette='Set2')
    plt.show()



    """def indicescustomSVM(x):   
        def M(w, width=32):
            intensity = multispectral(x, [w], width=width, Dif=False)
            return intensity.iloc[:, 0]
        green = M(450)
        red = M(555)
        re = M(675)
        nir = M(710)

        ids = {}
        
        ids['Band1'] = green
        ids['Band2'] = red
        ids['Band3'] = re
        ids['Band4'] = nir
        ids['CVI42'] = (nir-red) / (nir+red)
        ids['CVI41'] = (nir -green) / (nir+ green)
        ids['CVI42a'] = (nir-red) / (nir+red + 0.16)
        ids['CVI432'] = (nir-re) / (nir +red)
        ids['CVI43'] = (nir-re) / (nir+re)
    
        return pd.DataFrame(ids, index=x.index)
    """

    indices = indicescustomSVM(x)
    plot = indices.copy()
    features = ['Band2','CVI41']
    plot['leaf_type']=y
    fig2 = plt.figure(figsize=(10, 7))
    ax = fig2.add_subplot()
    groups = plot.groupby('leaf_type')
    for name, group in groups:
        print(name)
        ax.scatter(group[features[0]], group[features[1]],label=name,marker='o', alpha=0.7)


    plt.legend(['Old Gum', 'Young Gum', 'Oak', 'Pine'], 
            title='Leaf Species')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])

    plt.show()


    corr_matrix = plot.corr(method='pearson', numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,annot=True, annot_kws={"size": 30},cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")        
    plt.show()



    """
    INDEXES = ['Band1','Band2','Band3','Band4','CVI42','CVI41','CVI42a','CVI432','CVI43']
    plot = problemtype(plot, 'oak')
    fig, axes = plt.subplots(3,3,figsize=(30,20))
    axes = axes.flatten()
    train, test= train_test_split(plot, test_size=0.2, random_state=42, stratify = plot['leaf_type'])
    for i,index in enumerate(INDEXES):
        thresholds,labels = KDEfunction(train,index)
        indexplot1D(plot,index,thresholds, labels,  ax=axes[i], testing=True)
    plt.subplots_adjust(right=0.9, left=0.05, top=0.95, bottom=0.05,hspace=0.5, wspace=0.5)
    plt.show()
    """