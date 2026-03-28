
import matplotlib.pyplot as plt
from A_functions import read_data, indicesmav, indexplot1D, problemtype, KDEpriorfunction
from sklearn.model_selection import train_test_split
plt.rcParams.update({'font.size': 12})


filename = 'CSVfiles/datacalibrated.csv'

x,y = read_data(filename)
indices=indicesmav(x)
plot = indices.copy()
plot['leaf_type']=y

BINARY = False
OAK = True
ALL = False
GUM = False
#Binary Plot Production

if BINARY:

    plot_binary = problemtype(plot.copy(),'binary')
    train, test = train_test_split(plot_binary, test_size=0.2, random_state=42, stratify = plot_binary['leaf_type'])

    INDEXES = ['Green','Red','Red-edge','NIR','NDVI','GNDVI','OSAVI','LCI','NDRE']

    fig, axes = plt.subplots(3,3,figsize=(30,20))
    axes = axes.flatten()
    for i,index in enumerate(INDEXES):
        thresholds,labels = KDEpriorfunction(train,index)
        indexplot1D(plot_binary,index,thresholds, labels, labelsize=7, legendsize=9,  ax=axes[i], testing=True)

    plt.subplots_adjust(right=0.9, left=0.05, top=0.95, bottom=0.05,hspace=0.5, wspace=0.5)
    plt.show()


#Oak Plot Production

if OAK:
    plot_oak = problemtype(plot.copy(),'oak')
    train, test = train_test_split(plot_oak, test_size=0.2, random_state=42, stratify = plot_oak['leaf_type'])


    INDEXES = ['Green','Red','Red-edge','NIR','NDVI','GNDVI','OSAVI','LCI','NDRE']

    fig, axes = plt.subplots(3,3,figsize=(30,20))
    axes = axes.flatten()
    for i,index in enumerate(INDEXES):
        thresholds,labels = KDEpriorfunction(train,index)
        indexplot1D(plot_oak,index,thresholds, labels, labelsize=7, legendsize=9,  ax=axes[i], testing=True)

    plt.subplots_adjust(right=0.9, left=0.05, top=0.95, bottom=0.05,hspace=0.5, wspace=0.5)
    plt.show()

#All Plot Production

if ALL:
    plot_all = problemtype(plot.copy(), 'all')
    train, test = train_test_split(plot_all, test_size=0.2, random_state=42, stratify = plot_all['leaf_type'])

    INDEXES = ['Green','Red','Red-edge','NIR','NDVI','GNDVI','OSAVI','LCI','NDRE']

    fig, axes = plt.subplots(3,3,figsize=(30,20))
    axes = axes.flatten()
    for i,index in enumerate(INDEXES):
        thresholds,labels = KDEpriorfunction(train,index)
        indexplot1D(plot_all,index,thresholds, labels, labelsize=7, legendsize=9,  ax=axes[i], testing=True)

    plt.subplots_adjust(right=0.9, left=0.05, top=0.95, bottom=0.05,hspace=0.5, wspace=0.5)
    plt.show()

#Gum Plot Production
if GUM:
    plot_gum = problemtype(plot.copy(), 'gum')
    train, test = train_test_split(plot_gum, test_size=0.2, random_state=42, stratify = plot_gum['leaf_type'])

    INDEXES = ['Green','Red','Red-edge','NIR','NDVI','GNDVI','OSAVI','LCI','NDRE']

    fig, axes = plt.subplots(3,3,figsize=(30,20))
    axes = axes.flatten()
    for i,index in enumerate(INDEXES):
        thresholds,labels = KDEpriorfunction(train,index)
        indexplot1D(plot_gum,index,thresholds, labels, labelsize=7, legendsize=9,  ax=axes[i], testing=True)

    plt.subplots_adjust(right=0.9, left=0.05, top=0.95, bottom=0.05,hspace=0.5, wspace=0.5)
    plt.show()






