import pandas as pd
import matplotlib.pyplot as plt


PLOT_ARI1 = False
PLOT_NDVI = True 
PLOT_GNDVI = False 

rad = 16
file = pd.read_csv('CSVfiles/datacalibrated.csv')
data = file.drop(columns=['leaf_type','sample_id'])
wav = pd.to_numeric(data.columns)

def multimean(center):
    start = center-rad
    end = center+rad
    region = (wav>=start) & (wav<=end)
    cols = data.loc[:,region]
    return cols.mean(axis=1)


index = None
name = ""

if PLOT_ARI1:
    R550 =multimean(550)
    R700 =multimean(700)
    
    index = (1/R550)-(1/R700)
    name ="ARI1"  #anthocyanin

elif PLOT_NDVI:

    R830 = multimean(830)
    R650 = multimean(650)
    index = (R830-R650)/(R830+R650)
    name = "NDVI"

elif PLOT_GNDVI:
    R750 = multimean(750)
    R550 = multimean(550)
    
    index = (R750-R550)/(R750+R550)
    name = "GNDVI1" 

file['active_index'] =index
leaf_types = file['leaf_type'].unique()
plot =[file[file['leaf_type']==i]['active_index'] for i in leaf_types]

plt.figure(figsize=(10, 6))
plt.boxplot(plot, labels=leaf_types)
    
plt.title(f'{name} Distribution')
plt.ylabel(f'{name} Value')
plt.xlabel('Leaf type')
    
   
plt.show()