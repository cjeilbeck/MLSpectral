from A_functions import read_data, indicesmav, indices_single
import matplotlib.pyplot as plt
import seaborn as sns

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

x,y = read_data(filename)
indices=indicesmav(x)

plot = indices.copy()

plot['leaf_type']=y



fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

features = ['NDRE','GNDVI','Red-edge']

groups = plot.groupby('leaf_type')
for name, group in groups:
    print(name)
    ax.scatter(group[features[0]], group[features[1]], group[features[2]],label=name,marker='o', alpha=0.7)


plt.legend(['Old Gum', 'Young Gum', 'Oak', 'Pine'], 
           title='Leaf Species')
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_zlabel(features[2])
plt.show()

corr_matrix = plot.corr(method='pearson', numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")        
plt.show()

indices = indices_single(x, 'Total Intensity')
plot = indices.copy()
plot['leaf_type']=y
sns.histplot(data=plot, x='Total Intensity', hue='leaf_type', kde=True, palette='Set2')
plt.show()
